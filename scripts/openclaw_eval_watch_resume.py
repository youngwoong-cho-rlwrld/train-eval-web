#!/usr/bin/env python3
"""Resume detached OpenClaw copy/eval watchers without model-token use."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


RESUMABLE_OUTCOMES = {
    "copying_checkpoint",
    "verifying_checkpoint",
    "eval_submitting",
    "eval_submit_uncertain",
    "eval_submitted",
    # --replace-existing-eval clears eval_job_id after cancelling the old job;
    # a crash there must still resubmit (the worker never replaces on resume).
    "eval_replacing",
}


def pending_workflows(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    pending = []
    for state_key, value in state.items():
        if not isinstance(value, dict):
            continue
        legacy_timeout = (
            str(value.get("eval_state") or "").upper().startswith("TIMEOUT")
            and value.get("outcome") != "eval_resume_stalled"
        )
        if value.get("eval_terminal_notified_at") and not legacy_timeout:
            continue
        if value.get("eval_job_id") or value.get("outcome") in RESUMABLE_OUTCOMES:
            pending.append((state_key, value))
    return pending


def source_parts(state_key: str, entry: dict[str, Any]) -> tuple[str, str]:
    if "/" not in state_key:
        raise ValueError(f"invalid state key: {state_key}")
    key_cluster, key_job = state_key.split("/", 1)
    return str(entry.get("source_cluster") or key_cluster), key_job


def build_worker_command(
    *,
    worker: Path,
    state_file: Path,
    state_key: str,
    entry: dict[str, Any],
    api_base: str,
    slack_channel: str,
) -> list[str]:
    request_id = str(entry.get("request_id") or "")
    variant = str(entry.get("variant") or "")
    dest_cluster = str(entry.get("dest_cluster") or "")
    partition = str(entry.get("dest_partition") or "")
    if not all((request_id, variant, dest_cluster, partition)):
        raise ValueError(f"incomplete resumable state entry: {state_key}")
    source_cluster, source_job = source_parts(state_key, entry)
    copy_id = str(entry.get("copy_id") or f"direct-{request_id}")
    command = [
        sys.executable,
        str(worker),
        "--copy-id",
        copy_id,
        "--request-id",
        request_id,
        "--state-key",
        state_key,
        "--source-cluster",
        source_cluster,
        "--source-job-id",
        source_job,
        "--variant",
        variant,
        "--dest-cluster",
        dest_cluster,
        "--partition",
        partition,
        "--api-base",
        api_base,
        "--state-file",
        str(state_file),
        "--slack-channel",
        slack_channel,
        "--resume-timed-out",
    ]
    if entry.get("delete_source"):
        command.append("--delete-source")
    if entry.get("copy_skipped"):
        checkpoint_path = str(
            entry.get("dest_checkpoint_path") or entry.get("checkpoint_path") or ""
        ).strip()
        if not checkpoint_path:
            raise ValueError(f"direct eval checkpoint is missing: {state_key}")
        command.extend(["--skip-copy", "--checkpoint-path", checkpoint_path])
    dexjoco_task = str(entry.get("dexjoco_task") or "").strip()
    if dexjoco_task:
        command.extend(["--dexjoco-task", dexjoco_task])
    return command


def _api_request(base_url: str, method: str, path: str) -> Any:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=b"{}" if method == "POST" else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def watcher_owned_job_ids(state: dict[str, Any]) -> set[str]:
    """Eval job ids any watcher workflow owns (including stalled chains).

    The sweep must not resume these: live workflows resume their own evals
    with progress-based stall detection, and a stalled chain was stopped on
    purpose.
    """
    owned: set[str] = set()
    for value in state.values():
        if not isinstance(value, dict):
            continue
        for key in ("eval_job_id", "eval_resume_parent_job_id"):
            job_id = str(value.get(key) or "")
            if job_id:
                owned.add(job_id)
        for hop in value.get("eval_resume_chain") or []:
            if isinstance(hop, dict):
                for key in ("old_job_id", "new_job_id"):
                    job_id = str(hop.get(key) or "")
                    if job_id:
                        owned.add(job_id)
    return owned


def _chain_depth(job_id: str, rows_by_id: dict[str, dict[str, Any]]) -> int:
    depth = 0
    seen = {job_id}
    parent = str(rows_by_id.get(job_id, {}).get("resume_of") or "")
    while parent and parent in rows_by_id and parent not in seen:
        depth += 1
        seen.add(parent)
        parent = str(rows_by_id[parent].get("resume_of") or "")
    return depth


def sweep_timed_out_evals(
    *,
    api_base: str,
    clusters: list[str],
    hours: int,
    name_prefix: str,
    max_chain: int,
    owned_job_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resume TIMEOUT evals that no watcher workflow owns.

    Relies on the backend's idempotent resume endpoint (one child per parent),
    so a rerun of the sweep recovers instead of duplicating. The chain-depth
    cap is the stateless stand-in for the watcher's progress-based stall stop.
    """
    swept: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for cluster in clusters:
        query = urllib.parse.urlencode({"cluster": cluster, "hours": hours})
        try:
            payload = _api_request(api_base, "GET", f"/api/jobs?{query}")
        except Exception as exc:
            errors.append({"cluster": cluster, "error": f"job list failed: {exc}"})
            continue
        rows = payload.get("jobs", []) if isinstance(payload, dict) else []
        rows_by_id = {
            str(row.get("job_id") or ""): row
            for row in rows
            if isinstance(row, dict) and row.get("job_id")
        }
        resumed_parents = {
            str(row.get("resume_of") or "")
            for row in rows_by_id.values()
            if row.get("resume_of")
        }
        for job_id, row in sorted(rows_by_id.items()):
            job_name = str(row.get("job_name") or "")
            state = str(row.get("state") or "").upper()
            phase = row.get("phase")
            if not state.startswith("TIMEOUT"):
                continue
            if not job_name.startswith(name_prefix):
                continue
            if phase not in (None, "eval"):
                continue
            if job_id in owned_job_ids or job_id in resumed_parents:
                continue
            if _chain_depth(job_id, rows_by_id) >= max_chain:
                errors.append({
                    "cluster": cluster,
                    "job_id": job_id,
                    "error": f"auto-resume chain reached {max_chain}; needs a human look",
                })
                continue
            try:
                cluster_q = urllib.parse.quote(cluster, safe="")
                job_q = urllib.parse.quote(job_id, safe="")
                response = _api_request(
                    api_base, "POST", f"/api/jobs/{cluster_q}/{job_q}/resume"
                )
            except Exception as exc:
                errors.append({"cluster": cluster, "job_id": job_id, "error": str(exc)})
                continue
            swept.append({
                "cluster": cluster,
                "job_id": job_id,
                "resumed_as": str((response or {}).get("job_id") or ""),
                "recovered": bool((response or {}).get("recovered")),
            })
    return swept, errors


def watcher_is_running(state_dir: Path, request_id: str) -> bool:
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in request_id)
    lease_path = state_dir / f"eval-copy-{safe_id or 'request'}.watch.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open("a+") as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-file", default="~/.openclaw/workspace/state/train-watch.json"
    )
    parser.add_argument(
        "--worker",
        default=str(Path(__file__).with_name("openclaw_eval_copy_watch.py")),
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--slack-channel", default="channel:C0BETH2BDV3")
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="skip resuming TIMEOUT evals that were submitted outside openclaw",
    )
    parser.add_argument("--sweep-clusters", default="kakao,skt")
    parser.add_argument("--sweep-hours", type=int, default=48)
    parser.add_argument("--sweep-name-prefix", default="youngwoong_eval_")
    parser.add_argument(
        "--sweep-max-chain",
        type=int,
        default=3,
        help="stop auto-resuming a job after this many chained resumes",
    )
    args = parser.parse_args()

    state_file = Path(args.state_file).expanduser().resolve()
    worker = Path(args.worker).expanduser().resolve()
    try:
        state = json.loads(state_file.read_text())
    except FileNotFoundError:
        state = {}

    resumed = []
    errors = []
    for state_key, entry in pending_workflows(state):
        request_id = str(entry.get("request_id") or "")
        if not request_id or watcher_is_running(state_file.parent, request_id):
            continue
        try:
            command = build_worker_command(
                worker=worker,
                state_file=state_file,
                state_key=state_key,
                entry=entry,
                api_base=args.api_base,
                slack_channel=args.slack_channel,
            )
            log_path = state_file.parent / f"eval-copy-{request_id}.log"
            with log_path.open("ab", buffering=0) as log:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            resumed.append({"request_id": request_id, "pid": child.pid})
        except Exception as exc:
            errors.append({"state_key": state_key, "error": str(exc)})

    swept: list[dict[str, Any]] = []
    if not args.no_sweep:
        clusters = [c.strip() for c in args.sweep_clusters.split(",") if c.strip()]
        swept, sweep_errors = sweep_timed_out_evals(
            api_base=args.api_base,
            clusters=clusters,
            hours=args.sweep_hours,
            name_prefix=args.sweep_name_prefix,
            max_chain=args.sweep_max_chain,
            owned_job_ids=watcher_owned_job_ids(state),
        )
        errors.extend(sweep_errors)
    print(json.dumps({"resumed": resumed, "swept": swept, "errors": errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
