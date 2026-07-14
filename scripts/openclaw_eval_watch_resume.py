#!/usr/bin/env python3
"""Resume detached OpenClaw copy/eval watchers without model-token use."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


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


def _api_request(
    base_url: str,
    method: str,
    path: str,
    *,
    timeout_seconds: float = 60.0,
) -> Any:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=b"{}" if method == "POST" else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


JobKey = tuple[str, str]


def watcher_owned_jobs(state: dict[str, Any]) -> set[JobKey]:
    """Cluster-scoped eval jobs owned by watcher workflows.

    The sweep must not resume these: live workflows resume their own evals
    with progress-based stall detection, and a stalled chain was stopped on
    purpose. Slurm job ids are cluster-local, so a bare id is never a valid
    ownership key.
    """
    owned: set[JobKey] = set()
    for value in state.values():
        if not isinstance(value, dict):
            continue
        if (
            value.get("eval_terminal_notified_at")
            and value.get("outcome") != "eval_resume_stalled"
        ):
            continue
        cluster = str(value.get("dest_cluster") or "").strip()
        if not cluster:
            continue
        for key in ("eval_job_id", "eval_resume_parent_job_id"):
            job_id = str(value.get(key) or "")
            if job_id:
                owned.add((cluster, job_id))
        for hop in value.get("eval_resume_chain") or []:
            if isinstance(hop, dict):
                for key in ("old_job_id", "new_job_id"):
                    job_id = str(hop.get(key) or "")
                    if job_id:
                        owned.add((cluster, job_id))
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


def _is_exhausted_requeue_detail(row: dict[str, Any]) -> bool:
    state = str(row.get("State") or row.get("state") or "").strip().upper()
    exit_code = str(row.get("ExitCode") or row.get("exit_code") or "").strip()
    try:
        restarts = int(row.get("Restarts") or row.get("restarts") or 0)
    except (TypeError, ValueError):
        return False
    return state == "CANCELLED" and restarts >= 5 and exit_code in {"", "0:0"}


def sweep_timed_out_evals(
    *,
    api_base: str,
    clusters: list[str],
    hours: int,
    name_prefix: str,
    max_chain: int,
    owned_jobs: set[JobKey],
    notifier: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resume interrupted evals that no watcher workflow owns.

    Relies on the backend's idempotent resume endpoint (one child per parent),
    so a rerun of the sweep recovers instead of duplicating. TIMEOUT chains
    retain the safety cap. Exact ``CANCELLED`` jobs with >=5 Slurm restarts are
    automatic background-preemption exhaustion, not an intentional scancel;
    those are resubmitted even after the timeout chain cap.
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
            is_timeout = state.startswith("TIMEOUT")
            is_requeue_cancel = state.startswith("CANCEL")
            if not (is_timeout or is_requeue_cancel):
                continue
            if not job_name.startswith(name_prefix):
                continue
            if phase not in (None, "eval"):
                continue
            if (cluster, job_id) in owned_jobs or job_id in resumed_parents:
                continue
            cluster_q = urllib.parse.quote(cluster, safe="")
            job_q = urllib.parse.quote(job_id, safe="")
            if is_requeue_cancel:
                try:
                    detail = _api_request(
                        api_base, "GET", f"/api/jobs/{cluster_q}/{job_q}"
                    )
                except Exception as exc:
                    errors.append({
                        "cluster": cluster,
                        "job_id": job_id,
                        "error": f"cancel classification failed: {exc}",
                    })
                    continue
                if not isinstance(detail, dict) or not _is_exhausted_requeue_detail(detail):
                    continue
            if is_timeout and _chain_depth(job_id, rows_by_id) >= max_chain:
                errors.append({
                    "cluster": cluster,
                    "job_id": job_id,
                    "error": f"auto-resume chain reached {max_chain}; needs a human look",
                })
                continue
            try:
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
            if notifier is not None:
                try:
                    reason = (
                        "Slurm exhausted 5 automatic requeues"
                        if is_requeue_cancel
                        else "Slurm TIMEOUT"
                    )
                    notifier(
                        f"♻️ Eval resubmitted after {reason}: "
                        f"`{cluster}/{job_id}` → "
                        f"`{cluster}/{str((response or {}).get('job_id') or '')}`."
                    )
                except Exception as exc:
                    errors.append({
                        "cluster": cluster,
                        "job_id": job_id,
                        "error": f"Slack notification failed: {exc}",
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_workflow_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"workflow state is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("workflow state root must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sweep_is_running(state_dir: Path) -> bool:
    lease_path = state_dir / "eval-watch-resume.sweep.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open("a+") as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
    return False


def run_sweep_worker(args: argparse.Namespace) -> int:
    """Run one potentially slow cluster sweep outside the cron time budget."""
    state_file = Path(args.state_file).expanduser().resolve()
    state_dir = state_file.parent
    health_file = Path(
        args.sweep_health_file or state_dir / "eval-watch-resume.health.json"
    ).expanduser()
    lease_path = state_dir / "eval-watch-resume.sweep.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open("a+") as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(json.dumps({"status": "already_running"}, sort_keys=True))
            return 0

        previous_health = _read_json(health_file)
        started_at = _now_iso()
        clusters = [c.strip() for c in args.sweep_clusters.split(",") if c.strip()]
        try:
            state = _read_workflow_state(state_file)
            swept, errors = sweep_timed_out_evals(
                api_base=args.api_base,
                clusters=clusters,
                hours=args.sweep_hours,
                name_prefix=args.sweep_name_prefix,
                max_chain=args.sweep_max_chain,
                owned_jobs=watcher_owned_jobs(state),
                notifier=lambda message: notify_slack(args.slack_channel, message),
            )
        except Exception as exc:
            swept, errors = [], [{"worker": f"unexpected sweep failure: {exc}"}]
        status = "error" if errors else "ok"
        health = {
            "status": status,
            "started_at": started_at,
            "finished_at": _now_iso(),
            "swept": swept,
            "errors": errors,
        }

        # Alert on a new/changed failure and once when service recovers. The
        # scheduler remains at one-minute cadence because this detached worker,
        # not the cron supervisor, owns slow network operations and health.
        error_fingerprint = json.dumps(errors, sort_keys=True)
        previous_fingerprint = str(previous_health.get("error_fingerprint") or "")
        failure_alert_recorded = error_fingerprint == previous_fingerprint
        try:
            if errors and error_fingerprint != previous_fingerprint:
                notify_slack(
                    args.slack_channel,
                    "⚠️ Eval recovery sweep failed: " + "; ".join(
                        str(item.get("error") or item) for item in errors
                    ),
                )
                health["failure_notified_at"] = _now_iso()
                failure_alert_recorded = True
            elif not errors and previous_health.get("status") == "error":
                notify_slack(args.slack_channel, "✅ Eval recovery sweep recovered.")
                health["recovery_notified_at"] = _now_iso()
        except Exception as exc:
            health["health_notification_error"] = str(exc)
        if errors and failure_alert_recorded:
            health["error_fingerprint"] = error_fingerprint
        _write_json(health_file, health)
        print(json.dumps(health, sort_keys=True))
        return 1 if errors else 0


def spawn_sweep_worker(args: argparse.Namespace, log_path: Path) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--sweep-worker",
        "--state-file",
        str(Path(args.state_file).expanduser().resolve()),
        "--api-base",
        args.api_base,
        "--slack-channel",
        args.slack_channel,
        "--sweep-clusters",
        args.sweep_clusters,
        "--sweep-hours",
        str(args.sweep_hours),
        "--sweep-name-prefix",
        args.sweep_name_prefix,
        "--sweep-max-chain",
        str(args.sweep_max_chain),
    ]
    if args.sweep_health_file:
        command.extend(["--sweep-health-file", args.sweep_health_file])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return child.pid


def notify_slack(channel: str, message: str) -> None:
    result = subprocess.run(
        [
            "openclaw",
            "message",
            "send",
            "--channel",
            "slack",
            "--target",
            channel,
            "--message",
            message,
        ],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout).strip() or "Slack notification failed"
        )


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
    parser.add_argument("--sweep-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sweep-health-file")
    args = parser.parse_args()

    if args.sweep_worker:
        return run_sweep_worker(args)

    state_file = Path(args.state_file).expanduser().resolve()
    worker = Path(args.worker).expanduser().resolve()
    state = _read_workflow_state(state_file)

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

    sweep_worker: dict[str, Any] = {"status": "disabled"}
    if not args.no_sweep:
        if sweep_is_running(state_file.parent):
            sweep_worker = {"status": "running"}
        else:
            try:
                pid = spawn_sweep_worker(
                    args, state_file.parent / "eval-watch-resume.sweep.log"
                )
                sweep_worker = {"status": "started", "pid": pid}
            except Exception as exc:
                errors.append({"sweep_worker": str(exc)})
                sweep_worker = {"status": "failed"}
    health_path = Path(
        args.sweep_health_file
        or state_file.parent / "eval-watch-resume.health.json"
    ).expanduser()
    print(json.dumps({
        "resumed": resumed,
        "sweep_worker": sweep_worker,
        "sweep_health": _read_json(health_path),
        "errors": errors,
    }, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
