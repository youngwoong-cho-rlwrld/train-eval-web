#!/usr/bin/env python3
"""Finish an OpenClaw-approved checkpoint-copy -> Slurm eval workflow.

OpenClaw starts this process with ``--detach`` immediately after the copy API
returns a copy id.  The detached child waits without consuming model tokens,
verifies the persisted copy record, and submits exactly one eval job.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ALLOWED_EVAL_TARGETS = {
    "kakao": {"background", "h100"},
    "skt": {"l40s-gpu_background", "rlwrld-gpu_background"},
}
EVAL_TERMINAL_FAILURE_PREFIXES = (
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "ERROR",
)


class ApiError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, payload)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise ApiError(exc.code, f"{method} {path}: {body or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiError(None, f"{method} {path}: {exc}") from exc
        return json.loads(raw) if raw else None


@contextmanager
def locked_state(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            state = json.loads(path.read_text()) if path.exists() else {}
            yield state
            tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def workflow_lease(state_path: Path, request_id: str):
    """Allow only one detached watcher per Slack request."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in request_id)
    lease_path = state_path.parent / f"eval-copy-{safe_id or 'request'}.watch.lock"
    with lease_path.open("a+") as lease:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def read_entry(path: Path, state_key: str, request_id: str) -> dict[str, Any]:
    with locked_state(path) as state:
        entry = state.get(state_key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"state entry not found: {state_key}")
        if entry.get("request_id") != request_id:
            raise RuntimeError(f"request id mismatch for {state_key}")
        return dict(entry)


def update_entry(
    path: Path, state_key: str, request_id: str, **values: Any
) -> dict[str, Any]:
    with locked_state(path) as state:
        entry = state.get(state_key)
        if not isinstance(entry, dict) or entry.get("request_id") != request_id:
            raise RuntimeError(f"state entry/request mismatch: {state_key}")
        entry.update(values)
        return dict(entry)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
            f"Slack notification failed: "
            f"{(result.stderr or result.stdout).strip() or result.returncode}"
        )


def notify_with_retry(
    notifier: Callable[[str, str], None],
    channel: str,
    message: str,
    *,
    sleep: Callable[[float], None],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            notifier(channel, message)
            return
        except Exception:
            if time.monotonic() >= deadline:
                raise
            sleep(min(30.0, max(0.1, deadline - time.monotonic())))


def copy_history_path(source_cluster: str, source_job_id: str) -> str:
    cluster = urllib.parse.quote(source_cluster, safe="")
    job = urllib.parse.quote(source_job_id, safe="")
    return f"/api/jobs/{cluster}/{job}/checkpoint-copies"


def find_copy_record(
    api: ApiClient, source_cluster: str, source_job_id: str, copy_id: str
) -> dict[str, Any] | None:
    records = api.get(copy_history_path(source_cluster, source_job_id))
    return next((record for record in records if record.get("copy_id") == copy_id), None)


def wait_for_verified_copy(
    *,
    api: ApiClient,
    copy_id: str,
    source_cluster: str,
    source_job_id: str,
    dest_cluster: str,
    delete_source: bool,
    poll_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "copy did not produce a persisted record"
    while time.monotonic() < deadline:
        status: dict[str, Any] | None = None
        try:
            status = api.get(f"/api/copy-jobs/{urllib.parse.quote(copy_id, safe='')}")
        except ApiError as exc:
            # Copy state is in memory, but history is persisted. A backend
            # restart can therefore make this endpoint return 404 after a
            # successful copy; history below remains authoritative.
            if exc.status != 404:
                last_error = str(exc)

        if status and status.get("status") == "error":
            raise RuntimeError(status.get("error") or "checkpoint copy failed")

        try:
            record = find_copy_record(api, source_cluster, source_job_id, copy_id)
        except ApiError as exc:
            record = None
            last_error = str(exc)

        if record:
            if record.get("dest_cluster") != dest_cluster:
                raise RuntimeError("copy destination does not match the Slack selection")
            if record.get("dest_exists") is not True:
                last_error = "copied checkpoint is not present on destination"
            elif delete_source and record.get("source_exists") is not False:
                last_error = "source checkpoint deletion is not yet verified"
            else:
                return record

        if status and status.get("status") == "done" and not record:
            last_error = "copy is done but its persisted history is missing"
        sleep(poll_seconds)

    raise TimeoutError(last_error)


def _job_value(job: dict[str, Any], lower: str, upper: str) -> Any:
    value = job.get(lower)
    return job.get(upper) if value is None else value


def _job_id(job: dict[str, Any]) -> str:
    return str(_job_value(job, "job_id", "JobID") or "")


def _job_id_order(job: dict[str, Any]) -> tuple[int, int | str]:
    raw = _job_id(job)
    return (0, int(raw)) if raw.isdigit() else (1, raw)


def find_existing_eval(api: ApiClient, cluster: str, job_name: str) -> dict[str, Any] | None:
    """Find the original exact-name job in list API output.

    ``/api/jobs`` returns lowercase Pydantic fields, while the detail endpoint
    returns Slurm's uppercase fields. Accept both deliberately. If historical
    damage left duplicates, recover the oldest job rather than choosing an
    arbitrary row or creating yet another one. Terminal jobs count too: an
    explicit replacement is a separate user action.
    """
    query = urllib.parse.urlencode({"cluster": cluster, "hours": 168})
    payload = api.get(f"/api/jobs?{query}")
    rows = payload.get("jobs", []) if isinstance(payload, dict) else []
    matches = [
        job
        for job in rows
        if isinstance(job, dict)
        and _job_value(job, "job_name", "JobName") == job_name
        and _job_id(job)
    ]
    return min(matches, key=_job_id_order) if matches else None


def wait_for_existing_eval(
    *,
    api: ApiClient,
    cluster: str,
    job_name: str,
    poll_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    """Reconcile a submission whose HTTP response may have been lost."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        existing = find_existing_eval(api, cluster, job_name)
        if existing is not None or time.monotonic() >= deadline:
            return existing
        sleep(poll_seconds)


def _seconds_since(value: Any) -> float | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
        if stamp.tzinfo is None:
            stamp = stamp.astimezone()
        return max(0.0, (datetime.now().astimezone() - stamp).total_seconds())
    except (TypeError, ValueError):
        return None


def _idempotency_key(request_id: str) -> str:
    safe = "".join(
        c if c.isalnum() or c in "_.:-" else "_" for c in request_id
    ).strip("_.:-")
    return f"openclaw:{(safe or 'request')[:140]}"


def eval_terminal_kind(state: str) -> str | None:
    upper = (state or "").upper().split("+")[0]
    if upper.startswith("COMPLET") and upper != "COMPLETING":
        return "completed"
    if upper.startswith("CANCEL"):
        return "cancelled"
    if upper.startswith("TIMEOUT"):
        return "timeout"
    if upper.startswith(EVAL_TERMINAL_FAILURE_PREFIXES):
        return "failed"
    return None


def is_exhausted_requeue_job(job: dict[str, Any]) -> bool:
    state = str(job.get("State") or job.get("state") or "").strip().upper()
    exit_code = str(job.get("ExitCode") or job.get("exit_code") or "").strip()
    try:
        restarts = int(job.get("Restarts") or job.get("restarts") or 0)
    except (TypeError, ValueError):
        return False
    return state == "CANCELLED" and restarts >= 5 and exit_code in {"", "0:0"}


def eval_interruption_kind(job: dict[str, Any]) -> str | None:
    """Classify only terminal states that policy allows us to resubmit."""
    state = str(job.get("State") or job.get("state") or "")
    kind = eval_terminal_kind(state)
    if kind == "timeout":
        return "timeout"
    if kind == "cancelled" and is_exhausted_requeue_job(job):
        return "requeue_exhausted"
    return None


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _state_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item for item in value.split() if item]
    return []


def eval_run_progress(
    api: ApiClient,
    *,
    cluster: str,
    job_id: str,
    variant: str,
    entry: dict[str, Any],
) -> tuple[int, int | None]:
    """Return cumulative completed runs and the configured run total.

    Resumed evals reuse/seed the original output namespace, so the eval-runs
    endpoint is cumulative across the resume chain.  State overrides win when
    present; otherwise the experiment config remains the source of truth.
    """
    cluster_q = urllib.parse.quote(cluster, safe="")
    job_q = urllib.parse.quote(job_id, safe="")
    payload = api.get(f"/api/jobs/{cluster_q}/{job_q}/eval-runs")
    rows = payload.get("eval_runs", []) if isinstance(payload, dict) else []
    completed = len(rows) if isinstance(rows, list) else 0

    saved_total = entry.get("eval_total_runs")
    if saved_total is not None:
        try:
            return completed, int(saved_total)
        except (TypeError, ValueError):
            pass

    try:
        variant_q = urllib.parse.quote(variant, safe="")
        config = api.get(f"/api/variants/{variant_q}")
    except ApiError:
        return completed, None
    if not isinstance(config, dict):
        return completed, None

    arrays = config.get("arrays") if isinstance(config.get("arrays"), dict) else {}
    scalars = config.get("vars") if isinstance(config.get("vars"), dict) else {}
    tasks = _state_list(entry.get("eval_tasks"))
    if not tasks:
        tasks = _state_list(arrays.get("TASKS"))
    # A per-task DexJoCo submission intentionally overrides a multitask config.
    task_count = 1 if entry.get("dexjoco_task") else max(1, len(tasks))
    eval_sets = _state_list(entry.get("eval_sets"))
    if not eval_sets:
        eval_sets = _state_list(arrays.get("EVAL_SETS"))
    set_count = max(1, len(eval_sets))
    run_count = _positive_int(entry.get("eval_n_runs") or scalars.get("N_RUNS"))
    return completed, task_count * set_count * run_count


def _resumed_job_id(job: dict[str, Any]) -> str:
    return str(job.get("job_id") or job.get("JobID") or "")


def _oldest_resumed_job(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    jobs_with_ids = [row for row in rows if isinstance(row, dict) and _resumed_job_id(row)]
    if not jobs_with_ids:
        return None
    return min(
        jobs_with_ids,
        key=lambda row: (
            0,
            int(_resumed_job_id(row)),
        )
        if _resumed_job_id(row).isdigit()
        else (1, _resumed_job_id(row)),
    )


def resume_interrupted_eval(
    args: argparse.Namespace,
    *,
    api: ApiClient,
    notifier: Callable[[str, str], None],
    sleep: Callable[[float], None],
    job: dict[str, Any],
    interruption: str,
) -> dict[str, Any] | None:
    """Resume one scheduler interruption under the matching safety policy.

    TIMEOUT is an allocation-budget failure, so repeated windows with no
    completed runs stop for human inspection. Requeue exhaustion is a sequence
    of scheduler preemptions; zero completed runs is expected and must not turn
    an automatic recovery into a permanent stall.
    """
    if interruption not in {"timeout", "requeue_exhausted"}:
        raise ValueError(f"unsupported eval interruption: {interruption}")
    state_path = Path(args.state_file).expanduser()
    entry = read_entry(state_path, args.state_key, args.request_id)
    parent_id = str(entry.get("eval_job_id") or "")
    if not parent_id:
        raise RuntimeError("timed-out eval job id is missing from workflow state")

    completed, total = eval_run_progress(
        api,
        cluster=args.dest_cluster,
        job_id=parent_id,
        variant=args.variant,
        entry=entry,
    )
    parent_q = urllib.parse.quote(parent_id, safe="")
    cluster_q = urllib.parse.quote(args.dest_cluster, safe="")
    resume_path = f"/api/jobs/{cluster_q}/{parent_q}/resume"
    resumes_path = f"/api/jobs/{cluster_q}/{parent_q}/resumes"

    # Reconcile first. This covers a lost HTTP response or watcher crash after
    # sbatch succeeded but before the new job id reached local workflow state.
    child = _oldest_resumed_job(api.get(resumes_path))
    attempt_in_flight = (
        entry.get("outcome") == "eval_resuming"
        and str(entry.get("eval_resume_parent_job_id") or "") == parent_id
    )
    previous_completed = entry.get("eval_resume_last_completed_runs")
    if (
        interruption == "timeout"
        and child is None
        and not attempt_in_flight
        and previous_completed is not None
    ):
        try:
            made_progress = completed > int(previous_completed)
        except (TypeError, ValueError):
            made_progress = True
        if not made_progress:
            total_text = str(total) if total is not None else "?"
            message = (
                f"⚠️ Eval auto-resume stopped: `{args.dest_cluster}/{parent_id}` "
                f"was interrupted with no new completed runs ({completed}/{total_text})."
            )
            update_entry(
                state_path,
                args.state_key,
                args.request_id,
                outcome="eval_resume_stalled",
                eval_state=str(job.get("State") or job.get("state") or "").upper(),
                eval_terminal_at=job.get("End") or job.get("end") or now_iso(),
                eval_total_runs=total,
                eval_resume_stalled_at=now_iso(),
            )
            notify_with_retry(
                notifier,
                args.slack_channel,
                message,
                sleep=sleep,
                timeout_seconds=args.slack_timeout_seconds,
            )
            return update_entry(
                state_path,
                args.state_key,
                args.request_id,
                eval_terminal_notified_at=now_iso(),
            )

    if child is None:
        update_entry(
            state_path,
            args.state_key,
            args.request_id,
            outcome="eval_resuming",
            eval_state=str(job.get("State") or job.get("state") or "").upper(),
            eval_terminal_at=job.get("End") or job.get("end") or now_iso(),
            eval_resume_parent_job_id=parent_id,
            eval_resume_last_completed_runs=completed,
            eval_total_runs=total,
            eval_resume_attempted_at=now_iso(),
            eval_terminal_notified_at=None,
        )
        try:
            response = api.post(resume_path, {})
            child = response if isinstance(response, dict) else None
        except ApiError:
            # The request is ambiguous. Reconcile once more; if the sidecar is
            # not visible yet, leave eval_resuming persisted and fail closed.
            child = _oldest_resumed_job(api.get(resumes_path))
            if child is None:
                raise

    child_id = _resumed_job_id(child or {})
    if not child_id:
        raise RuntimeError(f"resume API returned no child job for {parent_id}")
    child_name = str((child or {}).get("job_name") or (child or {}).get("JobName") or "")
    entry = read_entry(state_path, args.state_key, args.request_id)
    chain = list(entry.get("eval_resume_chain") or [])
    if not any(str(item.get("new_job_id")) == child_id for item in chain if isinstance(item, dict)):
        chain.append(
            {
                "old_job_id": parent_id,
                "new_job_id": child_id,
                "completed_runs": completed,
                "total_runs": total,
                "resumed_at": now_iso(),
            }
        )
    update_entry(
        state_path,
        args.state_key,
        args.request_id,
        outcome="eval_resumed",
        eval_job_id=child_id,
        eval_job_name=child_name or entry.get("eval_job_name"),
        eval_resume_chain=chain,
        eval_resume_parent_job_id=parent_id,
        eval_resume_last_completed_runs=completed,
        eval_total_runs=total,
        eval_resumed_at=now_iso(),
        eval_state=None,
        eval_terminal_at=None,
        eval_terminal_notified_at=None,
        eval_monitor_error=None,
        eval_monitor_failed_at=None,
        eval_monitor_failure_notified_at=None,
    )
    total_text = str(total) if total is not None else "?"
    notify_with_retry(
        notifier,
        args.slack_channel,
        (
            f"♻️ Eval resumed: `{args.dest_cluster}/{parent_id}` → "
            f"`{args.dest_cluster}/{child_id}`, {completed}/{total_text} runs done."
        ),
        sleep=sleep,
        timeout_seconds=args.slack_timeout_seconds,
    )
    return read_entry(state_path, args.state_key, args.request_id)


def wait_for_eval_terminal(
    *,
    api: ApiClient,
    cluster: str,
    job_id: str,
    poll_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], None],
) -> tuple[dict[str, Any], str]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "eval did not reach a terminal state"
    cluster_q = urllib.parse.quote(cluster, safe="")
    job_q = urllib.parse.quote(job_id, safe="")
    while time.monotonic() < deadline:
        try:
            job = api.get(f"/api/jobs/{cluster_q}/{job_q}")
        except ApiError as exc:
            last_error = str(exc)
        else:
            state = str(job.get("State") or job.get("state") or "")
            kind = eval_terminal_kind(state)
            if kind:
                return job, kind
            last_error = f"eval is still {state or 'unknown'}"
        sleep(poll_seconds)
    raise TimeoutError(last_error)


def monitor_and_report_eval(
    args: argparse.Namespace,
    *,
    api: ApiClient,
    notifier: Callable[[str, str], None],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    state_path = Path(args.state_file).expanduser()
    while True:
        entry = read_entry(state_path, args.state_key, args.request_id)
        job_id = str(entry.get("eval_job_id") or "")
        if not job_id:
            raise RuntimeError("eval job id is missing from workflow state")
        if entry.get("eval_terminal_notified_at"):
            print(f"eval terminal state already reported: {job_id}", flush=True)
            return entry

        try:
            job, kind = wait_for_eval_terminal(
                api=api,
                cluster=args.dest_cluster,
                job_id=job_id,
                poll_seconds=args.eval_poll_seconds,
                timeout_seconds=args.eval_timeout_seconds,
                sleep=sleep,
            )
            interruption = eval_interruption_kind(job)
            if interruption is not None:
                resumed = resume_interrupted_eval(
                    args,
                    api=api,
                    notifier=notifier,
                    sleep=sleep,
                    job=job,
                    interruption=interruption,
                )
                if resumed and resumed.get("outcome") == "eval_resume_stalled":
                    return resumed
                continue
        except Exception as exc:
            failed_entry = update_entry(
                state_path,
                args.state_key,
                args.request_id,
                eval_monitor_error=str(exc),
                eval_monitor_failed_at=now_iso(),
            )
            if not failed_entry.get("eval_monitor_failure_notified_at"):
                notify_with_retry(
                    notifier,
                    args.slack_channel,
                    f"⚠️ Eval status monitoring failed for `{args.dest_cluster}/{job_id}` — {exc}",
                    sleep=sleep,
                    timeout_seconds=args.slack_timeout_seconds,
                )
                update_entry(
                    state_path,
                    args.state_key,
                    args.request_id,
                    eval_monitor_failure_notified_at=now_iso(),
                )
            raise
        break

    state = str(job.get("State") or job.get("state") or "").upper()
    if kind == "completed":
        outcome = "eval_completed"
        message = (
            f"✅ Eval completed: `{args.dest_cluster}/{job_id}` "
            f"`{args.variant}` on `{args.partition}`."
        )
    elif kind == "cancelled":
        outcome = "eval_cancelled"
        message = (
            f"🚫 Eval cancelled: `{args.dest_cluster}/{job_id}` "
            f"`{args.variant}` ({state})."
        )
    else:
        outcome = "eval_failed"
        message = (
            f"❌ Eval failed: `{args.dest_cluster}/{job_id}` "
            f"`{args.variant}` ({state})."
        )

    # Mark the terminal state first, but mark notification delivery only after
    # Slack accepts it. A restarted watcher can therefore retry safely.
    update_entry(
        state_path,
        args.state_key,
        args.request_id,
        outcome=outcome,
        eval_state=state,
        eval_terminal_at=job.get("End") or job.get("end") or now_iso(),
    )
    notify_with_retry(
        notifier,
        args.slack_channel,
        message,
        sleep=sleep,
        timeout_seconds=args.slack_timeout_seconds,
    )
    return update_entry(
        state_path,
        args.state_key,
        args.request_id,
        eval_terminal_notified_at=now_iso(),
    )


def run_workflow(
    args: argparse.Namespace,
    *,
    api: ApiClient | None = None,
    notifier: Callable[[str, str], None] = notify_slack,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if args.dest_cluster not in ALLOWED_EVAL_TARGETS:
        raise RuntimeError("evaluation destination must be kakao or skt")
    if args.partition not in ALLOWED_EVAL_TARGETS[args.dest_cluster]:
        raise RuntimeError(
            f"partition {args.partition!r} is not allowed on {args.dest_cluster}"
        )

    state_path = Path(args.state_file).expanduser()
    api = api or ApiClient(args.api_base, timeout=args.api_timeout_seconds)
    entry = read_entry(state_path, args.state_key, args.request_id)
    if entry.get("eval_job_id"):
        if (
            getattr(args, "resume_timed_out", False)
            and str(entry.get("eval_state") or "").upper().startswith("TIMEOUT")
            and entry.get("outcome") != "eval_resume_stalled"
        ):
            entry = update_entry(
                state_path,
                args.state_key,
                args.request_id,
                outcome="eval_timeout",
                eval_terminal_notified_at=None,
            )
        if not args.replace_existing_eval:
            print(f"eval already submitted: {entry['eval_job_id']}", flush=True)
            return monitor_and_report_eval(
                args, api=api, notifier=notifier, sleep=sleep
            )
        replaced_job_id = str(entry["eval_job_id"])
        api.delete(f"/api/jobs/{args.dest_cluster}/{replaced_job_id}")
        update_entry(
            state_path,
            args.state_key,
            args.request_id,
            outcome="eval_replacing",
            replaced_eval_job_id=replaced_job_id,
            eval_job_id=None,
            eval_job_name=None,
            eval_submit_uncertain_at=None,
            eval_submit_error=None,
            eval_replaced_at=now_iso(),
        )

    # The task may arrive via CLI on the first launch or via state on resume;
    # DexJoCo variants without DEXJOCO_TASK in config.sh hard-fail /api/submit
    # without it, so it must survive restarts.
    dexjoco_task = (
        str(args.dexjoco_task or "").strip()
        or str(entry.get("dexjoco_task") or "").strip()
    )
    if dexjoco_task and entry.get("dexjoco_task") != dexjoco_task:
        update_entry(
            state_path, args.state_key, args.request_id, dexjoco_task=dexjoco_task
        )

    if args.skip_copy:
        if args.source_cluster != args.dest_cluster:
            raise RuntimeError("--skip-copy requires the same source/destination cluster")
        if args.delete_source:
            raise RuntimeError("--delete-source is invalid with --skip-copy")
        checkpoint_path = (args.checkpoint_path or "").strip()
        if not checkpoint_path:
            raise RuntimeError("--checkpoint-path is required with --skip-copy")
        update_entry(
            state_path,
            args.state_key,
            args.request_id,
            outcome="verifying_checkpoint",
            copy_id=args.copy_id,
            copy_skipped=True,
            delete_source=False,
            dest_cluster=args.dest_cluster,
            dest_partition=args.partition,
            copy_watcher_started_at=now_iso(),
        )
        try:
            query = urllib.parse.urlencode({"path": checkpoint_path})
            path_status = api.get(
                f"/api/clusters/{urllib.parse.quote(args.dest_cluster, safe='')}/path-exists?{query}"
            )
            if not path_status.get("exists") or path_status.get("kind") != "dir":
                raise RuntimeError(
                    f"checkpoint directory does not exist on {args.dest_cluster}: "
                    f"{checkpoint_path}"
                )
        except Exception as exc:
            failed_entry = update_entry(
                state_path,
                args.state_key,
                args.request_id,
                outcome="checkpoint_verification_failed",
                checkpoint_verification_error=str(exc),
                checkpoint_verification_failed_at=now_iso(),
            )
            if not failed_entry.get("checkpoint_verification_failure_notified_at"):
                notify_with_retry(
                    notifier,
                    args.slack_channel,
                    f"❌ Checkpoint verification failed for `{args.variant}` — {exc}",
                    sleep=sleep,
                    timeout_seconds=args.slack_timeout_seconds,
                )
                update_entry(
                    state_path,
                    args.state_key,
                    args.request_id,
                    checkpoint_verification_failure_notified_at=now_iso(),
                )
            raise
        record = {
            "dest_path": checkpoint_path,
            "source_exists": True,
            "dest_exists": True,
        }
    else:
        if args.checkpoint_path:
            raise RuntimeError("--checkpoint-path is only valid with --skip-copy")
        update_entry(
            state_path,
            args.state_key,
            args.request_id,
            outcome="copying_checkpoint",
            copy_id=args.copy_id,
            copy_skipped=False,
            delete_source=args.delete_source,
            dest_cluster=args.dest_cluster,
            dest_partition=args.partition,
            copy_watcher_started_at=now_iso(),
        )

        try:
            record = wait_for_verified_copy(
                api=api,
                copy_id=args.copy_id,
                source_cluster=args.source_cluster,
                source_job_id=args.source_job_id,
                dest_cluster=args.dest_cluster,
                delete_source=args.delete_source,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
                sleep=sleep,
            )
        except Exception as exc:
            failed_entry = update_entry(
                state_path,
                args.state_key,
                args.request_id,
                outcome="copy_failed",
                copy_error=str(exc),
                copy_failed_at=now_iso(),
            )
            if not failed_entry.get("copy_failure_notified_at"):
                notify_with_retry(
                    notifier,
                    args.slack_channel,
                    f"❌ Checkpoint copy failed: `{args.copy_id}` — {exc}",
                    sleep=sleep,
                    timeout_seconds=args.slack_timeout_seconds,
                )
                update_entry(
                    state_path,
                    args.state_key,
                    args.request_id,
                    copy_failure_notified_at=now_iso(),
                )
            raise

    entry = read_entry(state_path, args.state_key, args.request_id)
    job_name = entry.get("eval_job_name")
    if not job_name:
        # The backend dedups evals on cluster+job_name, so two same-variant
        # evals submitted within the same wall-clock second must not share a
        # name; a request-derived token keeps names unique per workflow. The
        # variant parser tolerates the suffix (known-variant prefix match).
        token = hashlib.sha256(args.request_id.encode()).hexdigest()[:6]
        job_name = (
            f"youngwoong_eval_{args.variant}_{token}_"
            f"{datetime.now():%Y%m%d_%H%M%S}"
        )
    update_entry(
        state_path,
        args.state_key,
        args.request_id,
        outcome="eval_submitting",
        copy_finished_at=now_iso(),
        dest_checkpoint_path=record["dest_path"],
        source_checkpoint_exists=record.get("source_exists"),
        dest_checkpoint_exists=record.get("dest_exists"),
        eval_job_name=job_name,
    )

    existing = find_existing_eval(api, args.dest_cluster, job_name)
    if existing:
        response = {
            "job_id": _job_id(existing),
            "job_name": job_name,
            "recovered": True,
        }
    else:
        entry = read_entry(state_path, args.state_key, args.request_id)
        uncertain_age = _seconds_since(entry.get("eval_submit_uncertain_at"))
        if (
            uncertain_age is not None
            and uncertain_age < args.submit_retry_grace_seconds
        ):
            update_entry(
                state_path,
                args.state_key,
                args.request_id,
                outcome="eval_submit_uncertain",
                eval_reconciled_at=now_iso(),
            )
            raise RuntimeError(
                "previous eval submission response is uncertain; refusing to "
                "resubmit before the reconciliation grace period expires"
            )

        attempt_count = int(entry.get("eval_submit_attempt_count") or 0) + 1
        update_entry(
            state_path,
            args.state_key,
            args.request_id,
            outcome="eval_submitting",
            eval_submit_attempt_count=attempt_count,
            eval_submit_attempted_at=now_iso(),
        )
        payload = {
            "cluster": args.dest_cluster,
            "variant": args.variant,
            "phase": "eval",
            "partition": args.partition,
            "checkpoint_path": record["dest_path"],
            "job_name": job_name,
            "idempotency_key": _idempotency_key(args.request_id),
        }
        if dexjoco_task:
            payload["dexjoco_task"] = dexjoco_task
        try:
            response = api.post("/api/submit", payload)
        except ApiError as exc:
            # 4xx means the backend definitely rejected the request. A timeout,
            # transport error, or 5xx is ambiguous: sbatch may already have
            # succeeded. Persist that distinction before doing anything else.
            if exc.status is not None and 400 <= exc.status < 500:
                failed_entry = update_entry(
                    state_path,
                    args.state_key,
                    args.request_id,
                    outcome="eval_submit_failed",
                    eval_submit_error=str(exc),
                    eval_submit_failed_at=now_iso(),
                )
                if not failed_entry.get("eval_submit_failure_notified_at"):
                    notify_with_retry(
                        notifier,
                        args.slack_channel,
                        f"❌ Eval submission rejected for `{args.variant}` — {exc}",
                        sleep=sleep,
                        timeout_seconds=args.slack_timeout_seconds,
                    )
                    update_entry(
                        state_path,
                        args.state_key,
                        args.request_id,
                        eval_submit_failure_notified_at=now_iso(),
                    )
                raise

            uncertain_at = now_iso()
            entry = update_entry(
                state_path,
                args.state_key,
                args.request_id,
                outcome="eval_submit_uncertain",
                eval_submit_error=str(exc),
                eval_submit_uncertain_at=uncertain_at,
            )
            try:
                recovered = wait_for_existing_eval(
                    api=api,
                    cluster=args.dest_cluster,
                    job_name=job_name,
                    poll_seconds=args.submit_reconcile_poll_seconds,
                    timeout_seconds=args.submit_reconcile_seconds,
                    sleep=sleep,
                )
            except ApiError as reconcile_exc:
                recovered = None
                update_entry(
                    state_path,
                    args.state_key,
                    args.request_id,
                    eval_reconcile_error=str(reconcile_exc),
                    eval_reconciled_at=now_iso(),
                )
            if recovered is None:
                if not entry.get("eval_submit_uncertain_notified_at"):
                    notify_with_retry(
                        notifier,
                        args.slack_channel,
                        (
                            f"⚠️ Eval submission response was lost for `{args.variant}`. "
                            "No second job will be submitted until Slurm is "
                            "reconciled safely."
                        ),
                        sleep=sleep,
                        timeout_seconds=args.slack_timeout_seconds,
                    )
                    update_entry(
                        state_path,
                        args.state_key,
                        args.request_id,
                        eval_submit_uncertain_notified_at=now_iso(),
                    )
                raise
            response = {
                "job_id": _job_id(recovered),
                "job_name": job_name,
                "recovered": True,
            }

    job_id = str(response["job_id"])
    update_entry(
        state_path,
        args.state_key,
        args.request_id,
        outcome="eval_submitted",
        eval_job_id=job_id,
        eval_job_name=response.get("job_name", job_name),
        eval_submitted_at=now_iso(),
        eval_submit_recovered=bool(response.get("recovered")),
        eval_submit_uncertain_at=None,
        eval_submit_error=None,
    )
    action = "recovered" if response.get("recovered") else "submitted"
    checkpoint_message = (
        "✅ Checkpoint verified."
        if args.skip_copy
        else f"✅ Checkpoint copy `{args.copy_id}` completed and verified."
    )
    notify_with_retry(
        notifier,
        args.slack_channel,
        (
            f"{checkpoint_message} "
            f"🚀 Eval {action} on {args.dest_cluster} `{args.partition}`: "
            f"job `{job_id}`."
        ),
        sleep=sleep,
        timeout_seconds=args.slack_timeout_seconds,
    )
    print(json.dumps({"copy_id": args.copy_id, "eval_job_id": job_id}), flush=True)
    return monitor_and_report_eval(args, api=api, notifier=notifier, sleep=sleep)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-id", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--state-key", required=True)
    parser.add_argument("--source-cluster", required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dest-cluster", choices=sorted(ALLOWED_EVAL_TARGETS), required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--dexjoco-task")
    parser.add_argument("--replace-existing-eval", action="store_true")
    parser.add_argument(
        "--resume-timed-out",
        action="store_true",
        help="reopen a TIMEOUT that an older watcher already reported as terminal",
    )
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--api-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--state-file", default="~/.openclaw/workspace/state/train-watch.json"
    )
    parser.add_argument("--slack-channel", default="channel:C0BETH2BDV3")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--eval-poll-seconds", type=float, default=30.0)
    parser.add_argument("--eval-timeout-seconds", type=float, default=172800.0)
    parser.add_argument("--submit-reconcile-poll-seconds", type=float, default=10.0)
    parser.add_argument("--submit-reconcile-seconds", type=float, default=180.0)
    parser.add_argument("--submit-retry-grace-seconds", type=float, default=300.0)
    parser.add_argument("--slack-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--log-file")
    return parser


def spawn_detached(args: argparse.Namespace) -> int:
    log_path = Path(
        args.log_file
        or f"~/.openclaw/workspace/state/eval-copy-{args.copy_id}.log"
    ).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_args = [arg for arg in sys.argv[1:] if arg != "--detach"]
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *child_args],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    print(json.dumps({"pid": child.pid, "log_file": str(log_path)}), flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.detach:
        return spawn_detached(args)
    state_path = Path(args.state_file).expanduser()
    with workflow_lease(state_path, args.request_id) as acquired:
        if not acquired:
            print(f"watcher already running for request {args.request_id}", flush=True)
            return 0
        try:
            run_workflow(args)
        except Exception as exc:
            print(f"workflow failed: {exc}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
