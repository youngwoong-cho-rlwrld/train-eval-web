"""MLXP (k8s) jobs: list / detail / cancel / log streaming via kubectl.

Keeps the same shape as slurm's `jobs.py` so the existing /jobs and
/jobs/<cluster>/<id> UI consumes both without branching.

Conventions:
  - Job names produced by mlxp_submit are `youngwoong-train-<variant>-<ts>`.
  - Jobs are labelled `owner=youngwoong,tool=train-eval-web`. We list by label.
  - State synthesized from job.status counts (Pending/Running/Succeeded/Failed).
"""

import asyncio
import json
import shutil
from datetime import datetime, timezone
from typing import Any

from .jobs import Job


_NS = "p-rlwrld"


async def _kubectl_json(*args: str, timeout: float = 20.0) -> dict[str, Any]:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        "kubectl", *args, "-o", "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {stderr.decode(errors='replace').strip()}")
    return json.loads(stdout.decode())


def _state_from_pod_and_job(job_status: dict, pod: dict | None) -> str:
    """Derive a slurm-like state. Pod phase is authoritative when a pod
    exists: k8s Job's `status.active` counts both Pending and Running pods,
    so it shows 1 even when the pod is unscheduled — bad for the UI.
    Falls back to job counters only when no pod is present."""
    if pod:
        phase = ((pod.get("status") or {}).get("phase") or "").strip()
        if phase == "Pending":
            return "PENDING"
        if phase == "Running":
            return "RUNNING"
        if phase == "Succeeded":
            return "COMPLETED"
        if phase == "Failed":
            return "FAILED"
        # Unknown / empty: drop to job-level
    succeeded = int(job_status.get("succeeded") or 0)
    failed = int(job_status.get("failed") or 0)
    if succeeded > 0:
        return "COMPLETED"
    if failed > 0:
        return "FAILED"
    return "PENDING"


def _elapsed(start: str | None, end: str | None) -> str:
    if not start:
        return "0:00"
    try:
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
    except ValueError:
        return "0:00"
    e = datetime.now(timezone.utc)
    if end:
        try:
            e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            pass
    delta = int((e - s).total_seconds())
    h, rem = divmod(delta, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


async def list_jobs() -> list[Job]:
    """All MLXP jobs we know about: live k8s Jobs + archived runs derived
    from DDN checkpoint directories.

    k8s Jobs are GC'd 30 min after completion (`ttlSecondsAfterFinished`),
    so a Jobs page opened later wouldn't show them. We fall back to listing
    `<exp_dir>/checkpoints/<job-name>` on DDN — every completed training
    run leaves one of those behind for the lifetime of the checkpoints.
    """
    live = await _list_live_jobs()
    archived = await _list_archived_jobs(seen={j.job_name for j in live})
    return live + archived


async def _list_live_jobs() -> list[Job]:
    try:
        data = await _kubectl_json("get", "jobs", "-n", _NS, "-l", "owner=youngwoong")
    except RuntimeError:
        return []

    # We need pod-level info too (nodeName + start time of the actual pod, not the Job).
    pods_by_job: dict[str, dict] = {}
    try:
        pod_data = await _kubectl_json("get", "pods", "-n", _NS, "-l", "owner=youngwoong")
        for p in pod_data.get("items", []):
            labels = p.get("metadata", {}).get("labels", {})
            jn = labels.get("job-name") or labels.get("batch.kubernetes.io/job-name")
            if jn:
                pods_by_job[jn] = p
    except RuntimeError:
        pass

    out: list[Job] = []
    for j in data.get("items", []):
        name = j["metadata"]["name"]
        annotations = (j["metadata"].get("annotations") or {})
        display_name = annotations.get("train-eval-web/display-name") or name
        status = j.get("status", {}) or {}
        pod = pods_by_job.get(name)
        state = _state_from_pod_and_job(status, pod)
        start = status.get("startTime")
        end = status.get("completionTime")
        elapsed = _elapsed(start, end)

        nodelist = (pod or {}).get("spec", {}).get("nodeName") or "(unscheduled)"

        out.append(Job(
            cluster="mlxp", job_id=name, job_name=display_name, partition="mlxp",
            state=state, elapsed=elapsed, nodelist=nodelist,
            start=start, end=end,
        ))
    return out


async def _list_archived_jobs(seen: set[str]) -> list[Job]:
    """Scan DDN for completed runs whose k8s Job has been GC'd."""
    import asyncio
    from .mlxp_data_pod import ensure_listing_pod, NAMESPACE

    try:
        pod = await ensure_listing_pod()
    except Exception:
        return []

    # One liner per run: <variant>|<job_name>|<latest_step>|<start_epoch>|<end_epoch>
    # NOTE: array glob with nullglob, not `ls $glob` — `ls` with zero-arg
    # falls back to listing `.` which used to slip past the empty check and
    # produced fake rows (`experiment_cfg`, `logs`, `runs`) in the table.
    script = r"""
shopt -s nullglob
for d in /data/youngwoong/experiments/*/checkpoints/*/; do
    matches=( "$d"checkpoint-* )
    [ ${#matches[@]} -eq 0 ] && continue
    job_name=$(basename "$d")
    variant=$(basename "$(dirname "$(dirname "$d")")")
    latest=$(for m in "${matches[@]}"; do basename "$m" | sed 's:^checkpoint-::'; done | sort -n | tail -1)
    start_epoch=$(stat -c %Y "$d" 2>/dev/null || echo 0)
    end_epoch=$(stat -c %Y "$d"checkpoint-$latest 2>/dev/null || echo "$start_epoch")
    printf '%s|%s|%s|%s|%s\n' "$variant" "$job_name" "$latest" "$start_epoch" "$end_epoch"
done
"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", NAMESPACE, pod, "--", "bash", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
    except Exception:
        return []

    out: list[Job] = []
    for line in stdout.decode(errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        variant, job_name, latest, start_e, end_e = parts
        # The shell script only emits rows where checkpoint-N exists, so
        # `latest` is always a number for real runs. Anything else means the
        # shell got confused; drop it.
        if not latest.isdigit():
            continue
        if job_name in seen:
            continue
        try:
            start_iso = datetime.fromtimestamp(int(start_e), tz=timezone.utc).isoformat()
            end_iso = datetime.fromtimestamp(int(end_e), tz=timezone.utc).isoformat()
        except ValueError:
            continue
        elapsed = _elapsed(start_iso, end_iso)
        out.append(Job(
            cluster="mlxp", job_id=job_name, job_name=job_name, partition="mlxp",
            state="COMPLETED", elapsed=elapsed, nodelist="(archived)",
            start=start_iso, end=end_iso,
        ))
    return out


async def get_job(name: str) -> dict[str, Any]:
    """Return a slurm-sacct-shaped dict for one MLXP job.

    Falls back to the archived-on-DDN view when the k8s Job is gone
    (TTL'd), so the job-detail page works for completed runs too.
    """
    try:
        job_data = await _kubectl_json("get", "job", name, "-n", _NS)
    except RuntimeError:
        archived = await _archived_record(name)
        if archived:
            return archived
        raise FileNotFoundError(f"mlxp job not found: {name}")
    status = job_data.get("status", {}) or {}
    start = status.get("startTime") or ""
    end = status.get("completionTime") or ""

    pod_data = await _kubectl_json("get", "pods", "-n", _NS, "-l", f"job-name={name}")
    pods = pod_data.get("items", [])
    pod = pods[0] if pods else {}
    pod_status = pod.get("status", {}) or {}
    pod_name = pod.get("metadata", {}).get("name", "")
    node = pod.get("spec", {}).get("nodeName", "") or ""
    state = _state_from_pod_and_job(status, pod if pods else None)

    # Per-container exit code if available
    exit_code = ""
    for cs in pod_status.get("containerStatuses", []) or []:
        term = (cs.get("state") or {}).get("terminated") or (cs.get("lastState") or {}).get("terminated")
        if term:
            exit_code = str(term.get("exitCode", ""))
            break

    # Reason: phase / conditions
    reason = pod_status.get("phase", "") or ""
    for c in (pod_status.get("conditions") or []):
        if c.get("status") == "False" and c.get("reason"):
            reason = c["reason"]
            break

    display_name = (
        ((job_data.get("metadata") or {}).get("annotations") or {})
        .get("train-eval-web/display-name") or name
    )
    return {
        "JobID": name,
        "JobName": display_name,
        "Partition": "mlxp",
        "State": state,
        "ExitCode": exit_code,
        "Start": start,
        "End": end,
        "Elapsed": _elapsed(start, end),
        "NodeList": node or pod_name,
        "Reason": reason,
        "cluster": "mlxp",
        # Extra (for log streaming endpoint)
        "_pod_name": pod_name,
    }


async def cancel_job(name: str) -> None:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "delete", "job", name, "-n", _NS,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl delete failed: {stderr.decode(errors='replace').strip()}")


async def tail_logs(job_name: str, follow: bool = True):
    """Stream log lines for an MLXP job.

    Primary source is `kubectl logs` on the job's pod. Once a job's k8s
    record has been GC'd we fall back to the on-DDN log file the body
    script left at `<exp_dir>/checkpoints/logs/training_rank0.log`.
    """
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    pod_data = await _kubectl_json("get", "pods", "-n", _NS, "-l", f"job-name={job_name}")
    pods = pod_data.get("items", [])
    if not pods:
        async for line in _tail_archived_log(job_name):
            yield line
        return
    pod_name = pods[0]["metadata"]["name"]

    # No --tail: stream the full log from the start. Frontend handles the
    # scrollback; backend just delivers everything kubectl will emit.
    args = ["kubectl", "logs", "-n", _NS, pod_name]
    if follow:
        args.append("-f")
    # 1MB stream limit: tqdm progress lines use \r instead of \n, and at
    # 64KB (asyncio's default) a long-running tqdm bar overflows readline.
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 20,
    )
    assert proc.stdout is not None
    try:
        async for line in _iter_logical_lines(proc.stdout):
            yield line
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _tail_archived_log(job_name: str):
    """Serve <variant>/checkpoints/logs/training_rank0.log via the data pod,
    holding the connection open with tail -F so EventSource doesn't loop."""
    import shlex
    from .details import parse_phase_and_variant
    from .mlxp_data_pod import ensure_listing_pod

    _, variant = parse_phase_and_variant(job_name, "mlxp")
    if not variant:
        return
    log_path = f"/data/youngwoong/experiments/{variant}/checkpoints/logs/training_rank0.log"
    try:
        pod = await ensure_listing_pod()
    except Exception:
        return
    cmd = (
        f"if [ -f {shlex.quote(log_path)} ]; then "
        f"  exec tail -n +1 -F {shlex.quote(log_path)}; "
        "else "
        f"  echo '(no log file at {log_path})'; "
        "  sleep 86400; "  # hold the connection so EventSource doesn't reconnect-loop
        "fi"
    )
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-n", _NS, pod, "--", "bash", "-c", cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1 << 20,
    )
    assert proc.stdout is not None
    try:
        async for line in _iter_logical_lines(proc.stdout):
            yield line
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _iter_logical_lines(reader):
    """Yield logical lines, splitting on either \\n or \\r so tqdm progress
    bars (which use \\r) don't accumulate into one giant buffer."""
    buf = b""
    while True:
        try:
            chunk = await reader.read(8192)
        except Exception:
            return
        if not chunk:
            if buf:
                yield buf.decode(errors="replace")
            return
        buf += chunk
        while True:
            i_n = buf.find(b"\n")
            i_r = buf.find(b"\r")
            idx = min(i for i in (i_n, i_r) if i != -1) if (i_n != -1 or i_r != -1) else -1
            if idx == -1:
                if len(buf) > (1 << 19):  # 512KB safety flush
                    yield buf.decode(errors="replace")
                    buf = b""
                break
            line = buf[:idx]
            buf = buf[idx + 1:]
            if line:
                yield line.decode(errors="replace")


async def _archived_record(name: str) -> dict[str, Any] | None:
    """Return a sacct-shaped dict for a single archived (k8s-gone) job by
    walking DDN. Returns None if no checkpoint dir matches the name."""
    archived = await _list_archived_jobs(seen=set())
    for j in archived:
        if j.job_name == name:
            return {
                "JobID": j.job_id,
                "JobName": j.job_name,
                "Partition": j.partition,
                "State": j.state,
                "ExitCode": "",
                "Start": j.start or "",
                "End": j.end or "",
                "Elapsed": j.elapsed,
                "NodeList": j.nodelist,
                "Reason": "(archived; k8s record GC'd)",
                "cluster": "mlxp",
            }
    return None
