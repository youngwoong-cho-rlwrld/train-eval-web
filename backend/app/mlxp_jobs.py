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


def _state_from_job_status(status: dict) -> str:
    """Map k8s Job status counters to a slurm-like state string."""
    active = int(status.get("active") or 0)
    succeeded = int(status.get("succeeded") or 0)
    failed = int(status.get("failed") or 0)
    if active > 0:
        return "RUNNING"
    if succeeded > 0:
        return "COMPLETED"
    if failed > 0:
        return "FAILED"
    # Job exists but no pods yet → pending
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
    """All MLXP jobs in p-rlwrld with owner=youngwoong label."""
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
        status = j.get("status", {}) or {}
        state = _state_from_job_status(status)
        start = status.get("startTime")
        end = status.get("completionTime")
        elapsed = _elapsed(start, end)

        pod = pods_by_job.get(name) or {}
        nodelist = pod.get("spec", {}).get("nodeName") or "(unscheduled)"

        out.append(Job(
            cluster="mlxp", job_id=name, job_name=name, partition="mlxp",
            state=state, elapsed=elapsed, nodelist=nodelist,
            start=start, end=end,
        ))
    return out


async def get_job(name: str) -> dict[str, Any]:
    """Return a slurm-sacct-shaped dict for one MLXP job."""
    try:
        job_data = await _kubectl_json("get", "job", name, "-n", _NS)
    except RuntimeError as e:
        raise FileNotFoundError(f"mlxp job not found: {name}: {e}")
    status = job_data.get("status", {}) or {}
    state = _state_from_job_status(status)
    start = status.get("startTime") or ""
    end = status.get("completionTime") or ""

    pod_data = await _kubectl_json("get", "pods", "-n", _NS, "-l", f"job-name={name}")
    pods = pod_data.get("items", [])
    pod = pods[0] if pods else {}
    pod_status = pod.get("status", {}) or {}
    pod_name = pod.get("metadata", {}).get("name", "")
    node = pod.get("spec", {}).get("nodeName", "") or ""

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

    return {
        "JobID": name,
        "JobName": name,
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
    """Yield log lines for the job's pod via `kubectl logs`."""
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    # Resolve pod name first.
    pod_data = await _kubectl_json("get", "pods", "-n", _NS, "-l", f"job-name={job_name}")
    pods = pod_data.get("items", [])
    if not pods:
        return
    pod_name = pods[0]["metadata"]["name"]

    args = ["kubectl", "logs", "-n", _NS, pod_name, "--tail=200"]
    if follow:
        args.append("-f")
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode(errors="replace").rstrip("\n")
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
