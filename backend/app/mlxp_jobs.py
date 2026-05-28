"""MLXP (k8s) jobs: list / detail / cancel / log streaming via kubectl.

Keeps the same shape as slurm's `jobs.py` so the existing /jobs and
/jobs/<cluster>/<id> UI consumes both without branching.

Conventions:
  - k8s Job name = 6-char alpha-leading id (e.g. `m41d60`). Display name
    is carried as the annotation `train-eval-web/display-name` with shape
    `{phase}_{variant}_{YYYYMMDD}_{HHMMSS}` — same as slurm's job_name.
  - Jobs are labelled with the configured owner/tool labels. We list by owner.
  - State synthesized from job.status counts (Pending/Running/Succeeded/Failed).
"""

import asyncio
import json
import shlex
import shutil
from datetime import datetime, timezone
from typing import Any

from .job_identity import parse_comment_fields
from .jobs import Job
from .k8s_resources import affinity_node
from .kubectl_errors import is_kubectl_transport_error
from .mlxp_config import get_settings, owner_selector
from .time_utils import to_kst_iso


def _path_leaf(path: str | None) -> str | None:
    if not path:
        return None
    return path.rstrip("/").rsplit("/", 1)[-1] or None


async def _kubectl_json(*args: str, timeout: float = 20.0) -> dict[str, Any]:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        "kubectl", *args, "-o", "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"kubectl {' '.join(args)} timed out after {timeout:g}s")
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {stderr.decode(errors='replace').strip()}")
    try:
        return json.loads(stdout.decode())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"kubectl {' '.join(args)} returned invalid JSON: {exc}") from exc


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


def _parse_k8s_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _latest_k8s_time(values: list[str | None]) -> str | None:
    latest: tuple[datetime, str] | None = None
    for value in values:
        parsed = _parse_k8s_time(value)
        if parsed is None or value is None:
            continue
        if latest is None or parsed > latest[0]:
            latest = (parsed, value)
    return latest[1] if latest else None


def _terminated_container_times(pod: dict | None) -> list[str | None]:
    if not pod:
        return []
    status = pod.get("status") or {}
    out: list[str | None] = []
    for cs in status.get("containerStatuses", []) or []:
        for key in ("state", "lastState"):
            term = (cs.get(key) or {}).get("terminated")
            if term:
                out.append(term.get("finishedAt"))
    return out


def _end_time(job_status: dict, pod: dict | None = None) -> str | None:
    """Kubernetes does not consistently set Job.status.completionTime for
    failed Jobs. Use condition transition and container termination times as
    fallbacks so failed jobs sort by actual termination time in the UI."""
    candidates = [job_status.get("completionTime")]
    for cond in job_status.get("conditions", []) or []:
        if cond.get("status") == "True" and cond.get("type") in {
            "Complete",
            "SuccessCriteriaMet",
            "Failed",
            "FailureTarget",
        }:
            candidates.append(cond.get("lastTransitionTime"))
    candidates.extend(_terminated_container_times(pod))
    return _latest_k8s_time(candidates)


def _actual_pod_start(state: str, pod: dict | None, job_status: dict) -> str | None:
    if state.upper() == "PENDING":
        return None
    pod_status = (pod or {}).get("status") or {}
    return to_kst_iso(pod_status.get("startTime") or job_status.get("startTime"))


async def list_jobs(
    start: str | None = None,
    end: str | None = None,
    active_only: bool = False,
) -> list[Job]:
    """All MLXP jobs we know about: live k8s Jobs + archived runs derived
    from DDN checkpoint directories.

    k8s Jobs are GC'd 30 min after completion (`ttlSecondsAfterFinished`),
    so a Jobs page opened later wouldn't show them. We fall back to listing
    `<exp_dir>/checkpoints/<job-name>` on DDN — every completed training
    run leaves one of those behind for the lifetime of the checkpoints.
    """
    live, seen = await _list_live_jobs()
    if active_only:
        return [j for j in live if _is_active_state(j.state)]
    archived = await _list_archived_jobs(seen=seen)
    rows = live + archived
    if not start and not end:
        return rows
    return [j for j in rows if _is_active_state(j.state) or _job_in_range(j, start, end)]


def _is_active_state(state: str) -> bool:
    return state.upper() in {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"}


def _time_bound(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _job_in_range(job: Job, start: str | None, end: str | None) -> bool:
    start_ts = _time_bound(start)
    end_ts = _time_bound(end)
    job_ts = _time_bound(job.end) or _time_bound(job.start)
    if job_ts is None:
        return False
    if start_ts is not None and job_ts < start_ts:
        return False
    if end_ts is not None and job_ts > end_ts:
        return False
    return True


async def _list_live_jobs() -> tuple[list[Job], set[str]]:
    from .job_identity import parse_comment_metadata, parse_phase_and_variant

    settings = get_settings()
    data = await _kubectl_json(
        "get", "jobs", "-n", settings.namespace, "-l", owner_selector(settings)
    )

    # We need pod-level info too (nodeName + start time of the actual pod, not the Job).
    pods_by_job: dict[str, dict] = {}
    pod_data = await _kubectl_json(
        "get", "pods", "-n", settings.namespace, "-l", owner_selector(settings)
    )
    for p in pod_data.get("items", []):
        labels = p.get("metadata", {}).get("labels", {})
        jn = labels.get("job-name") or labels.get("batch.kubernetes.io/job-name")
        if jn:
            pods_by_job[jn] = p
    queue_positions = _pending_queue_positions(pods_by_job)

    out: list[Job] = []
    seen: set[str] = set()
    for j in data.get("items", []):
        job_id = j["metadata"]["name"]
        annotations = (j["metadata"].get("annotations") or {})
        job_name = annotations.get("train-eval-web/display-name") or job_id
        comment = annotations.get("train-eval-web/comment") or ""
        comment_fields = parse_comment_fields(comment)
        seen.update(x for x in (
            job_id,
            job_name,
            comment_fields.get("output_namespace"),
            _path_leaf(comment_fields.get("checkpoint_dir")),
        ) if x)
        phase, variant = parse_comment_metadata(comment)
        if not phase or not variant:
            phase, variant = parse_phase_and_variant(job_name)
        status = j.get("status", {}) or {}
        pod = pods_by_job.get(job_id)
        state = _state_from_pod_and_job(status, pod)
        start = _actual_pod_start(state, pod, status)
        end = to_kst_iso(_end_time(status, pod))
        elapsed = _elapsed(start, end)

        pod_spec = (pod or {}).get("spec") or {}
        nodelist = pod_spec.get("nodeName") or affinity_node(pod_spec) or "(unscheduled)"

        out.append(Job(
            cluster="mlxp", job_id=job_id, job_name=job_name, partition="mlxp",
            state=state, elapsed=elapsed, nodelist=nodelist,
            queue_position=queue_positions.get(job_id),
            start=start, end=end,
            phase=None if phase == "unknown" else phase,
            variant=variant,
        ))
    return out, seen


def _pending_queue_positions(pods_by_job: dict[str, dict]) -> dict[str, int]:
    pending: list[tuple[str, str | None, datetime, str]] = []
    max_time = datetime.max.replace(tzinfo=timezone.utc)
    for job_id, pod in pods_by_job.items():
        phase = ((pod.get("status") or {}).get("phase") or "").strip()
        if phase != "Pending":
            continue
        metadata = pod.get("metadata") or {}
        spec = pod.get("spec") or {}
        node = spec.get("nodeName") or affinity_node(spec)
        created = _parse_k8s_time(metadata.get("creationTimestamp")) or max_time
        pending.append((job_id, node, created, metadata.get("name") or job_id))

    pending.sort(key=lambda item: (item[2], item[3]))
    per_node: dict[str, int] = {}
    positions: dict[str, int] = {}
    global_position = 0
    for job_id, node, _, _ in pending:
        global_position += 1
        if node:
            per_node[node] = per_node.get(node, 0) + 1
            positions[job_id] = per_node[node]
        else:
            positions[job_id] = global_position
    return positions


async def _list_archived_jobs(seen: set[str]) -> list[Job]:
    """Scan DDN for completed runs whose k8s Job has been GC'd."""
    out: list[Job] = []
    for run in await _list_archived_runs():
        identities = {run["job_id"], run["job_name"], run["output_namespace"]}
        if identities & seen:
            continue
        out.append(Job(
            cluster="mlxp",
            job_id=run["job_id"],
            job_name=run["job_name"],
            partition="mlxp",
            state="COMPLETED",
            elapsed=run["elapsed"],
            nodelist="(archived)",
            start=run["start"],
            end=run["end"],
            phase="train",
            variant=run["variant"],
        ))
    return out


async def _list_archived_runs() -> list[dict[str, str]]:
    import asyncio
    from .mlxp_data_pod import ensure_listing_pod

    try:
        pod = await ensure_listing_pod()
    except Exception:
        return []

    settings = get_settings()
    meta_index = await _archived_metadata_index(pod)
    experiments_root = shlex.quote(settings.experiments_dir)
    # One liner per run: <variant>|<output_namespace>|<latest_step>|<start_epoch>|<end_epoch>
    # NOTE: array glob with nullglob, not `ls $glob` — `ls` with zero-arg
    # falls back to listing `.` which used to slip past the empty check and
    # produced fake rows (`experiment_cfg`, `logs`, `runs`) in the table.
    script = r"""
shopt -s nullglob
for d in __EXPERIMENTS_ROOT__/*/checkpoints/ __EXPERIMENTS_ROOT__/*/checkpoints/*/ __EXPERIMENTS_ROOT__/*/checkpoints/*/*/; do
    matches=( "$d"checkpoint-* )
    [ ${#matches[@]} -eq 0 ] && continue
    output_namespace=$(basename "$d")
    rel="${d#__EXPERIMENTS_ROOT__/}"
    variant="${rel%%/*}"
    [ "$output_namespace" = "checkpoints" ] && output_namespace="$variant"
    latest=$(for m in "${matches[@]}"; do basename "$m" | sed 's:^checkpoint-::'; done | sort -n | tail -1)
    start_epoch=$(stat -c %Y "$d" 2>/dev/null || echo 0)
    end_epoch=$(stat -c %Y "$d"checkpoint-$latest 2>/dev/null || echo "$start_epoch")
    printf '%s|%s|%s|%s|%s\n' "$variant" "$output_namespace" "$latest" "$start_epoch" "$end_epoch"
done
""".replace("__EXPERIMENTS_ROOT__", experiments_root)
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
    except Exception:
        return []

    out: list[dict[str, str]] = []
    for line in stdout.decode(errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        variant, output_namespace, latest, start_e, end_e = parts
        if not latest.isdigit():
            continue
        meta = meta_index.get(output_namespace, {})
        try:
            start_i = int(start_e)
            end_i = int(end_e)
            if end_i < start_i:
                start_i, end_i = end_i, start_i
            start_iso = to_kst_iso(datetime.fromtimestamp(start_i, tz=timezone.utc).isoformat())
            end_iso = to_kst_iso(datetime.fromtimestamp(end_i, tz=timezone.utc).isoformat())
        except ValueError:
            continue
        job_id = str(meta.get("job_id") or output_namespace)
        job_name = str(meta.get("job_name") or output_namespace)
        out.append({
            "job_id": job_id,
            "job_name": job_name,
            "output_namespace": output_namespace,
            "variant": str(meta.get("variant") or variant),
            "train_note": str(meta.get("train_note") or ""),
            "start": start_iso or "",
            "end": end_iso or "",
            "elapsed": _elapsed(start_iso, end_iso),
        })
    return out


async def _archived_metadata_index(pod: str) -> dict[str, dict[str, Any]]:
    settings = get_settings()
    experiments_root = shlex.quote(settings.experiments_dir)
    script = r"""
shopt -s nullglob
for f in __EXPERIMENTS_ROOT__/*/config_*.meta.json; do
    printf '\036%s\n' "$f"
    cat "$f"
    printf '\n'
done
""".replace("__EXPERIMENTS_ROOT__", experiments_root)
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
    except Exception:
        return {}

    index: dict[str, dict[str, Any]] = {}
    for chunk in stdout.decode(errors="replace").split("\x1e"):
        if not chunk.strip():
            continue
        _, _, body = chunk.partition("\n")
        try:
            meta = json.loads(body)
        except json.JSONDecodeError:
            continue
        keys = {
            str(meta.get("output_namespace") or ""),
            str(meta.get("job_id") or ""),
            _path_leaf(str(meta.get("checkpoint_dir") or "")) or "",
        }
        for key in keys:
            if key:
                index[key] = meta
    return index


async def get_job(name: str) -> dict[str, Any]:
    """Return a slurm-sacct-shaped dict for one MLXP job.

    Falls back to the archived-on-DDN view when the k8s Job is gone
    (TTL'd), so the job-detail page works for completed runs too.
    """
    settings = get_settings()
    try:
        job_data = await _kubectl_json("get", "job", name, "-n", settings.namespace)
    except RuntimeError:
        archived = await _archived_record(name)
        if archived:
            return archived
        raise FileNotFoundError(f"mlxp job not found: {name}")
    status = job_data.get("status", {}) or {}
    pod_data = await _kubectl_json("get", "pods", "-n", settings.namespace, "-l", f"job-name={name}")
    pods = pod_data.get("items", [])
    pod = pods[0] if pods else {}
    pod_status = pod.get("status", {}) or {}
    pod_name = pod.get("metadata", {}).get("name", "")
    node = pod.get("spec", {}).get("nodeName", "") or ""
    state = _state_from_pod_and_job(status, pod if pods else None)
    start = _actual_pod_start(state, pod if pods else None, status) or ""
    end = to_kst_iso(_end_time(status, pod if pods else None)) or ""

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

    annotations = ((job_data.get("metadata") or {}).get("annotations") or {})
    job_name = annotations.get("train-eval-web/display-name") or name
    job_comment = annotations.get("train-eval-web/comment") or ""
    train_note = annotations.get("train-eval-web/train-note") or ""
    return {
        "JobID": name,
        "JobName": job_name,
        "JobComment": job_comment,
        "JobTrainNote": train_note,
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
    last_error = ""
    for attempt in range(5):
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            "delete",
            "job",
            name,
            "-n",
            settings.namespace,
            "--wait=false",
            "--ignore-not-found=true",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        if proc.returncode == 0:
            return
        last_error = err or out or "kubectl delete failed"
        if _kubectl_not_found(last_error) or await _job_deleted(name):
            return
        if not is_kubectl_transport_error(last_error) or attempt == 4:
            break
        await asyncio.sleep(1.5 * (attempt + 1))

    if is_kubectl_transport_error(last_error):
        raise RuntimeError(
            "transient Kubernetes transport failure while deleting job after retries: "
            f"{last_error}"
        )
    raise RuntimeError(f"kubectl delete failed: {last_error}")


def _kubectl_not_found(message: str) -> bool:
    lower = message.lower()
    return "notfound" in lower or "not found" in lower


async def _job_deleted(name: str) -> bool:
    settings = get_settings()
    try:
        await _kubectl_json("get", "job", name, "-n", settings.namespace, timeout=10.0)
    except RuntimeError as e:
        message = str(e)
        if _kubectl_not_found(message):
            return True
        # A transport failure during verification is not proof that delete
        # succeeded, so keep retrying the original delete path.
        return False
    return False


async def tail_logs(job_name: str, follow: bool = True, start_line: int = 1):
    """Stream log lines for an MLXP job.

    Primary source is `kubectl logs` on the job's pod. Once a job's k8s
    record has been GC'd we fall back to the on-DDN log file the body
    script left at `<exp_dir>/checkpoints/logs/training_rank0.log`.
    """
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    settings = get_settings()
    pod_data = await _kubectl_json("get", "pods", "-n", settings.namespace, "-l", f"job-name={job_name}")
    pods = pod_data.get("items", [])
    if not pods:
        async for line in _tail_archived_log(job_name, start_line=start_line):
            yield line
        return
    pod_name = pods[0]["metadata"]["name"]

    # No --tail: stream the full log from the start. Frontend handles the
    # scrollback; backend just delivers everything kubectl will emit.
    args = ["kubectl", "logs", "-n", settings.namespace, pod_name]
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
        line_no = 1
        async for line in _iter_logical_lines(proc.stdout):
            if line_no >= start_line:
                yield line
            line_no += 1
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _tail_archived_log(job_name: str, start_line: int = 1):
    """Serve <variant>/checkpoints/logs/training_rank0.log via the data pod,
    holding the connection open with tail -F so EventSource doesn't loop."""
    from .job_identity import parse_comment_metadata, parse_phase_and_variant
    from .mlxp_data_pod import ensure_listing_pod

    record = await _archived_record(job_name)
    variant = None
    if record:
        _, variant = parse_comment_metadata(record.get("JobComment") or "")
    if not variant:
        _, variant = parse_phase_and_variant(job_name)
    if not variant:
        return
    settings = get_settings()
    log_paths = [
        f"{settings.experiments_dir}/{variant}/checkpoints/{job_name}/logs/training.log",
        f"{settings.experiments_dir}/{variant}/checkpoints/logs/training_rank0.log",
        f"{settings.experiments_dir}/{variant}/logs/{job_name}/eval.log",
    ]
    try:
        pod = await ensure_listing_pod()
    except Exception:
        return
    tests = " ".join(shlex.quote(p) for p in log_paths)
    safe_start = max(1, int(start_line))
    cmd = (
        f"for p in {tests}; do "
        f'  if [ -f "$p" ]; then exec tail -n +{safe_start} -F "$p"; fi; '
        "done; "
        "echo '(archived MLXP pod logs are unavailable for this run; no persisted training log was found)'; "
        "sleep 86400"
    )
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-c", cmd,
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
            yield line.decode(errors="replace")


async def _archived_record(name: str) -> dict[str, Any] | None:
    """Return a sacct-shaped dict for a single archived (k8s-gone) job by
    walking DDN. Returns None if no checkpoint dir matches the name."""
    archived = None
    for run in await _list_archived_runs():
        if name in {run["job_id"], run["job_name"], run["output_namespace"]}:
            archived = run
            break
    if archived is None:
        return None
    return {
        "JobID": archived["job_id"],
        "JobName": archived["job_name"],
        "JobTrainNote": archived.get("train_note", ""),
        "JobComment": (
            f"phase=train;variant={archived['variant']};"
            f"output_namespace={archived['output_namespace']}"
        ),
        "Partition": "mlxp",
        "State": "COMPLETED",
        "ExitCode": "",
        "Start": archived["start"],
        "End": archived["end"],
        "Elapsed": archived["elapsed"],
        "NodeList": "(archived)",
        "Reason": "(archived; k8s record GC'd)",
        "cluster": "mlxp",
    }
