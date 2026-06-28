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

from __future__ import annotations

import asyncio
import json
import shlex
from datetime import datetime, timezone
from typing import Any

from .job_identity import (
    parse_comment_fields,
    parse_comment_metadata,
    parse_phase_and_variant,
    resolve_identity,
)
from .jobs import ACTIVE_STATES, Job
from .k8s_resources import (
    affinity_node,
    ensure_kubectl,
    kubectl_json,
    parse_k8s_time,
    requested_gpus,
)
from .kubectl_errors import is_kubectl_not_found, is_kubectl_transport_error
from .mlxp_config import get_settings, owner_selector
from .mlxp_data_pod import ensure_listing_pod
from .paths import CHECKPOINT_COPY_HISTORY_REL
from .ssh import iter_logical_lines
from .time_utils import to_kst_iso


def _path_leaf(path: str | None) -> str | None:
    if not path:
        return None
    return path.rstrip("/").rsplit("/", 1)[-1] or None


def _state_from_pod_and_job(job_status: dict[str, Any], pod: dict[str, Any] | None, *, suspended: bool = False) -> str:
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
    if suspended:
        # Queue job yielded to a higher class (or awaiting admission); the
        # scheduler auto-resumes it when capacity frees up.
        return "SUSPENDED"
    return "PENDING"


def _elapsed(start: str | None, end: str | None) -> str:
    if not start:
        return "0:00"
    s = parse_k8s_time(start)
    if s is None:
        return "0:00"
    e = datetime.now(timezone.utc)
    if end:
        e = parse_k8s_time(end) or e
    delta = int((e - s).total_seconds())
    h, rem = divmod(delta, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _latest_k8s_time(values: list[str | None]) -> str | None:
    latest: tuple[datetime, str] | None = None
    for value in values:
        if not value:
            continue
        parsed = parse_k8s_time(value)
        if parsed is None:
            continue
        if latest is None or parsed > latest[0]:
            latest = (parsed, value)
    return latest[1] if latest else None


def _terminated_container_times(pod: dict[str, Any] | None) -> list[str | None]:
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


def _end_time(job_status: dict[str, Any], pod: dict[str, Any] | None = None) -> str | None:
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


def _actual_pod_start(state: str, pod: dict[str, Any] | None, job_status: dict[str, Any]) -> str | None:
    if state.upper() == "PENDING":
        return None
    pod_status = (pod or {}).get("status") or {}
    return to_kst_iso(pod_status.get("startTime") or job_status.get("startTime"))


async def list_jobs(
    start: str | None = None,
    end: str | None = None,
    active_only: bool = False,
) -> list[Job]:
    """All MLXP jobs we know about: live k8s Jobs + archived training runs.

    k8s Jobs are GC'd 30 min after completion (`ttlSecondsAfterFinished`),
    so a Jobs page opened later wouldn't show them. We rebuild completed
    training rows from submit metadata plus durable artifact evidence.
    """
    live, seen = await _list_live_jobs()
    if active_only:
        return [j for j in live if j.state.upper() in ACTIVE_STATES]
    # Date-bound the (expensive) DDN archived scan: only walk runs whose
    # artifact mtime is at/after the window start, minus a 1-day margin. That
    # mtime is exactly what _job_in_range filters on, so this can't drop an
    # in-window run while turning a ~60s full walk into a ~3s pruned one.
    since_epoch: int | None = None
    if start:
        st = _time_bound(start)
        if st is not None:
            since_epoch = int(st) - 86400
    archived = await _list_archived_jobs(seen=seen, since_epoch=since_epoch)
    rows = live + archived
    if not start and not end:
        return rows
    return [j for j in rows if j.state.upper() in ACTIVE_STATES or _job_in_range(j, start, end)]


def _time_bound(value: str | None) -> float | None:
    parsed = parse_k8s_time(value)
    return parsed.timestamp() if parsed else None


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
    settings = get_settings()
    data = await kubectl_json(
        "get", "jobs", "-n", settings.namespace, "-l", owner_selector(settings)
    )

    # We need pod-level info too (nodeName + start time of the actual pod, not the Job).
    pods_by_job: dict[str, dict] = {}
    pod_data = await kubectl_json(
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
        phase, variant = resolve_identity(None, job_name, {"JobComment": comment})
        status = j.get("status", {}) or {}
        pod = pods_by_job.get(job_id)
        state = _state_from_pod_and_job(
            status, pod, suspended=bool((j.get("spec") or {}).get("suspend")),
        )
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
        created = parse_k8s_time(metadata.get("creationTimestamp")) or max_time
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


async def _list_archived_jobs(seen: set[str], since_epoch: int | None = None) -> list[Job]:
    """Scan DDN for completed runs whose k8s Job has been GC'd."""
    out: list[Job] = []
    for run in await _list_archived_runs(since_epoch):
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


async def _list_archived_runs(since_epoch: int | None = None) -> list[dict[str, str]]:
    try:
        pod = await ensure_listing_pod()
    except Exception:
        return []

    out: list[dict[str, str]] = []
    metas, artifacts, copies = await asyncio.gather(
        _archived_metadata_records(pod),
        _archived_artifact_index(pod, since_epoch),
        _copy_history_index(pod),
    )

    for meta in metas:
        if str(meta.get("phase") or "").strip() != "train":
            continue
        output_namespace = str(
            meta.get("output_namespace")
            or _path_leaf(str(meta.get("checkpoint_dir") or ""))
            or ""
        ).strip()
        if not output_namespace:
            continue

        keys = _archived_identity_keys(meta, output_namespace)
        artifact = next((artifacts[k] for k in keys if k in artifacts), None)
        copy_record = next((copies[k] for k in keys if k in copies), None)
        if not _has_completed_artifact(artifact, copy_record):
            continue

        train_meta = meta.get("train") if isinstance(meta.get("train"), dict) else {}
        start_iso = _metadata_start_time(meta, artifact)
        end_iso = _archived_end_time(meta, artifact, copy_record)
        job_id = str(meta.get("job_id") or output_namespace)
        job_name = str(meta.get("job_name") or output_namespace)
        artifact_variant = str((artifact or {}).get("variant") or "")
        out.append({
            "job_id": job_id,
            "job_name": job_name,
            "output_namespace": output_namespace,
            "variant": str(meta.get("variant") or artifact_variant),
            "train_note": str(meta.get("train_note") or ""),
            "num_gpus": str(train_meta.get("num_gpus") or meta.get("num_gpus") or ""),
            "start": start_iso or "",
            "end": end_iso or "",
            "elapsed": _elapsed(start_iso, end_iso),
        })
    return out


def _epoch_to_kst(value: Any) -> str | None:
    try:
        epoch = float(value)
    except (TypeError, ValueError):
        return None
    if epoch <= 0:
        return None
    return to_kst_iso(datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat())


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metadata_start_time(meta: dict[str, Any], artifact: dict[str, Any] | None) -> str | None:
    created = str(meta.get("created_at") or "").strip()
    if created:
        return to_kst_iso(created)
    if artifact:
        return _epoch_to_kst(artifact.get("start_epoch"))
    return None


def _archived_end_time(
    meta: dict[str, Any],
    artifact: dict[str, Any] | None,
    copy_record: dict[str, Any] | None,
) -> str | None:
    if artifact:
        end = _epoch_to_kst(artifact.get("end_epoch"))
        if end:
            return end
    if copy_record:
        copied = _epoch_to_kst(copy_record.get("copied_at"))
        if copied:
            return copied
    created = str(meta.get("created_at") or "").strip()
    return to_kst_iso(created) if created else None


def _archived_identity_keys(meta: dict[str, Any], output_namespace: str) -> list[str]:
    keys = [
        output_namespace,
        str(meta.get("job_id") or ""),
        str(meta.get("job_name") or ""),
        _path_leaf(str(meta.get("checkpoint_dir") or "")) or "",
    ]
    return [k for k in dict.fromkeys(k.strip() for k in keys if k and k.strip())]


def _has_completed_artifact(
    artifact: dict[str, Any] | None,
    copy_record: dict[str, Any] | None,
) -> bool:
    if artifact and (artifact.get("has_checkpoint_subdir") or artifact.get("has_final_model")):
        return True
    return bool(copy_record and copy_record.get("dest_exists") is True)


async def _archived_metadata_records(pod: str) -> list[dict[str, Any]]:
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
        return []

    records: list[dict[str, Any]] = []
    for chunk in stdout.decode(errors="replace").split("\x1e"):
        if not chunk.strip():
            continue
        _, _, body = chunk.partition("\n")
        try:
            meta = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(meta, dict):
            records.append(meta)
    return records


async def _archived_artifact_index(pod: str, since_epoch: int | None = None) -> dict[str, dict[str, Any]]:
    settings = get_settings()
    experiments_root = shlex.quote(settings.experiments_dir)
    # The per-dir body costs several NFS round-trips, so it dominates the walk
    # (~60s over all runs). When a window start is known we list only checkpoint
    # dirs modified at/after it via `find -newermt` and run the body on those;
    # otherwise we fall back to the full glob. The body itself is unchanged.
    script = r"""
SINCE_EPOCH='__SINCE_EPOCH__'
shopt -s nullglob
emit() {
    local d="$1" output_namespace rel variant latest has_checkpoint end_path has_final start_epoch end_epoch
    local matches
    output_namespace=$(basename "$d")
    rel="${d#__EXPERIMENTS_ROOT__/}"
    variant="${rel%%/*}"
    [ "$output_namespace" = "checkpoints" ] && output_namespace="$variant"

    matches=( "$d"checkpoint-* )
    latest=""
    has_checkpoint=0
    end_path="$d"
    if [ ${#matches[@]} -gt 0 ]; then
        has_checkpoint=1
        latest=$(for m in "${matches[@]}"; do basename "$m" | sed 's:^checkpoint-::'; done | sort -n | tail -1)
        end_path="${d}checkpoint-${latest}"
        # Prefer a file written once at the final save: the step DIRECTORY's
        # mtime moves whenever children change (e.g. trainer-state cleanup
        # deleting optimizer files), which made old runs look freshly ended.
        if [ -f "${end_path}/model.safetensors.index.json" ]; then
            end_path="${end_path}/model.safetensors.index.json"
        fi
    fi

    has_final=0
    if [ -f "${d}model.safetensors.index.json" ] || compgen -G "${d}model-*.safetensors" >/dev/null; then
        has_final=1
        if [ "$has_checkpoint" = 0 ] && [ -f "${d}model.safetensors.index.json" ]; then
            end_path="${d}model.safetensors.index.json"
        fi
    fi
    [ "$has_checkpoint" = 0 ] && [ "$has_final" = 0 ] && return

    start_epoch=$(stat -c %Y "$d" 2>/dev/null || echo 0)
    end_epoch=$(stat -c %Y "$end_path" 2>/dev/null || echo "$start_epoch")
    printf '%s|%s|%s|%s|%s|%s\n' "$variant" "$output_namespace" "$has_checkpoint" "$has_final" "$start_epoch" "$end_epoch"
}
if [ -n "$SINCE_EPOCH" ]; then
    while IFS= read -r d; do emit "$d"; done < <(find __EXPERIMENTS_ROOT__ -mindepth 2 -maxdepth 4 -type d \( -name checkpoints -o -path '*/checkpoints/*' \) -newermt "@$SINCE_EPOCH" -printf '%p/\n' 2>/dev/null)
else
    for d in __EXPERIMENTS_ROOT__/*/checkpoints/ __EXPERIMENTS_ROOT__/*/checkpoints/*/ __EXPERIMENTS_ROOT__/*/checkpoints/*/*/; do emit "$d"; done
fi
""".replace("__EXPERIMENTS_ROOT__", experiments_root).replace("__SINCE_EPOCH__", str(since_epoch) if since_epoch is not None else "")
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
    for line in stdout.decode(errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) != 6:
            continue
        variant, output_namespace, has_checkpoint, has_final, start_e, end_e = parts
        if not output_namespace:
            continue
        record = {
            "variant": variant,
            "output_namespace": output_namespace,
            "has_checkpoint_subdir": has_checkpoint == "1",
            "has_final_model": has_final == "1",
            "start_epoch": start_e,
            "end_epoch": end_e,
        }
        index[output_namespace] = record
    return index


async def _copy_history_index(pod: str) -> dict[str, dict[str, Any]]:
    settings = get_settings()
    hist_dir = shlex.quote(f"{settings.experiments_dir}/{CHECKPOINT_COPY_HISTORY_REL}")
    script = f"cat {hist_dir}/*.jsonl 2>/dev/null || true"
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except Exception:
        return {}

    index: dict[str, dict[str, Any]] = {}
    for line in stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("dest_exists") is not True:
            continue
        keys = [
            str(record.get("source_job") or ""),
            _path_leaf(str(record.get("source_path") or "")) or "",
            _path_leaf(str(record.get("dest_path") or "")) or "",
        ]
        for key in keys:
            if not key:
                continue
            previous = index.get(key)
            if not previous or _float_or_zero(record.get("copied_at")) > _float_or_zero(previous.get("copied_at")):
                index[key] = record
    return index


async def get_job(name: str) -> dict[str, Any]:
    """Return a slurm-sacct-shaped dict for one MLXP job.

    Falls back to the archived-on-DDN view when the k8s Job is gone
    (TTL'd), so the job-detail page works for completed runs too.
    """
    settings = get_settings()
    try:
        job_data = await kubectl_json("get", "job", name, "-n", settings.namespace)
    except RuntimeError:
        archived = await _archived_record(name)
        if archived:
            return archived
        raise FileNotFoundError(f"mlxp job not found: {name}")
    status = job_data.get("status", {}) or {}
    pod_data = await kubectl_json("get", "pods", "-n", settings.namespace, "-l", f"job-name={name}")
    pods = pod_data.get("items", [])
    pod = pods[0] if pods else {}
    pod_status = pod.get("status", {}) or {}
    pod_name = pod.get("metadata", {}).get("name", "")
    pod_spec = pod.get("spec", {}) or {}
    template_spec = (((job_data.get("spec") or {}).get("template") or {}).get("spec") or {})
    node = pod_spec.get("nodeName", "") or ""
    gpu_count = requested_gpus(pod_spec) or requested_gpus(template_spec)
    state = _state_from_pod_and_job(
        status,
        pod if pods else None,
        suspended=bool((job_data.get("spec") or {}).get("suspend")),
    )
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
        "GPUs": str(gpu_count) if gpu_count else "",
        "cluster": "mlxp",
        # Extra (for log streaming endpoint)
        "_pod_name": pod_name,
    }


async def cancel_job(name: str) -> None:
    ensure_kubectl()
    settings = get_settings()
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
        if is_kubectl_not_found(last_error) or await _job_deleted(name):
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


async def _job_deleted(name: str) -> bool:
    settings = get_settings()
    try:
        await kubectl_json("get", "job", name, "-n", settings.namespace, timeout=10.0)
    except RuntimeError as e:
        message = str(e)
        if is_kubectl_not_found(message):
            return True
        # A transport failure during verification is not proof that delete
        # succeeded, so keep retrying the original delete path.
        return False
    return False


async def tail_logs(job_name: str, start_line: int = 1):
    """Stream log lines for an MLXP job.

    Primary source is `kubectl logs` on the job's pod. Once a job's k8s
    record has been GC'd we fall back to the on-DDN log file the body
    script left at `<exp_dir>/checkpoints/logs/training_rank0.log`.
    """
    ensure_kubectl()
    settings = get_settings()
    pod_data = await kubectl_json("get", "pods", "-n", settings.namespace, "-l", f"job-name={job_name}")
    pods = pod_data.get("items", [])
    if not pods:
        async for line in _tail_archived_log(job_name, start_line=start_line):
            yield line
        return
    pod_name = pods[0]["metadata"]["name"]

    # No --tail: stream the full log from the start. Frontend handles the
    # scrollback; backend just delivers everything kubectl will emit.
    args = ["kubectl", "logs", "-n", settings.namespace, pod_name, "-f"]
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
        async for line in iter_logical_lines(proc.stdout):
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
    record = await _archived_record(job_name)
    variant = None
    output_namespace = None
    if record:
        comment = record.get("JobComment") or ""
        _, variant = parse_comment_metadata(comment)
        output_namespace = parse_comment_fields(comment).get("output_namespace")
    if not variant:
        _, variant = parse_phase_and_variant(job_name)
    if not variant:
        return
    settings = get_settings()
    checkpoint_log_names = [x for x in dict.fromkeys([output_namespace, job_name]) if x]
    log_paths = [
        *[
            f"{settings.experiments_dir}/{variant}/checkpoints/{name}/logs/training.log"
            for name in checkpoint_log_names
        ],
        f"{settings.experiments_dir}/{variant}/checkpoints/logs/training_rank0.log",
        *[
            f"{settings.experiments_dir}/{variant}/logs/{name}/eval.log"
            for name in checkpoint_log_names
        ],
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
        async for line in iter_logical_lines(proc.stdout):
            yield line
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


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
        "GPUs": archived.get("num_gpus", ""),
        "cluster": "mlxp",
    }
