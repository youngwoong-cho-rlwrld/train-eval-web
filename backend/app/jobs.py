"""Slurm job listing + status + cancel."""

from __future__ import annotations

import asyncio
import shlex
from datetime import timezone

from pydantic import BaseModel

from .clusters import load_cluster, list_clusters
from .eval_completion import eval_job_completed_from_log_dir
from .job_identity import resolve_identity
from .partitions import gpu_count_from_tres
from .slurm_meta import read_slurm_meta, read_slurm_meta_many
from .ssh import ssh_run
from .time_utils import scheduler_timezone, to_kst_iso


class Job(BaseModel):
    cluster: str
    job_id: str
    job_name: str
    partition: str
    state: str
    elapsed: str
    nodelist: str
    reason: str = ""
    time_left: str | None = None     # `squeue %L`; None for sacct-only rows
    queue_position: int | None = None # 1-based pending position within partition
    start: str | None = None         # ISO timestamp, or None / "Unknown"
    end: str | None = None           # ISO timestamp, or None / "Unknown"
    phase: str | None = None
    variant: str | None = None
    resume_of: str | None = None
    resubmit_action: str | None = None


_SQUEUE_FMT = "%i|%j|%P|%T|%M|%R|%L|%S"
_SACCT_LIST_FMT = "JobID,JobName,Partition,State,Elapsed,Start,End,NodeList"
ACTIVE_STATES = {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"}

# Slurm terminal-state prefix groups (sacct may suffix states, e.g.
# "CANCELLED by <uid>"), so classify by uppercase prefix.
TIMEOUT_PREFIXES = ("TIMEOUT",)
# Failures a `retry` should re-submit; TIMEOUT is handled by `resume` and
# CANCELLED is not retryable, so both are intentionally excluded here.
RETRYABLE_FAILURE_PREFIXES = ("FAIL", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPT")
TERMINAL_NON_COMPLETED_PREFIXES = (
    "FAIL",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPT",
    "CANCEL",
)


def short_state(raw: str) -> str:
    """Truncate sacct's "CANCELLED by <uid>" suffix to the bare state token."""
    return raw.split(" ")[0]


def is_timeout(state: str) -> bool:
    return state.upper().startswith(TIMEOUT_PREFIXES)


def is_retryable_failure(state: str) -> bool:
    return state.upper().startswith(RETRYABLE_FAILURE_PREFIXES)


def is_terminal_non_completed(state: str) -> bool:
    upper = state.upper()
    if upper in ACTIVE_STATES or upper.startswith("COMPLET"):
        return False
    return upper.startswith(TERMINAL_NON_COMPLETED_PREFIXES)


def _actual_start_for_state(state: str, value: str | None, source_tz: timezone | None = None) -> str | None:
    # In squeue, %S is the actual start for running jobs, but for pending jobs
    # Slurm may report an estimated future start time. The UI's "Started"
    # field should stay empty until a job has actually started.
    if state.upper() == "PENDING":
        return None
    return to_kst_iso(value, source_tz)


async def list_jobs(
    clusters: list[str] | None = None,
    hours: int = 24,
    start: str | None = None,
    end: str | None = None,
) -> list[Job]:
    """Active jobs from `squeue` + finished jobs from `sacct`.

    Fans out across clusters (and across squeue/sacct within each cluster)
    so latency is roughly the slowest single SSH call, not their sum.
    squeue takes precedence on overlap (its state is fresher than sacct's
    for jobs that just finished).
    """
    from . import mlxp_jobs
    target_clusters = clusters or list_clusters()
    active_only = not start and not end and hours <= 0

    async def _for_cluster(c: str) -> list[Job]:
        if c == "mlxp":
            try:
                return await mlxp_jobs.list_jobs(start=start, end=end, active_only=active_only)
            except Exception as exc:
                raise RuntimeError(f"mlxp job listing failed: {exc}") from exc
        try:
            env = await load_cluster(c)
        except FileNotFoundError:
            return []
        host = env.ssh_alias
        source_tz_task = asyncio.create_task(scheduler_timezone(host))

        async def _squeue() -> str:
            try:
                r = await ssh_run(host, f'squeue -u "$USER" -h -o "{_SQUEUE_FMT}"', timeout=15.0)
            except Exception as exc:
                raise RuntimeError(f"{c} squeue failed: {exc}") from exc
            if r.returncode != 0:
                message = (r.stderr or r.stdout or "unknown error").strip()
                raise RuntimeError(f"{c} squeue failed: {message}")
            return r.stdout

        async def _queue_positions() -> dict[str, int]:
            try:
                r = await ssh_run(host, "squeue -h -t PD -o '%i|%P'", timeout=15.0)
            except Exception:
                return {}
            if r.returncode != 0:
                return {}
            per_partition: dict[str, int] = {}
            positions: dict[str, int] = {}
            for line in r.stdout.strip().splitlines():
                parts = line.split("|", 1)
                if len(parts) != 2:
                    continue
                job_id = parts[0].strip()
                partition = parts[1].strip()
                if not job_id or not partition:
                    continue
                per_partition[partition] = per_partition.get(partition, 0) + 1
                positions[job_id] = per_partition[partition]
            return positions

        async def _sacct() -> str:
            if start or end:
                start_arg = shlex.quote(start or f"now-{hours}hours")
                end_arg = f" -E {shlex.quote(end)}" if end else ""
                window = f"-S {start_arg}{end_arg}"
            else:
                window = f"-S now-{hours}hours"
            try:
                r = await ssh_run(
                    host,
                    f'sacct -X -u "$USER" {window} -P -n -o {_SACCT_LIST_FMT}',
                    timeout=30.0,
                )
            except Exception as exc:
                raise RuntimeError(f"{c} sacct failed: {exc}") from exc
            if r.returncode != 0:
                message = (r.stderr or r.stdout or "unknown error").strip()
                raise RuntimeError(f"{c} sacct failed: {message}")
            return r.stdout

        if active_only:
            sq_out, queue_positions, source_tz = await asyncio.gather(
                _squeue(),
                _queue_positions(),
                source_tz_task,
            )
            sa_out = ""
        else:
            sq_out, sa_out, queue_positions, source_tz = await asyncio.gather(
                _squeue(),
                _sacct(),
                _queue_positions(),
                source_tz_task,
            )

        local: list[Job] = []
        seen: set[str] = set()
        by_id: dict[str, Job] = {}
        for line in sq_out.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 8:
                continue
            seen.add(parts[0])
            time_left = parts[6] if parts[6] not in ("", "N/A") else None
            job_start = _actual_start_for_state(parts[3], parts[7], source_tz)
            job = Job(
                cluster=c, job_id=parts[0], job_name=parts[1], partition=parts[2],
                state=parts[3], elapsed=parts[4], nodelist=parts[5],
                time_left=time_left, queue_position=queue_positions.get(parts[0]),
                start=job_start,
            )
            local.append(job)
            by_id[parts[0]] = job
        for line in sa_out.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 8:
                continue
            jid = parts[0]
            # Truncate sacct's CANCELLED+by labels for cleaner display.
            state = short_state(parts[3])
            job_start = _actual_start_for_state(state, parts[5], source_tz)
            job_end = to_kst_iso(parts[6], source_tz)
            if jid in seen:
                # squeue lingers in COMPLETING during a job's epilog/cleanup;
                # sacct already holds the real terminal outcome (e.g. PREEMPTED),
                # which the sacct-first detail view shows too. Overlay it so the
                # list and detail agree instead of the table looking "completed".
                existing = by_id.get(jid)
                if (
                    existing is not None
                    and existing.state.upper() == "COMPLETING"
                    and (state.upper() == "COMPLETED" or is_terminal_non_completed(state))
                ):
                    existing.state = state
                    existing.end = job_end
                continue
            local.append(Job(
                cluster=c, job_id=jid, job_name=parts[1], partition=parts[2],
                state=state, elapsed=parts[4], nodelist=parts[7],
                start=job_start, end=job_end,
            ))
        meta_by_job_id = await read_slurm_meta_many(host, [j.job_id for j in local])
        _attach_phase_metadata(local, meta_by_job_id)
        if not active_only:
            await _normalize_completed_eval_jobs(host, env.vars["LOG_DIR"], local, meta_by_job_id)
        return local

    per_cluster = await asyncio.gather(*[_for_cluster(c) for c in target_clusters])
    out: list[Job] = []
    for group in per_cluster:
        out.extend(group)
    return out


def _attach_phase_metadata(rows: list[Job], meta_by_job_id: dict[str, dict[str, str]]) -> None:
    for job in rows:
        meta = meta_by_job_id.get(job.job_id, {})
        phase, variant = resolve_identity(meta, job.job_name)
        job.phase = None if phase == "unknown" else phase
        job.variant = variant
        job.resume_of = meta.get("resume_of") or None
        job.resubmit_action = meta.get("resubmit_action") or None


_SACCT_DETAIL_FMT = "JobID,JobName,Partition,State,ExitCode,Start,End,Elapsed,NodeList,Reason"
_SACCT_GPU_DETAIL_FMT = f"{_SACCT_DETAIL_FMT},AllocTRES,ReqTRES"
_SQUEUE_DETAIL_FMT = "%i|%j|%P|%T|%V|%S|%M|%N|%R|%b"


def _gpu_count_from_tres(value: str | None) -> str | None:
    count = gpu_count_from_tres(value)
    return str(count) if count is not None else None


def _gpu_count_from_meta(meta: dict[str, str]) -> str | None:
    for key in ("eval_num_gpus", "train_num_gpus", "num_gpus"):
        value = (meta.get(key) or "").strip()
        if value:
            return value
    return None


def _attach_gpu_count(record: dict[str, str], meta: dict[str, str] | None = None) -> None:
    gpu_count = (
        _gpu_count_from_tres(record.get("AllocTRES"))
        or _gpu_count_from_tres(record.get("ReqTRES"))
        or _gpu_count_from_tres(record.get("TresPerNode"))
        or _gpu_count_from_meta(meta or {})
    )
    if gpu_count:
        record["GPUs"] = gpu_count


async def _sacct_job_record(host: str, cluster: str, job_id: str) -> dict[str, str] | None:
    for fmt in (_SACCT_GPU_DETAIL_FMT, _SACCT_DETAIL_FMT):
        r = await ssh_run(
            host,
            f'sacct -j {shlex.quote(job_id)} -X --parsable2 --format={fmt}',
            timeout=15.0,
        )
        if r.returncode != 0:
            continue
        lines = r.stdout.strip().splitlines()
        if len(lines) < 2:
            continue
        header = lines[0].split("|")
        row = lines[1].split("|")
        return {**dict(zip(header, row)), "cluster": cluster}
    return None


async def get_job(cluster: str, job_id: str) -> dict:
    """Return a flat dict of job fields. Dispatches on cluster type."""
    if cluster == "mlxp":
        from . import mlxp_jobs
        return await mlxp_jobs.get_job(job_id)
    env = await load_cluster(cluster)
    host = env.ssh_alias
    source_tz, meta, d = await asyncio.gather(
        scheduler_timezone(host),
        read_slurm_meta(host, job_id),
        _sacct_job_record(host, cluster, job_id),
    )
    if d:
        d["Start"] = _actual_start_for_state(d.get("State", ""), d.get("Start"), source_tz) or ""
        d["End"] = to_kst_iso(d.get("End"), source_tz) or ""
        _attach_gpu_count(d, meta)
        await _normalize_completed_eval_record(host, env.vars["LOG_DIR"], d, meta)
        return d

    # Fall back to squeue for jobs that don't have an sacct record yet.
    sq = await ssh_run(
        host,
        f'squeue -j {shlex.quote(job_id)} -h -o "{_SQUEUE_DETAIL_FMT}"',
        timeout=15.0,
    )
    if sq.returncode == 0 and sq.stdout.strip():
        parts = sq.stdout.strip().split("|")
        # parts: JobID|JobName|Partition|State|SubmitTime|StartTime|Elapsed|NodeList|Reason|TresPerNode
        keys = [
            "JobID", "JobName", "Partition", "State", "Submit", "Start",
            "Elapsed", "NodeList", "Reason", "TresPerNode",
        ]
        d = dict(zip(keys, parts))
        d["cluster"] = cluster
        d["Submit"] = to_kst_iso(d.get("Submit"), source_tz) or ""
        d["Start"] = _actual_start_for_state(d.get("State", ""), d.get("Start"), source_tz) or ""
        d.setdefault("ExitCode", "")
        d.setdefault("End", "")
        _attach_gpu_count(d, meta)
        await _normalize_completed_eval_record(host, env.vars["LOG_DIR"], d, meta)
        return d

    raise FileNotFoundError(f"no job record for {job_id} on {cluster}")


async def cancel_job(cluster: str, job_id: str) -> None:
    if cluster == "mlxp":
        from . import mlxp_jobs
        await mlxp_jobs.cancel_job(job_id)
        return
    env = await load_cluster(cluster)
    r = await ssh_run(env.ssh_alias, f"scancel {job_id}", timeout=15.0)
    if r.returncode != 0:
        raise RuntimeError(f"scancel failed: {r.stderr}")


EVAL_OVERRIDE_REASON = "Slurm exited nonzero after eval artifacts completed"


async def _eval_artifacts_completed(
    host: str,
    log_dir: str,
    job_id: str,
    job_name: str,
    state: str,
    meta: dict[str, str],
    record: dict | None = None,
) -> bool:
    """True when a terminal-but-nonzero eval job actually finished writing its
    artifacts, so callers can rewrite its state to COMPLETED."""
    if not is_terminal_non_completed(state):
        return False
    phase, variant = resolve_identity(meta, job_name, record)
    if phase != "eval" or not variant:
        return False
    try:
        return await eval_job_completed_from_log_dir(host, log_dir, job_id, variant, meta)
    except Exception:
        return False


async def _normalize_completed_eval_jobs(
    host: str,
    log_dir: str,
    rows: list[Job],
    meta_by_job_id: dict[str, dict[str, str]],
) -> None:
    async def _one(job: Job) -> None:
        meta = meta_by_job_id.get(job.job_id, {})
        if await _eval_artifacts_completed(
            host, log_dir, job.job_id, job.job_name, job.state, meta
        ):
            job.state = "COMPLETED"
            job.reason = job.reason or EVAL_OVERRIDE_REASON

    await asyncio.gather(*(_one(j) for j in rows), return_exceptions=True)


async def _normalize_completed_eval_record(
    host: str,
    log_dir: str,
    record: dict,
    meta: dict[str, str] | None = None,
) -> None:
    state = str(record.get("State") or "")
    meta = meta or {}
    if not await _eval_artifacts_completed(
        host,
        log_dir,
        str(record.get("JobID") or ""),
        str(record.get("JobName") or ""),
        state,
        meta,
        record,
    ):
        return
    record.setdefault("SlurmState", state)
    record["State"] = "COMPLETED"
    if not record.get("Reason") or record.get("Reason") == "None":
        record["Reason"] = EVAL_OVERRIDE_REASON
