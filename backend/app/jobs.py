"""Slurm job listing + status + cancel."""

import asyncio
import shlex

from pydantic import BaseModel

from .clusters import load_cluster, list_clusters
from .eval_completion import eval_job_completed_from_log_dir
from .job_identity import phase_variant_from_meta, resolve_phase_and_variant
from .slurm_meta import read_slurm_meta
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
    start: str | None = None         # ISO timestamp, or None / "Unknown"
    end: str | None = None           # ISO timestamp, or None / "Unknown"
    phase: str | None = None
    variant: str | None = None


_SQUEUE_FMT = "%i|%j|%P|%T|%M|%R|%L|%S"
_SACCT_LIST_FMT = "JobID,JobName,Partition,State,Elapsed,Start,End,NodeList"
_ACTIVE_STATES = {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"}


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

    async def _for_cluster(c: str) -> list[Job]:
        if c == "mlxp":
            try:
                return await mlxp_jobs.list_jobs(start=start, end=end)
            except Exception:
                return []
        try:
            env = await load_cluster(c)
        except FileNotFoundError:
            return []
        host = env.ssh_alias
        source_tz_task = asyncio.create_task(scheduler_timezone(host))

        async def _squeue() -> str | None:
            try:
                r = await ssh_run(host, f'squeue -u "$USER" -h -o "{_SQUEUE_FMT}"', timeout=15.0)
            except Exception:
                return None
            return r.stdout if r.returncode == 0 else None

        async def _sacct() -> str | None:
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
            except Exception:
                return None
            return r.stdout if r.returncode == 0 else None

        sq_out, sa_out, source_tz = await asyncio.gather(_squeue(), _sacct(), source_tz_task)

        local: list[Job] = []
        seen: set[str] = set()
        if sq_out is not None:
            for line in sq_out.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 8:
                    continue
                seen.add(parts[0])
                time_left = parts[6] if parts[6] not in ("", "N/A") else None
                job_start = to_kst_iso(parts[7], source_tz)
                local.append(Job(
                    cluster=c, job_id=parts[0], job_name=parts[1], partition=parts[2],
                    state=parts[3], elapsed=parts[4], nodelist=parts[5],
                    time_left=time_left, start=job_start,
                ))
        if sa_out is not None:
            for line in sa_out.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 8:
                    continue
                jid = parts[0]
                if jid in seen:
                    continue
                # Truncate sacct's CANCELLED+by labels for cleaner display.
                state = parts[3].split(" ")[0]
                job_start = to_kst_iso(parts[5], source_tz)
                job_end = to_kst_iso(parts[6], source_tz)
                local.append(Job(
                    cluster=c, job_id=jid, job_name=parts[1], partition=parts[2],
                    state=state, elapsed=parts[4], nodelist=parts[7],
                    start=job_start, end=job_end,
                ))
        await _attach_phase_metadata(host, local)
        await _normalize_completed_eval_jobs(host, env.vars["LOG_DIR"], local)
        return local

    per_cluster = await asyncio.gather(*[_for_cluster(c) for c in target_clusters])
    out: list[Job] = []
    for group in per_cluster:
        out.extend(group)
    return out


async def _attach_phase_metadata(host: str, rows: list[Job]) -> None:
    async def _one(job: Job) -> None:
        meta = await read_slurm_meta(host, job.job_id)
        phase, variant = phase_variant_from_meta(meta)
        if not phase or not variant:
            phase, variant = resolve_phase_and_variant(job.job_name)
        job.phase = None if phase == "unknown" else phase
        job.variant = variant

    await asyncio.gather(*(_one(j) for j in rows))


_SACCT_FMT = "JobID,JobName,Partition,State,ExitCode,Start,End,Elapsed,NodeList,Reason"
_SQUEUE_DETAIL_FMT = "%i|%j|%P|%T|%V|%S|%M|%R"


async def get_job(cluster: str, job_id: str) -> dict:
    """Return a flat dict of job fields. Dispatches on cluster type."""
    if cluster == "mlxp":
        from . import mlxp_jobs
        return await mlxp_jobs.get_job(job_id)
    env = await load_cluster(cluster)
    source_tz = await scheduler_timezone(env.ssh_alias)
    r = await ssh_run(env.ssh_alias,
                      f'sacct -j {job_id} -X --parsable2 --format={_SACCT_FMT}',
                      timeout=15.0)
    if r.returncode == 0:
        lines = r.stdout.strip().splitlines()
        if len(lines) >= 2:
            header = lines[0].split("|")
            row = lines[1].split("|")
            d = {**dict(zip(header, row)), "cluster": cluster}
            d["Start"] = to_kst_iso(d.get("Start"), source_tz) or ""
            d["End"] = to_kst_iso(d.get("End"), source_tz) or ""
            await _normalize_completed_eval_record(env.ssh_alias, env.vars["LOG_DIR"], d)
            return d

    # Fall back to squeue for jobs that don't have an sacct record yet.
    sq = await ssh_run(env.ssh_alias,
                       f'squeue -j {job_id} -h -o "{_SQUEUE_DETAIL_FMT}"',
                       timeout=15.0)
    if sq.returncode == 0 and sq.stdout.strip():
        parts = sq.stdout.strip().split("|")
        # parts: JobID|JobName|Partition|State|SubmitTime|StartTime|Elapsed|Reason
        keys = ["JobID", "JobName", "Partition", "State", "Submit", "Start", "Elapsed", "Reason"]
        d = dict(zip(keys, parts))
        d["cluster"] = cluster
        d["Submit"] = to_kst_iso(d.get("Submit"), source_tz) or ""
        d["Start"] = to_kst_iso(d.get("Start"), source_tz) or ""
        d.setdefault("ExitCode", "")
        d.setdefault("End", "")
        d.setdefault("NodeList", parts[7] if len(parts) > 7 else "")
        await _normalize_completed_eval_record(env.ssh_alias, env.vars["LOG_DIR"], d)
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


def _terminal_non_completed(state: str) -> bool:
    upper = state.upper()
    if upper in _ACTIVE_STATES or upper.startswith("COMPLET"):
        return False
    return upper.startswith((
        "FAIL",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPT",
        "CANCEL",
    ))


async def _normalize_completed_eval_jobs(host: str, log_dir: str, rows: list[Job]) -> None:
    async def _one(job: Job) -> None:
        if not _terminal_non_completed(job.state):
            return
        meta: dict[str, str] = {}
        phase, variant = resolve_phase_and_variant(job.job_name)
        if phase != "eval" or not variant:
            meta = await read_slurm_meta(host, job.job_id)
            p, v = phase_variant_from_meta(meta)
            if p and v:
                phase, variant = p, v
        elif phase == "eval":
            meta = await read_slurm_meta(host, job.job_id)
        if phase != "eval" or not variant:
            return
        try:
            if await eval_job_completed_from_log_dir(host, log_dir, job.job_id, variant, meta):
                job.state = "COMPLETED"
                job.reason = job.reason or "Slurm exited nonzero after eval artifacts completed"
        except Exception:
            return

    await asyncio.gather(*(_one(j) for j in rows))


async def _normalize_completed_eval_record(host: str, log_dir: str, record: dict) -> None:
    state = str(record.get("State") or "")
    if not _terminal_non_completed(state):
        return
    meta = await read_slurm_meta(host, str(record.get("JobID") or ""))
    phase, variant = resolve_phase_and_variant(str(record.get("JobName") or ""), record)
    if phase != "eval" or not variant:
        p, v = phase_variant_from_meta(meta)
        if p and v:
            phase, variant = p, v
    if phase != "eval" or not variant:
        return
    try:
        completed = await eval_job_completed_from_log_dir(
            host,
            log_dir,
            str(record.get("JobID") or ""),
            variant,
            meta,
        )
    except Exception:
        return
    if not completed:
        return
    record.setdefault("SlurmState", state)
    record["State"] = "COMPLETED"
    if not record.get("Reason") or record.get("Reason") == "None":
        record["Reason"] = "Slurm exited nonzero after eval artifacts completed"
