"""Slurm job listing + status + cancel."""

import asyncio

from pydantic import BaseModel

from .clusters import ClusterEnv, load_cluster, list_clusters
from .eval_completion import eval_job_completed_from_log_dir
from .job_identity import phase_variant_from_meta, resolve_phase_and_variant
from .ssh import ssh_run


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


_SQUEUE_FMT = "%i|%j|%P|%T|%M|%R|%L|%S"
_SACCT_LIST_FMT = "JobID,JobName,Partition,State,Elapsed,Start,End,NodeList"
_ACTIVE_STATES = {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"}


async def list_jobs(clusters: list[str] | None = None, hours: int = 24) -> list[Job]:
    """Active jobs from `squeue` + recent finished from `sacct` (last `hours`).

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
                return await mlxp_jobs.list_jobs()
            except Exception:
                return []
        try:
            env = await load_cluster(c)
        except FileNotFoundError:
            return []
        host = env.ssh_alias

        async def _squeue() -> str | None:
            try:
                r = await ssh_run(host, f'squeue -u "$USER" -h -o "{_SQUEUE_FMT}"', timeout=15.0)
            except Exception:
                return None
            return r.stdout if r.returncode == 0 else None

        async def _sacct() -> str | None:
            try:
                r = await ssh_run(
                    host,
                    f'sacct -X -u "$USER" -S now-{hours}hours -P -n -o {_SACCT_LIST_FMT}',
                    timeout=30.0,
                )
            except Exception:
                return None
            return r.stdout if r.returncode == 0 else None

        sq_out, sa_out = await asyncio.gather(_squeue(), _sacct())

        local: list[Job] = []
        seen: set[str] = set()
        if sq_out is not None:
            for line in sq_out.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 8:
                    continue
                seen.add(parts[0])
                time_left = parts[6] if parts[6] not in ("", "N/A") else None
                start = parts[7] if parts[7] not in ("", "N/A", "Unknown") else None
                local.append(Job(
                    cluster=c, job_id=parts[0], job_name=parts[1], partition=parts[2],
                    state=parts[3], elapsed=parts[4], nodelist=parts[5],
                    time_left=time_left, start=start,
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
                start = parts[5] if parts[5] not in ("", "Unknown") else None
                end = parts[6] if parts[6] not in ("", "Unknown") else None
                local.append(Job(
                    cluster=c, job_id=jid, job_name=parts[1], partition=parts[2],
                    state=state, elapsed=parts[4], nodelist=parts[7],
                    start=start, end=end,
                ))
        await _normalize_completed_eval_jobs(host, env.vars["LOG_DIR"], local)
        return local

    per_cluster = await asyncio.gather(*[_for_cluster(c) for c in target_clusters])
    out: list[Job] = []
    for group in per_cluster:
        out.extend(group)
    return out


_SACCT_FMT = "JobID,JobName,Partition,State,ExitCode,Start,End,Elapsed,NodeList,Reason"
_SQUEUE_DETAIL_FMT = "%i|%j|%P|%T|%V|%S|%M|%R"


async def get_job(cluster: str, job_id: str) -> dict:
    """Return a flat dict of job fields. Dispatches on cluster type."""
    if cluster == "mlxp":
        from . import mlxp_jobs
        return await mlxp_jobs.get_job(job_id)
    env = await load_cluster(cluster)
    r = await ssh_run(env.ssh_alias,
                      f'sacct -j {job_id} -X --parsable2 --format={_SACCT_FMT}',
                      timeout=15.0)
    if r.returncode == 0:
        lines = r.stdout.strip().splitlines()
        if len(lines) >= 2:
            header = lines[0].split("|")
            row = lines[1].split("|")
            d = {**dict(zip(header, row)), "cluster": cluster}
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
            meta = await _read_slurm_meta(host, job.job_id)
            p, v = phase_variant_from_meta(meta)
            if p and v:
                phase, variant = p, v
        elif phase == "eval":
            meta = await _read_slurm_meta(host, job.job_id)
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
    meta = await _read_slurm_meta(host, str(record.get("JobID") or ""))
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


async def _read_slurm_meta(host: str, job_id: str) -> dict[str, str]:
    r = await ssh_run(
        host,
        f"cat $HOME/.train-eval-web/jobs/{job_id}.meta 2>/dev/null",
        timeout=10.0,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    fields: dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields
