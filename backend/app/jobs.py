"""Slurm job listing + status + cancel."""

from pydantic import BaseModel

from .clusters import ClusterEnv, load_cluster, list_clusters
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


async def list_jobs(clusters: list[str] | None = None, hours: int = 24) -> list[Job]:
    """Active jobs from `squeue` + recent finished from `sacct` (last `hours`).

    squeue takes precedence on overlap (its state is fresher than sacct's
    for jobs that just finished).
    """
    from . import mlxp_jobs
    target_clusters = clusters or list_clusters()
    out: list[Job] = []
    for c in target_clusters:
        if c == "mlxp":
            try:
                out.extend(await mlxp_jobs.list_jobs())
            except Exception:
                pass
            continue
        try:
            env = await load_cluster(c)
        except FileNotFoundError:
            continue
        host = env.ssh_alias
        seen: set[str] = set()

        # Active jobs (PENDING / RUNNING / COMPLETING / etc.).
        try:
            sq = await ssh_run(host, f'squeue -u "$USER" -h -o "{_SQUEUE_FMT}"', timeout=15.0)
        except Exception:
            sq = None
        if sq and sq.returncode == 0:
            for line in sq.stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 8:
                    continue
                seen.add(parts[0])
                time_left = parts[6] if parts[6] not in ("", "N/A") else None
                start = parts[7] if parts[7] not in ("", "N/A", "Unknown") else None
                out.append(Job(
                    cluster=c, job_id=parts[0], job_name=parts[1], partition=parts[2],
                    state=parts[3], elapsed=parts[4], nodelist=parts[5],
                    time_left=time_left, start=start,
                ))

        # Recent finished jobs from sacct.
        try:
            sa = await ssh_run(
                host,
                f'sacct -X -u "$USER" -S now-{hours}hours -P -n -o {_SACCT_LIST_FMT}',
                timeout=30.0,
            )
        except Exception:
            sa = None
        if sa and sa.returncode == 0:
            for line in sa.stdout.strip().splitlines():
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
                out.append(Job(
                    cluster=c, job_id=jid, job_name=parts[1], partition=parts[2],
                    state=state, elapsed=parts[4], nodelist=parts[7],
                    start=start, end=end,
                ))
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
            return {**dict(zip(header, row)), "cluster": cluster}

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
