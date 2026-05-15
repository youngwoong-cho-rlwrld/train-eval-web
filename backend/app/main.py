"""FastAPI entrypoint."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from . import clusters, datasets, details, jobs, partitions, submit, variants
from .ssh import ssh_tail_lines


app = FastAPI(title="train-eval-web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── clusters ──

@app.get("/api/clusters")
async def get_clusters():
    return {"clusters": clusters.list_clusters()}


@app.get("/api/clusters/{name}")
async def get_cluster(name: str):
    try:
        env = await clusters.load_cluster(name)
    except FileNotFoundError:
        raise HTTPException(404, f"cluster {name} not found")
    return env


@app.get("/api/clusters/{name}/partitions", response_model=list[partitions.PartitionInfo])
async def get_cluster_partitions(name: str):
    try:
        return await partitions.list_partitions(name)
    except FileNotFoundError:
        raise HTTPException(404, f"cluster {name} not found")
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.get("/api/clusters/{name}/datasets", response_model=list[datasets.DatasetInfo])
async def get_cluster_datasets(name: str):
    try:
        return await datasets.list_datasets(name)
    except FileNotFoundError:
        raise HTTPException(404, f"cluster {name} not found")
    except RuntimeError as e:
        raise HTTPException(500, str(e))


# ── variants ──

@app.get("/api/variants")
async def get_variants():
    return {"variants": variants.list_variants()}


@app.get("/api/variants/{name}")
async def get_variant(name: str):
    try:
        return await variants.load_variant(name)
    except FileNotFoundError:
        raise HTTPException(404, f"variant {name} not found")


# ── submit ──

@app.post("/api/submit", response_model=submit.SubmitResponse)
async def post_submit(req: submit.SubmitRequest):
    try:
        return await submit.submit(req)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


# ── jobs ──

@app.get("/api/jobs")
async def get_jobs(cluster: str | None = None, hours: int = 24):
    target = [cluster] if cluster else None
    js = await jobs.list_jobs(target, hours=hours)
    return {"jobs": [j.model_dump() for j in js]}


@app.get("/api/jobs/{cluster}/{job_id}")
async def get_job(cluster: str, job_id: str):
    try:
        return await jobs.get_job(cluster, job_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.get("/api/jobs/{cluster}/{job_id}/details", response_model=details.JobDetails)
async def get_job_details(cluster: str, job_id: str):
    try:
        return await details.get_details(cluster, job_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.delete("/api/jobs/{cluster}/{job_id}")
async def delete_job(cluster: str, job_id: str):
    try:
        await jobs.cancel_job(cluster, job_id)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"status": "cancelled"}


@app.get("/api/jobs/{cluster}/{job_id}/logs")
async def stream_logs(cluster: str, job_id: str, stream: str = "out"):
    """Server-Sent Events stream of log lines.

    stream:
      out  — slurm stdout (.out)
      err  — slurm stderr (.err)
      isaac — Isaac Sim server logs from the eval body's $EXP_DIR/logs/server_*.log
    """
    if stream not in ("out", "err", "isaac"):
        raise HTTPException(400, "stream must be 'out', 'err', or 'isaac'")
    try:
        env = await clusters.load_cluster(cluster)
    except FileNotFoundError:
        raise HTTPException(404, f"cluster {cluster} not found")

    if stream == "isaac":
        det = await details.get_details(cluster, job_id)
        if not det.paths.isaac_logs_glob:
            raise HTTPException(400, "isaac logs only available for eval jobs")
        pattern = det.paths.isaac_logs_glob
    else:
        log_dir = env.vars["LOG_DIR"]
        pattern = f"{log_dir}/*_{job_id}.{stream}"

    async def gen():
        async for line in ssh_tail_lines(env.ssh_alias, pattern):
            yield {"event": "line", "data": line}

    return EventSourceResponse(gen())
