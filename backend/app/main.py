"""FastAPI entrypoint."""

import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import clusters, copy_checkpoint, datasets, details, flags, job_resume, jobs, mlxp, mlxp_submit, partitions, results, submission_snapshot, submit, training_models, variants, wandb_auth
from .paths import CLUSTER_STAGING_REL
from .ssh import ssh_tail_lines
from .variant_values import variant_int


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


@app.get("/api/mlxp/gpus", response_model=list[mlxp.MlxpNode])
async def get_mlxp_gpus():
    try:
        return await mlxp.list_nodes()
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/api/clusters/{name}/path-exists")
async def get_path_exists(name: str, path: str):
    """Check whether `path` exists (file or dir) on the cluster.

    Used by the submit page to verify a user-typed eval checkpoint path
    before launching. Slurm-only; mlxp eval isn't wired yet.
    """
    if name == "mlxp":
        return {"exists": False, "kind": None}
    if not path or not path.strip():
        return {"exists": False, "kind": None}
    try:
        env = await clusters.load_cluster(name)
    except FileNotFoundError:
        raise HTTPException(404, f"cluster {name} not found")
    from .ssh import ssh_run
    import shlex
    p = shlex.quote(path.strip())
    cmd = f'if [ -d {p} ]; then echo dir; elif [ -f {p} ]; then echo file; else echo none; fi'
    r = await ssh_run(env.ssh_alias, cmd, timeout=10.0)
    kind = r.stdout.strip()
    if kind not in ("dir", "file"):
        return {"exists": False, "kind": None}
    return {"exists": True, "kind": kind}


@app.get("/api/clusters/{name}/datasets", response_model=list[datasets.DatasetInfo])
async def get_cluster_datasets(name: str, path: str | None = None):
    if name not in clusters.list_clusters():
        raise HTTPException(404, f"cluster {name} not found")
    try:
        return await datasets.list_datasets(name, path)
    except FileNotFoundError:
        raise HTTPException(404, f"cluster {name} not found")
    except RuntimeError as e:
        raise HTTPException(503, str(e))


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


@app.get("/api/variants/{name}/flags")
async def get_variant_flags(name: str, cluster: str, phase: str = "train"):
    try:
        v = await variants.load_variant(name)
    except FileNotFoundError:
        raise HTTPException(404, f"variant {name} not found")
    out = flags.flags_for(v, cluster, phase)
    return {"flags": [{"flag": f, "value": val} for f, val in out]}


@app.get("/api/variants/{name}/selected-checkpoint")
async def get_selected_checkpoint(name: str, cluster: str):
    """The checkpoint path the eval body would pick at runtime, mirroring
    lib/eval_body_n16.sh's nested-then-flat lookup. Slurm-only — MLXP eval
    isn't wired."""
    if cluster == "mlxp":
        return {"path": None, "step": None}
    try:
        env = await clusters.load_cluster(cluster)
    except FileNotFoundError:
        raise HTTPException(404, f"cluster {cluster} not found")
    cmd = (
        f'D=$HOME/.train-eval-web/experiments/{name}/checkpoints; '
        f'p=$(ls -d "$D"/*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); '
        f'[ -z "$p" ] && p=$(ls -d "$D"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); '
        f'printf "%s\\n" "$p"'
    )
    from .ssh import ssh_run
    r = await ssh_run(env.ssh_alias, cmd, timeout=15.0)
    path = r.stdout.strip()
    if not path:
        return {"path": None, "step": None}
    try:
        step = int(path.rsplit("-", 1)[-1])
    except ValueError:
        step = None
    return {"path": path, "step": step}


# ── submit ──

class ConfigPreviewFlag(BaseModel):
    flag: str
    value: str


class SubmitConfigPreview(BaseModel):
    path: str | None = None
    model_id: str | None = None
    model_label: str | None = None
    model_repo_path: str | None = None
    model_repo_error: str | None = None
    text: str
    flags: list[ConfigPreviewFlag]

@app.get("/api/submit/git-status", response_model=submission_snapshot.GitStatus)
async def get_submit_git_status(cluster: str, variant: str):
    try:
        v = await variants.load_variant(variant)
        model = training_models.resolve_training_model(v)
        repo_label = submission_snapshot.training_repo_label(model)
        if cluster == "mlxp":
            return await submission_snapshot.mlxp_git_status(
                repo_path=mlxp_submit.mlxp_training_repo_path(model),
                repo_label=repo_label,
            )

        env = await clusters.load_cluster(cluster)
        return await submission_snapshot.slurm_git_status(
            host=env.ssh_alias,
            repo_path=submission_snapshot.slurm_training_repo_path(env.vars, model),
            repo_label=repo_label,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.post("/api/submit/config-preview", response_model=SubmitConfigPreview)
async def post_submit_config_preview(req: submit.SubmitRequest):
    try:
        variant = await variants.load_variant(req.variant)
        model = training_models.resolve_training_model(variant)
        job_name = submit.resolve_job_name(req.job_name, req.phase, req.variant)
        partition = req.partition
        node = req.node
        env = None
        model_repo_path: str | None = None
        model_repo_error: str | None = None
        if req.cluster != "mlxp":
            env = await clusters.load_cluster(req.cluster)
            partition = partition or env.vars["PARTITION"]
            try:
                model_repo_path = submission_snapshot.slurm_training_repo_path(env.vars, model)
            except ValueError as e:
                model_repo_error = str(e)
        elif not node:
            node = mlxp_submit.DEFAULT_NODE
            try:
                model_repo_path = mlxp_submit.mlxp_training_repo_path(model)
            except ValueError as e:
                model_repo_error = str(e)
        else:
            try:
                model_repo_path = mlxp_submit.mlxp_training_repo_path(model)
            except ValueError as e:
                model_repo_error = str(e)

        path: str | None = None
        if req.phase == "train":
            train_num_gpus = req.train_num_gpus or variant_int(variant, "TRAIN_NUM_GPUS", 2)
            train_max_steps = req.train_max_steps or variant_int(variant, "MAX_STEPS", 30000)
            train_save_steps = req.train_save_steps or variant_int(variant, "SAVE_STEPS", 1000)
            train_global_batch_size = req.train_global_batch_size
            if train_global_batch_size is None:
                for key in ("TRAIN_GLOBAL_BATCH_SIZE", "GLOBAL_BATCH_SIZE"):
                    raw = variant.vars.get(key)
                    if raw:
                        try:
                            train_global_batch_size = int(raw)
                            break
                        except ValueError:
                            pass
                if train_global_batch_size is None:
                    train_global_batch_size = variant_int(variant, "TRAIN_BATCH_SIZE", 64) * train_num_gpus
            if req.train_global_batch_size is not None and model.family == "n1.5":
                if req.train_global_batch_size % train_num_gpus != 0:
                    raise ValueError("train_global_batch_size must be divisible by train_num_gpus for n1.5 training")

            suffix = submission_snapshot.snapshot_suffix(job_name)
            if req.cluster == "mlxp":
                path = f"{mlxp_submit.MLXP_EXPERIMENTS_DIR}/{req.variant}/config_{suffix}.sh"
            else:
                path = f"$HOME/{CLUSTER_STAGING_REL}/experiments/{req.variant}/config_{suffix}.sh"
            text = submission_snapshot.render_training_config_snapshot(
                base_config=variant.raw,
                variant=req.variant,
                model=model.family,
                job_name=job_name,
                cluster=req.cluster,
                partition=partition,
                node=node,
                dataset_override=req.dataset_override,
                extra_args=req.extra_args,
                train_num_gpus=train_num_gpus,
                train_global_batch_size=train_global_batch_size,
                train_max_steps=train_max_steps,
                train_save_steps=train_save_steps,
                git=None,
            )
        elif req.phase == "eval":
            eval_sets = submit._normalize_eval_sets(req.eval_sets)
            text = submission_snapshot.render_eval_config_preview(
                base_config=variant.raw,
                variant=req.variant,
                job_name=job_name,
                cluster=req.cluster,
                partition=partition,
                node=node,
                dataset_override=req.dataset_override,
                eval_n_episodes=req.eval_n_episodes,
                eval_n_runs=req.eval_n_runs,
                eval_sets=eval_sets,
                eval_overwrite_results=req.eval_overwrite_results,
                checkpoint_path=req.checkpoint_path,
                extra_args=req.extra_args,
            )
        else:
            raise ValueError(f"unsupported phase: {req.phase}")

        effective_variant = await variants.parse_variant_text(req.variant, text)
        out = flags.flags_for(effective_variant, req.cluster, req.phase)
        return {
            "path": path,
            "model_id": model.id,
            "model_label": model.label,
            "model_repo_path": model_repo_path,
            "model_repo_error": model_repo_error,
            "text": text,
            "flags": [{"flag": f, "value": val} for f, val in out],
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.post("/api/submit")
async def post_submit(req: submit.SubmitRequest):
    """Dispatches to the per-cluster submitter.

    - kakao / skt → slurm sbatch over SSH (submit.submit)
    - mlxp        → render+apply a k8s Job (mlxp_submit.submit_mlxp)
    """
    try:
        if req.cluster == "mlxp":
            if req.phase != "train":
                raise ValueError("MLXP currently supports phase=train only")
            # GPU count defaults to TRAIN_NUM_GPUS; submit-time overrides use
            # the same request fields as Slurm and then map to MLXP CPU/RAM.
            v = await variants.load_variant(req.variant)
            try:
                num_gpus = req.train_num_gpus or int(v.vars.get("TRAIN_NUM_GPUS", "2"))
            except ValueError:
                raise ValueError(
                    f"variant {req.variant}: TRAIN_NUM_GPUS must be an integer"
                )
            mlxp_req = mlxp_submit.MlxpSubmitRequest(
                variant=req.variant,
                num_gpus=num_gpus,
                global_batch_size=req.train_global_batch_size,
                max_steps=req.train_max_steps,
                save_steps=req.train_save_steps,
                node=req.node,
                dataset_override=req.dataset_override,
                extra_args=req.extra_args,
                job_name=req.job_name,
                commit_dirty_changes=req.commit_dirty_changes,
            )
            r = await mlxp_submit.submit_mlxp(mlxp_req)
            return {
                "job_id": r.job_id,
                "job_name": r.job_name,
                "partition": f"mlxp/{mlxp_req.num_gpus}gpu",
                "sbatch_cmd": "kubectl apply (rendered Job YAML)",
                "rsync_stdout": "",
                "sbatch_stdout": r.apply_stdout,
            }
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
async def get_job_details(cluster: str, job_id: str, include_gpu: bool = False):
    try:
        return await details.get_details(cluster, job_id, include_gpu=include_gpu)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.post("/api/jobs/{cluster}/{job_id}/resume", response_model=submit.SubmitResponse)
async def post_resume_job(cluster: str, job_id: str):
    try:
        return await job_resume.resume_timed_out_job(cluster, job_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.delete("/api/jobs/{cluster}/{job_id}")
async def delete_job(cluster: str, job_id: str):
    try:
        await jobs.cancel_job(cluster, job_id)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"status": "cancelled"}


# ── results ──

@app.get("/api/results", response_model=results.ResultsResponse)
async def get_results(cluster: str | None = None):
    try:
        return await results.list_results(cluster)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


def _sse_next_line(request: Request) -> int:
    last_event_id = request.headers.get("last-event-id", "").strip()
    try:
        return max(1, int(last_event_id) + 1)
    except ValueError:
        return 1


@app.get("/api/jobs/{cluster}/{job_id}/logs")
async def stream_logs(request: Request, cluster: str, job_id: str, stream: str = "out"):
    """Server-Sent Events stream of log lines.

    stream (slurm clusters):
      out  — slurm stdout (.out)
      err  — slurm stderr (.err)
      isaac — Isaac Sim server logs ($EXP_DIR/logs/server_*.log)
    MLXP has a single container log, so `stream` is ignored.
    """
    start_line = _sse_next_line(request)
    if cluster == "mlxp":
        from . import mlxp_jobs
        async def gen_mlxp():
            line_no = start_line
            while not await request.is_disconnected():
                saw_line = False
                try:
                    async for line in mlxp_jobs.tail_logs(job_id, start_line=line_no):
                        if await request.is_disconnected():
                            return
                        saw_line = True
                        yield {"event": "line", "id": str(line_no), "retry": 10000, "data": line}
                        line_no += 1
                except RuntimeError as e:
                    yield {"event": "line", "id": str(line_no), "retry": 10000, "data": f"(kubectl error: {e})"}
                    line_no += 1
                await asyncio.sleep(1 if saw_line else 2)
        return EventSourceResponse(gen_mlxp())

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
        line_no = start_line
        while not await request.is_disconnected():
            saw_line = False
            async for line in ssh_tail_lines(env.ssh_alias, pattern, start_line=line_no):
                if await request.is_disconnected():
                    return
                saw_line = True
                yield {"event": "line", "id": str(line_no), "retry": 10000, "data": line}
                line_no += 1
            await asyncio.sleep(1 if saw_line else 2)

    return EventSourceResponse(gen())


# ── wandb ──

@app.get("/api/wandb/status", response_model=wandb_auth.WandbStatus)
async def get_wandb_status():
    return await wandb_auth.get_status()


@app.post("/api/wandb/login", response_model=wandb_auth.WandbStatus)
async def post_wandb_login(req: wandb_auth.LoginRequest):
    if not req.key.strip():
        raise HTTPException(400, "key must not be empty")
    return await wandb_auth.login(req.key)


@app.post("/api/wandb/project", response_model=wandb_auth.WandbStatus)
async def post_wandb_project(req: wandb_auth.ProjectRequest):
    if not req.project.strip():
        raise HTTPException(400, "project must not be empty")
    return await wandb_auth.set_project_endpoint(req.project)


# ── flags ──

@app.get("/api/jobs/{cluster}/{job_id}/flags")
async def get_job_flags(cluster: str, job_id: str):
    """All flags the training/eval entrypoint receives for this job."""
    det = await details.get_details(cluster, job_id)
    if not det.variant:
        return {"flags": []}
    try:
        if det.config_snapshot and det.config_snapshot.text:
            v = await variants.parse_variant_text(det.variant, det.config_snapshot.text)
        else:
            v = await variants.load_variant(det.variant)
    except FileNotFoundError:
        raise HTTPException(404, f"variant {det.variant} not found")
    out = flags.flags_for(v, cluster, det.phase)
    if cluster != "mlxp" and det.phase == "eval":
        env = await clusters.load_cluster(cluster)
        meta = await details._read_slurm_meta(env.ssh_alias, job_id)
        envs_override = (
            meta.get("eval_num_envs_per_gpu")
            or meta.get("eval_parallel_sims_per_gpu")
            or ""
        ).strip()
        try:
            if int(envs_override) > submit.MAX_EVAL_NUM_ENVS_PER_GPU:
                envs_override = str(submit.MAX_EVAL_NUM_ENVS_PER_GPU)
        except ValueError:
            pass
        overrides = {
            "EVAL_NUM_ENVS_PER_GPU": envs_override,
            "--n-episodes": (meta.get("eval_n_episodes") or "").strip(),
            "--n-runs": (meta.get("eval_n_runs") or "").strip(),
            "(eval_sets)": (meta.get("eval_sets") or "").strip(),
        }
        overrides = {k: v for k, v in overrides.items() if v}
        if overrides:
            rewritten: list[tuple[str, str]] = []
            replaced: set[str] = set()
            for flag, val in out:
                if flag in overrides:
                    rewritten.append((flag, overrides[flag]))
                    replaced.add(flag)
                else:
                    rewritten.append((flag, val))
            for flag, value in overrides.items():
                if flag not in replaced:
                    rewritten.append((flag, value))
            out = rewritten
    return {"flags": [{"flag": f, "value": val} for f, val in out]}


# ── copy checkpoint ──

@app.get(
    "/api/jobs/{cluster}/{job_id}/checkpoints",
    response_model=list[copy_checkpoint.CheckpointEntry],
)
async def get_checkpoints(cluster: str, job_id: str):
    try:
        return await copy_checkpoint.list_checkpoints(cluster, job_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.post(
    "/api/jobs/{cluster}/{job_id}/copy-checkpoint",
    response_model=copy_checkpoint.CopyCheckpointStartResponse,
)
async def post_copy_checkpoint(
    cluster: str, job_id: str, req: copy_checkpoint.CopyCheckpointRequest
):
    try:
        copy_id = await copy_checkpoint.start_copy(
            src_cluster=cluster,
            src_job=job_id,
            dest_cluster=req.dest_cluster,
            sources=req.sources,
            dest_path_root=req.dest_path_root,
            delete_source=req.delete_source,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    return copy_checkpoint.CopyCheckpointStartResponse(copy_id=copy_id)


@app.get(
    "/api/copy-jobs/{copy_id}",
    response_model=copy_checkpoint.CopyJobStatus,
)
async def get_copy_status(copy_id: str):
    status = copy_checkpoint.get_copy_status(copy_id)
    if not status:
        raise HTTPException(404, f"copy job {copy_id} not found")
    return status


@app.post("/api/copy-jobs/{copy_id}/cancel")
async def post_cancel_copy(copy_id: str):
    if not copy_checkpoint.cancel_copy(copy_id):
        raise HTTPException(404, f"copy job {copy_id} not running")
    return {"status": "cancelled"}
