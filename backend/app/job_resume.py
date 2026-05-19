"""Resume timed-out jobs from their original train/eval intent."""

from . import details, jobs, submit
from .clusters import load_cluster


def _is_timeout(state: str) -> bool:
    return state.upper().startswith("TIMEOUT")


def _original_phase(phase: str) -> str:
    # Historical jobs may have phase=resume in their sidecar/job_name. Treat
    # those as training jobs; new submissions never use resume as a phase.
    return "train" if phase == "resume" else phase


async def resume_timed_out_job(cluster: str, job_id: str) -> submit.SubmitResponse:
    if cluster == "mlxp":
        raise ValueError("MLXP timed-out job resume is not supported")

    record = await jobs.get_job(cluster, job_id)
    state = str(record.get("State") or "")
    if not _is_timeout(state):
        raise ValueError(f"job {job_id} on {cluster} is {state or 'unknown'}, not TIMEOUT")

    det = await details.get_details(cluster, job_id)
    variant = det.variant
    if not variant:
        raise ValueError(f"cannot resume job {job_id}: variant is unknown")

    phase = _original_phase(det.phase)
    if phase not in ("train", "eval"):
        raise ValueError(f"cannot resume job {job_id}: phase is {det.phase or 'unknown'}")

    partition = str(record.get("Partition") or "").strip() or None
    job_name = det.job_name or str(record.get("JobName") or "").strip() or None

    if phase == "train":
        env = await load_cluster(cluster)
        meta = await details._read_slurm_meta(env.ssh_alias, job_id)

        def int_meta(key: str) -> int | None:
            try:
                raw = (meta.get(key) or "").strip()
                return int(raw) if raw else None
            except ValueError:
                return None

        return await submit.submit(
            submit.SubmitRequest(
                cluster=cluster,
                variant=variant,
                phase="train",
                partition=partition,
                train_num_gpus=int_meta("train_num_gpus"),
                train_global_batch_size=int_meta("train_global_batch_size"),
                train_max_steps=int_meta("train_max_steps"),
                train_save_steps=int_meta("train_save_steps"),
                job_name=job_name,
                resume=True,
                resume_of=job_id,
            )
        )

    checkpoint = det.paths.eval_checkpoint
    if not checkpoint:
        raise ValueError(f"cannot resume eval job {job_id}: checkpoint is unknown")

    seed_eval_dirs = [det.paths.eval_dir] if det.paths.eval_dir else []
    env = await load_cluster(cluster)
    meta = await details._read_slurm_meta(env.ssh_alias, job_id)
    eval_num_envs = (meta.get("eval_num_envs_per_gpu") or meta.get("eval_parallel_sims_per_gpu") or "").strip()
    try:
        eval_num_envs_per_gpu = int(eval_num_envs) if eval_num_envs else None
    except ValueError:
        eval_num_envs_per_gpu = None
    if eval_num_envs_per_gpu and eval_num_envs_per_gpu > submit.MAX_EVAL_NUM_ENVS_PER_GPU:
        eval_num_envs_per_gpu = submit.MAX_EVAL_NUM_ENVS_PER_GPU
    try:
        eval_n_episodes = int(meta.get("eval_n_episodes", "").strip()) if meta.get("eval_n_episodes") else None
    except ValueError:
        eval_n_episodes = None
    try:
        eval_n_runs = int(meta.get("eval_n_runs", "").strip()) if meta.get("eval_n_runs") else None
    except ValueError:
        eval_n_runs = None
    eval_sets = [s for s in (meta.get("eval_sets") or "").split() if s] or None
    resume_of = (meta.get("resume_of") or "").strip()
    if resume_of and resume_of != job_id:
        try:
            original = await details.get_details(cluster, resume_of)
            if original.paths.eval_dir and original.paths.eval_dir not in seed_eval_dirs:
                seed_eval_dirs.append(original.paths.eval_dir)
        except Exception:
            pass

    return await submit.submit(
        submit.SubmitRequest(
            cluster=cluster,
            variant=variant,
            phase="eval",
            partition=partition,
            eval_num_envs_per_gpu=eval_num_envs_per_gpu,
            eval_n_episodes=eval_n_episodes,
            eval_n_runs=eval_n_runs,
            eval_sets=eval_sets,
            checkpoint_path=checkpoint,
            seed_eval_results_from=seed_eval_dirs,
            job_name=job_name,
            resume_of=job_id,
        )
    )
