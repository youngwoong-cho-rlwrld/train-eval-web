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
        return await submit.submit(
            submit.SubmitRequest(
                cluster=cluster,
                variant=variant,
                phase="train",
                partition=partition,
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
            checkpoint_path=checkpoint,
            seed_eval_results_from=seed_eval_dirs,
            job_name=job_name,
            resume_of=job_id,
        )
    )
