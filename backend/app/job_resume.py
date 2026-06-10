"""Resume timed-out jobs from their original train/eval intent."""

import shlex

from . import details, jobs, submit
from .clusters import load_cluster
from .slurm_meta import read_slurm_meta, read_slurm_meta_many
from .ssh import ssh_run


def _is_timeout(state: str) -> bool:
    return state.upper().startswith("TIMEOUT")


def _is_failure(state: str) -> bool:
    return state.upper().startswith((
        "FAIL",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPT",
    ))


def _original_phase(phase: str) -> str:
    # Historical jobs may have phase=resume in their sidecar/job_name. Treat
    # those as training jobs; new submissions never use resume as a phase.
    return "train" if phase == "resume" else phase


async def resume_timed_out_job(cluster: str, job_id: str) -> submit.SubmitResponse:
    return await _resubmit_slurm_job(cluster, job_id, action="resume")


async def retry_failed_job(cluster: str, job_id: str) -> submit.SubmitResponse:
    return await _resubmit_slurm_job(cluster, job_id, action="retry")


async def _resubmit_slurm_job(
    cluster: str,
    job_id: str,
    *,
    action: str,
) -> submit.SubmitResponse:
    if action not in {"resume", "retry"}:
        raise ValueError(f"unsupported job resubmit action: {action}")
    if cluster == "mlxp":
        raise ValueError(f"MLXP job {action} is not supported")

    record = await jobs.get_job(cluster, job_id)
    state = str(record.get("State") or "")
    if action == "resume" and not _is_timeout(state):
        raise ValueError(f"job {job_id} on {cluster} is {state or 'unknown'}, not TIMEOUT")
    if action == "retry" and not _is_failure(state):
        raise ValueError(f"job {job_id} on {cluster} is {state or 'unknown'}, not FAILED")

    det = await details.get_details(cluster, job_id, include_progress=False)
    variant = det.variant
    if not variant:
        raise ValueError(f"cannot resume job {job_id}: variant is unknown")

    phase = _original_phase(det.phase)
    if phase not in ("train", "eval"):
        raise ValueError(f"cannot resume job {job_id}: phase is {det.phase or 'unknown'}")

    partition = str(record.get("Partition") or "").strip() or None
    job_name = det.job_name or str(record.get("JobName") or "").strip() or None
    retrying = action == "retry"

    env = await load_cluster(cluster)
    meta = await read_slurm_meta(env.ssh_alias, job_id)

    def int_meta(key: str) -> int | None:
        try:
            raw = (meta.get(key) or "").strip()
            return int(raw) if raw else None
        except ValueError:
            return None

    if phase == "train":
        return await submit.submit(
            submit.SubmitRequest(
                cluster=cluster,
                variant=variant,
                phase="train",
                train_note=(meta.get("train_note") or "").strip() or None,
                partition=partition,
                train_num_gpus=int_meta("train_num_gpus"),
                train_global_batch_size=int_meta("train_global_batch_size"),
                train_max_steps=int_meta("train_max_steps"),
                train_save_steps=int_meta("train_save_steps"),
                train_action_horizon=int_meta("train_action_horizon"),
                job_name=None if retrying else job_name,
                output_namespace=(meta.get("output_namespace") or "").strip() or None,
                resume=not retrying,
                resume_of=job_id,
                resubmit_action=action,
            )
        )

    checkpoint = det.paths.eval_checkpoint
    if not checkpoint:
        raise ValueError(f"cannot resume eval job {job_id}: checkpoint is unknown")

    seed_eval_dirs = [det.paths.eval_dir] if det.paths.eval_dir else []
    eval_num_envs = (meta.get("eval_num_envs_per_gpu") or meta.get("eval_parallel_sims_per_gpu") or "").strip()
    try:
        eval_num_envs_per_gpu = int(eval_num_envs) if eval_num_envs else None
    except ValueError:
        eval_num_envs_per_gpu = None
    if eval_num_envs_per_gpu and eval_num_envs_per_gpu > submit.MAX_EVAL_NUM_ENVS_PER_GPU:
        eval_num_envs_per_gpu = submit.MAX_EVAL_NUM_ENVS_PER_GPU
    eval_n_episodes = int_meta("eval_n_episodes")
    eval_n_runs = int_meta("eval_n_runs")
    eval_sets = [s for s in (meta.get("eval_sets") or "").split() if s] or None
    resume_of = (meta.get("resume_of") or "").strip()
    if resume_of and resume_of != job_id:
        try:
            original = await details.get_details(cluster, resume_of, include_progress=False)
            if original.paths.eval_dir and original.paths.eval_dir not in seed_eval_dirs:
                seed_eval_dirs.append(original.paths.eval_dir)
        except Exception:
            pass

    return await submit.submit(
        submit.SubmitRequest(
            cluster=cluster,
            variant=variant,
            phase="eval",
            train_note=(meta.get("train_note") or "").strip() or None,
            partition=partition,
            train_num_gpus=int_meta("eval_num_gpus"),
            eval_num_envs_per_gpu=eval_num_envs_per_gpu,
            eval_n_episodes=eval_n_episodes,
            eval_n_runs=eval_n_runs,
            eval_sets=eval_sets,
            checkpoint_path=checkpoint,
            seed_eval_results_from=seed_eval_dirs,
            job_name=None if retrying else job_name,
            output_namespace=(meta.get("output_namespace") or "").strip() or None,
            resume_of=job_id,
            resubmit_action=action,
        )
    )


async def list_resumed_jobs(cluster: str, job_id: str) -> list[jobs.Job]:
    """Return direct jobs submitted by resuming `job_id`.

    Resume links are persisted in Slurm sidecar metadata as
    `resume_of=<job_id>`. Historical jobs without that sidecar cannot be linked.
    """
    if cluster == "mlxp":
        return []

    env = await load_cluster(cluster)
    cmd = (
        'for f in "$HOME/.train-eval-web/jobs"/*.meta; do '
        '[ -s "$f" ] || continue; '
        f"if grep -qx {shlex.quote(f'resume_of={job_id}')} \"$f\"; then "
        'b="${f##*/}"; printf "%s\\n" "${b%.meta}"; '
        "fi; "
        "done"
    )
    r = await ssh_run(env.ssh_alias, cmd, timeout=15.0)
    if r.returncode != 0:
        return []

    child_ids = sorted({line.strip() for line in r.stdout.splitlines() if line.strip()}, reverse=True)
    if not child_ids:
        return []

    meta_by_job_id = await read_slurm_meta_many(env.ssh_alias, child_ids)
    linked: list[jobs.Job] = []
    for child_id in child_ids:
        meta = meta_by_job_id.get(child_id, {})
        try:
            record = await jobs.get_job(cluster, child_id)
            state = str(record.get("State") or "")
            linked.append(
                jobs.Job(
                    cluster=cluster,
                    job_id=child_id,
                    job_name=str(record.get("JobName") or meta.get("job_name") or child_id),
                    partition=str(record.get("Partition") or ""),
                    state=state.split(" ")[0],
                    elapsed=str(record.get("Elapsed") or ""),
                    nodelist=str(record.get("NodeList") or record.get("Reason") or ""),
                    start=str(record.get("Start") or "") or None,
                    end=str(record.get("End") or "") or None,
                    phase=meta.get("phase") or None,
                    variant=meta.get("variant") or None,
                    resume_of=meta.get("resume_of") or None,
                    resubmit_action=meta.get("resubmit_action") or None,
                )
            )
        except Exception:
            linked.append(
                jobs.Job(
                    cluster=cluster,
                    job_id=child_id,
                    job_name=meta.get("job_name") or child_id,
                    partition="",
                    state="UNKNOWN",
                    elapsed="",
                    nodelist="",
                    phase=meta.get("phase") or None,
                    variant=meta.get("variant") or None,
                    resume_of=meta.get("resume_of") or None,
                    resubmit_action=meta.get("resubmit_action") or None,
                )
            )
    return linked
