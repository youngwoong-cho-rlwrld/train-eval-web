"""Resume timed-out jobs from their original train/eval intent."""

from __future__ import annotations

import re
import shlex
from datetime import datetime

from . import details, jobs, submit
from .clusters import load_cluster
from .job_identity import phase_variant_from_meta
from .slurm_meta import read_slurm_meta, read_slurm_meta_many
from .ssh import ssh_run
from .variants import load_variant


def _original_phase(phase: str) -> str:
    # Historical jobs may have phase=resume in their sidecar/job_name. Treat
    # those as training jobs; new submissions never use resume as a phase.
    return "train" if phase == "resume" else phase


_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}_\d{6}$")


def _retry_job_name(original: str | None) -> str | None:
    """Re-stamp the original job name instead of regenerating it.

    Regenerating a default name drops any prefix the original submission
    carried; kakao's slurmctld rejects job names shorter than 50 chars, so
    keep the original name and only refresh its trailing timestamp.
    """
    name = (original or "").strip()
    if not name:
        return None
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    if _TIMESTAMP_SUFFIX.search(name):
        return _TIMESTAMP_SUFFIX.sub(f"_{stamp}", name)
    return f"{name}_{stamp}"


def _resume_job_name(original: str | None, parent_job_id: str) -> str:
    """Return the stable child name for one direct resume operation.

    The same parent always yields the same child name, which makes Slurm itself
    a durable reconciliation source if the backend loses the response or the
    sidecar write. Replace an earlier ``_r<job>`` suffix so resume chains do not
    grow the name indefinitely.
    """
    safe_parent = re.sub(r"[^A-Za-z0-9.-]+", "-", parent_job_id).strip(".-")
    if not safe_parent:
        raise ValueError(f"invalid parent job id {parent_job_id!r}")
    name = (original or "").strip()
    if not name:
        candidate = f"resubmission_unknown_job_r{safe_parent}_20000101_000000"
    else:
        match = _TIMESTAMP_SUFFIX.search(name)
        if not match:
            candidate = f"{name}_r{safe_parent}_20000101_000000"
        else:
            stem = re.sub(r"_r[A-Za-z0-9.-]+$", "", name[:match.start()])
            candidate = f"{stem}_r{safe_parent}{match.group(0)}"
    if len(candidate) < 50:
        candidate = f"openclaw_recovered_{candidate}"
    return candidate


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
    request = await build_resubmit_request(cluster, job_id, action=action)
    return await submit.submit(request)


async def build_resubmit_request(
    cluster: str,
    job_id: str,
    *,
    action: str,
) -> submit.SubmitRequest:
    """Reconstruct and validate one resubmission without submitting it."""
    if action not in {"resume", "retry"}:
        raise ValueError(f"unsupported job resubmit action: {action}")
    if cluster == "mlxp":
        raise ValueError(f"MLXP job {action} is not supported")

    record = await jobs.get_job(cluster, job_id)
    state = str(record.get("State") or "")
    if action == "resume" and not (
        jobs.is_timeout(state) or jobs.is_exhausted_requeue(record)
    ):
        raise ValueError(
            f"job {job_id} on {cluster} is not TIMEOUT or an exhausted "
            "automatic requeue"
        )
    if action == "retry" and not jobs.is_retryable_failure(state):
        raise ValueError(f"job {job_id} on {cluster} is {state or 'unknown'}, not FAILED")

    det = await details.get_details(cluster, job_id, include_progress=False)
    variant = det.variant
    if not variant:
        raise ValueError(f"cannot resume job {job_id}: variant is unknown")
    try:
        await load_variant(variant)
    except FileNotFoundError:
        raise ValueError(
            f"cannot {action} job {job_id}: experiment '{variant}' no longer exists "
            "locally (renamed or deleted since this job ran). Submit a new job for "
            "the current experiment from the Submit page instead."
        )

    phase = _original_phase(det.phase)
    if phase not in ("train", "eval"):
        raise ValueError(f"cannot resume job {job_id}: phase is {det.phase or 'unknown'}")

    partition = str(record.get("Partition") or "").strip() or None
    job_name = det.job_name or str(record.get("JobName") or "").strip() or None
    retrying = action == "retry"
    # Each direct parent gets one stable child display/log name. The immutable
    # output_namespace below makes eval resumes reuse and seed prior results;
    # the stable name makes a lost submission response safely reconcilable.
    resolved_job_name = (
        _resume_job_name(job_name, job_id)
        if action == "resume"
        else _retry_job_name(job_name)
    )
    idempotency_key = f"resume:{cluster}:{job_id}" if action == "resume" else None

    env = await load_cluster(cluster)
    meta = await read_slurm_meta(env.ssh_alias, job_id)

    def int_meta(key: str) -> int | None:
        try:
            raw = (meta.get(key) or "").strip()
            return int(raw) if raw else None
        except ValueError:
            return None

    if phase == "train":
        return submit.SubmitRequest(
            cluster=cluster,
            variant=variant,
            phase="train",
            train_note=(meta.get("train_note") or "").strip() or None,
            partition=partition,
            train_num_gpus=int_meta("train_num_gpus"),
            train_global_batch_size=int_meta("train_global_batch_size"),
            train_max_steps=int_meta("train_max_steps"),
            train_save_steps=int_meta("train_save_steps"),
            train_num_workers=int_meta("train_num_workers"),
            train_action_horizon=int_meta("train_action_horizon"),
            job_name=resolved_job_name,
            output_namespace=(meta.get("output_namespace") or "").strip() or None,
            resume=not retrying,
            resume_of=job_id,
            resubmit_action=action,
            idempotency_key=idempotency_key,
        )

    checkpoint = det.paths.eval_checkpoint
    if not checkpoint:
        raise ValueError(f"cannot resume eval job {job_id}: checkpoint is unknown")

    seed_eval_dirs = [det.paths.eval_dir] if det.paths.eval_dir else []
    eval_num_envs_per_gpu = submit.clamp_eval_num_envs(int_meta("eval_num_envs_per_gpu"))
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

    return submit.SubmitRequest(
        cluster=cluster,
        variant=variant,
        phase="eval",
        train_note=(meta.get("train_note") or "").strip() or None,
        partition=partition,
        # The sidecar records the eval allocation actually passed to sbatch;
        # thread it back as the eval override (it used to be smuggled in as
        # train_num_gpus, which skewed batch validation and SUBMIT_TRAIN_*).
        eval_num_gpus=int_meta("eval_num_gpus"),
        eval_num_envs_per_gpu=eval_num_envs_per_gpu,
        eval_n_episodes=eval_n_episodes,
        eval_n_runs=eval_n_runs,
        eval_sets=eval_sets,
        eval_tasks=[t for t in (meta.get("eval_tasks") or "").split() if t] or None,
        dexjoco_task=(meta.get("dexjoco_task") or "").strip() or None,
        checkpoint_path=checkpoint,
        seed_eval_results_from=seed_eval_dirs,
        job_name=resolved_job_name,
        output_namespace=(meta.get("output_namespace") or "").strip() or None,
        resume_of=job_id,
        resubmit_action=action,
        idempotency_key=idempotency_key,
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
        raise RuntimeError(
            f"failed to reconcile resumed jobs on {cluster}: "
            f"{(r.stderr or r.stdout or 'unknown error').strip()}"
        )

    child_ids = sorted({line.strip() for line in r.stdout.splitlines() if line.strip()}, reverse=True)
    if not child_ids:
        return []

    meta_by_job_id = await read_slurm_meta_many(env.ssh_alias, child_ids)
    linked: list[jobs.Job] = []
    for child_id in child_ids:
        meta = meta_by_job_id.get(child_id, {})
        phase, variant = phase_variant_from_meta(meta)
        meta_kwargs = {
            "cluster": cluster,
            "job_id": child_id,
            "phase": phase,
            "variant": variant or None,
            "resume_of": meta.get("resume_of") or None,
            "resubmit_action": meta.get("resubmit_action") or None,
        }
        try:
            record = await jobs.get_job(cluster, child_id)
            state = str(record.get("State") or "")
            linked.append(
                jobs.Job(
                    **meta_kwargs,
                    job_name=str(record.get("JobName") or meta.get("job_name") or child_id),
                    partition=str(record.get("Partition") or ""),
                    state=jobs.short_state(state),
                    elapsed=str(record.get("Elapsed") or ""),
                    nodelist=str(record.get("NodeList") or record.get("Reason") or ""),
                    start=str(record.get("Start") or "") or None,
                    end=str(record.get("End") or "") or None,
                )
            )
        except Exception:
            linked.append(
                jobs.Job(
                    **meta_kwargs,
                    job_name=meta.get("job_name") or child_id,
                    partition="",
                    state="UNKNOWN",
                    elapsed="",
                    nodelist="",
                )
            )
    return linked
