"""Per-job extended details: phase, paths, wandb url, progress.

Parses metadata out of the job_name (shape `{train|resume|eval}_{variant}_{YYYYMMDD}_{HHMMSS}`,
identical across slurm and mlxp), reads variant config locally, and asks
the cluster a few small questions over SSH (slurm) or kubectl (mlxp) to
compute progress.
"""

import json
import os
import re
import shlex
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field

from .clusters import load_cluster
from .data_interface import DataInterfaceSummary, summarize_data_interface_text
from .eval_completion import (
    eval_job_completed,
    eval_shape,
    exp_dir_rel_candidates,
)
from .job_identity import (
    parse_comment_metadata,
    parse_phase_and_variant,
    phase_variant_from_meta,
    resolve_phase_and_variant,
)
from .jobs import get_job
from .mlxp_config import get_settings
from .paths import CLUSTER_STAGING_REL
from .submission_snapshot import SubmitGitInfo, render_training_config_snapshot
from .ssh import ssh_run
from .slurm_meta import read_slurm_meta
from .training_models import resolve_training_model
from .variant_values import variant_int
from .variants import load_variant


from .wandb_config import get_project

# Wandb config:
#   - run id: WANDB_RUN_ID pinned by submit.py (slurm) / body script
#     (mlxp) to job_name. Already in hand here.
#   - entity: wandb.Api().default_entity after `wandb login` on this
#     laptop. Resolved lazily in _wandb_entity.
#   - project: captured at submit time in job metadata/config snapshot; if an
#     older job lacks that field, the current Settings value is the fallback.
WANDB_ENTITY_OVERRIDE = os.environ.get("TRAIN_EVAL_WEB_WANDB_ENTITY")
WANDB_WORKSPACE_OVERRIDE = os.environ.get("TRAIN_EVAL_WEB_WANDB_WORKSPACE")


class Paths(BaseModel):
    stdout: str
    stderr: str
    exp_dir: str
    ckpt_dir: str | None = None
    eval_checkpoint: str | None = None
    eval_dir: str | None = None
    isaac_logs_glob: str | None = None


class Progress(BaseModel):
    phase: str
    # train: current step / max steps
    current_step: int | None = None
    max_steps: int | None = None
    # eval: completed runs / total runs
    completed_runs: int | None = None
    total_runs: int | None = None
    current_label: str | None = None     # e.g. "0cm / run 2/3"
    percent: float | None = None         # 0..100


class GpuDeviceUsage(BaseModel):
    index: int
    name: str | None = None
    utilization_gpu_percent: int | None = None
    used_gb: float
    total_gb: float
    used_mib: int
    total_mib: int


class GpuUsage(BaseModel):
    node: str | None = None
    utilization_gpu_percent: float | None = None
    used_gb: float | None = None
    total_gb: float | None = None
    devices: list[GpuDeviceUsage] = Field(default_factory=list)
    error: str | None = None


class ConfigSnapshot(BaseModel):
    path: str | None = None
    meta_path: str | None = None
    text: str | None = None
    extra_args_path: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    wandb_project: str | None = None
    git_repo_path: str | None = None
    git_repo_label: str | None = None
    git_branch: str | None = None
    git_commit: str | None = None
    git_dirty_at_submit: bool | None = None
    git_committed_dirty: bool | None = None
    error: str | None = None


class EvalRun(BaseModel):
    task: str | None = None
    eval_set: str
    run: str
    seed: int | None = None
    success_count: int | None = None
    total_episodes: int | None = None
    success_rate: float | None = None
    path: str


class JobDetails(BaseModel):
    cluster: str
    job_id: str
    job_name: str
    phase: str            # "train" | "resume" | "eval" | "unknown"
    variant: str | None
    resume_of: str | None = None
    state: str
    elapsed: str
    wandb_project: str | None = None
    wandb_url: str | None
    paths: Paths
    progress: Progress
    gpu: GpuUsage | None = None
    config_snapshot: ConfigSnapshot | None = None
    data_interface: DataInterfaceSummary | None = None
    eval_runs: list[EvalRun] = Field(default_factory=list)


def _metadata_fields(text: str | None) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not text:
        return fields
    for chunk in text.split(";"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def _snapshot_wandb_project(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"^SUBMIT_WANDB_PROJECT=(.*)$", text, flags=re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw:
        return None
    try:
        parts = shlex.split(raw, comments=False, posix=True)
        if parts:
            return parts[0]
    except ValueError:
        pass
    return raw.strip("'\"") or None


async def _read_slurm_scontrol_comment(host: str, job_id: str) -> str | None:
    """Pull the Comment field out of `scontrol show job` (the live
    controller's view — sacct doesn't archive Comment on this cluster)."""
    r = await ssh_run(
        host,
        f"scontrol show job {job_id} 2>/dev/null | tr ' ' '\\n' | grep -m1 '^Comment='",
        timeout=10.0,
    )
    if r.returncode != 0:
        return None
    line = r.stdout.strip()
    if not line.startswith("Comment="):
        return None
    return line[len("Comment="):]


async def _resolve_slurm_log_paths(
    host: str,
    log_dir: str,
    job_name: str,
    job_id: str,
) -> tuple[str, str, str | None]:
    """Return stdout/stderr paths and the job-name portion from real logs.

    Historical jobs may not have sidecar metadata, and Slurm accounting can be
    less reliable than the actual log filenames. The logs all end in
    `_<job_id>.out|err`, regardless of the job-name convention.
    """
    default_stdout = f"{log_dir}/{job_name}_{job_id}.out"
    default_stderr = f"{log_dir}/{job_name}_{job_id}.err"
    r = await ssh_run(
        host,
        f"ls -1 {shlex.quote(log_dir)}/*_{shlex.quote(job_id)}.out 2>/dev/null | head -1",
        timeout=10.0,
    )
    stdout = r.stdout.strip().splitlines()[0] if r.stdout.strip() else default_stdout
    if not stdout.endswith(f"_{job_id}.out"):
        return default_stdout, default_stderr, None

    stderr = f"{stdout[:-4]}.err"
    leaf = stdout.rsplit("/", 1)[-1]
    log_job_name = leaf[: -len(f"_{job_id}.out")]
    return stdout, stderr, log_job_name or None


async def _resolve_runtime_exp_dir(
    host: str,
    stdout_path: str,
    phase: str,
) -> str | None:
    """Recover the experiment directory the job actually used from stdout."""
    if phase in ("train", "resume"):
        patterns = r"Output:[[:space:]]+|output_dir:[[:space:]]+|--output-dir[[:space:]]+"
    elif phase == "eval":
        patterns = (
            r"DONE[[:space:]]+|Saved to .*/results\.json|"
            r"Running eval -> |SKIP .*results\.json already exists"
        )
    else:
        return None

    r = await ssh_run(
        host,
        f"grep -E {shlex.quote(patterns)} {shlex.quote(stdout_path)} 2>/dev/null | tail -20",
        timeout=10.0,
    )
    for line in reversed(r.stdout.splitlines()):
        if phase in ("train", "resume"):
            m = re.search(r"Output:\s*(\S+)|output_dir:\s*(\S+)|--output-dir\s+(\S+)", line)
            if not m:
                continue
            ckpt_dir = next((g for g in m.groups() if g), "").strip("'\"")
            if "/checkpoints" in ckpt_dir:
                return ckpt_dir.split("/checkpoints", 1)[0]
        else:
            m = re.search(r"(?:DONE\s+|Saved to )(\S+)/results\.json", line)
            if m:
                result_dir = m.group(1)
                if "/eval_results/" in result_dir:
                    return result_dir.split("/eval_results/", 1)[0]
                return result_dir
            m = re.search(r"(/\S+?)/eval_results(?:/|\s|$)", line)
            if m:
                return m.group(1)
    return None


async def get_details(cluster: str, job_id: str, include_gpu: bool = False) -> JobDetails:
    sacct = await get_job(cluster, job_id)
    job_name = sacct.get("JobName", "")
    state = sacct.get("State", "")
    elapsed = sacct.get("Elapsed", "")

    phase, variant = resolve_phase_and_variant(job_name, sacct)

    if cluster == "mlxp":
        return await _mlxp_details(
            job_id,
            job_name,
            state,
            elapsed,
            phase,
            variant,
            include_gpu=include_gpu,
            pod_name=sacct.get("_pod_name") or None,
            node=sacct.get("NodeList") or None,
            job_comment=sacct.get("JobComment") or None,
        )

    env = await load_cluster(cluster)
    slurm_meta: dict[str, str] = await read_slurm_meta(env.ssh_alias, job_id)
    meta_phase, meta_variant = phase_variant_from_meta(slurm_meta)
    if meta_phase and meta_variant:
        phase, variant = meta_phase, meta_variant

    # Slurm: if sacct didn't return Comment (slurmdbd doesn't archive it on
    # this cluster), check scontrol (works for jobs still in the
    # controller) and then the on-disk .meta sidecar (permanent).
    if not variant:
        scontrol_comment = await _read_slurm_scontrol_comment(env.ssh_alias, job_id)
        if scontrol_comment:
            p, v = parse_comment_metadata(scontrol_comment)
            if p and v:
                phase, variant = p, v
    log_dir = env.vars["LOG_DIR"]
    stdout_path, stderr_path, log_job_name = await _resolve_slurm_log_paths(
        env.ssh_alias,
        log_dir,
        job_name,
        job_id,
    )
    if not variant and log_job_name:
        p, v = parse_phase_and_variant(log_job_name)
        if p != "unknown" and v:
            phase, variant = p, v

    # The per-variant experiment dir on the cluster depends on who submitted:
    # web-submitted jobs use ~/.train-eval-web/experiments/<variant>; jobs
    # launched via the bash `./submit` use ~/train-eval-scripts/experiments/<variant>.
    # Probe both, prefer one that actually exists.
    exp_dir_remote = await _resolve_exp_dir(env.ssh_alias, job_id, variant) if variant else f"$HOME/{CLUSTER_STAGING_REL}/experiments"
    runtime_exp_dir = await _resolve_runtime_exp_dir(env.ssh_alias, stdout_path, phase)
    if runtime_exp_dir:
        exp_dir_remote = runtime_exp_dir
    ckpt_dir = (
        slurm_meta.get("checkpoint_dir")
        or (f"{exp_dir_remote}/checkpoints" if phase in ("train", "resume") else None)
    )
    eval_dir = (
        slurm_meta.get("eval_dir")
        or (f"{exp_dir_remote}/eval_results" if phase == "eval" else None)
    )
    eval_checkpoint = (
        await _resolve_eval_checkpoint(
            env.ssh_alias,
            stdout_path,
            exp_dir_remote,
            slurm_meta.get("checkpoint_path"),
        )
        if phase == "eval" and variant
        else None
    )
    isaac_logs_glob = (
        f"{slurm_meta.get('job_log_dir')}/server_*.log"
        if phase == "eval" and slurm_meta.get("job_log_dir")
        else (f"{exp_dir_remote}/logs/{job_id}/server_*.log" if phase == "eval" else None)
    )
    paths = Paths(
        stdout=stdout_path,
        stderr=stderr_path,
        exp_dir=exp_dir_remote,
        ckpt_dir=ckpt_dir,
        eval_checkpoint=eval_checkpoint,
        eval_dir=eval_dir,
        isaac_logs_glob=isaac_logs_glob,
    )
    config_snapshot = await _slurm_config_snapshot(env.ssh_alias, slurm_meta, cluster=cluster)
    data_interface = await _slurm_data_interface_snapshot(
        env.ssh_alias,
        slurm_meta,
        config_snapshot,
        variant=variant,
    )
    job_wandb_project = (
        slurm_meta.get("wandb_project")
        or slurm_meta.get("submit_wandb_project")
        or (config_snapshot.wandb_project if config_snapshot else None)
    )

    wandb_url: str | None = None
    if phase in ("train", "resume"):
        # submit.py pins WANDB_RUN_ID = job_name via sbatch --export.
        wandb_url = await _wandb_url(job_name, project=job_wandb_project)

    progress = await _compute_progress(
        cluster,
        job_id,
        phase,
        variant,
        stdout_path,
        stderr_path,
        ckpt_dir,
        eval_dir,
        slurm_meta,
        wandb_project=job_wandb_project,
    )
    if phase == "eval" and variant and eval_dir:
        try:
            if await eval_job_completed(env.ssh_alias, stdout_path, eval_dir, variant, slurm_meta):
                state = "COMPLETED"
        except Exception:
            pass
    gpu = (
        await _slurm_gpu_usage(
            env.ssh_alias,
            job_id,
            sacct.get("NodeList") or "",
            state,
        )
        if include_gpu
        else None
    )
    eval_runs = await _slurm_eval_runs(env.ssh_alias, eval_dir) if phase == "eval" and eval_dir else []

    return JobDetails(
        cluster=cluster, job_id=job_id, job_name=job_name,
        phase=phase, variant=variant, resume_of=slurm_meta.get("resume_of") or None,
        state=state, elapsed=elapsed,
        wandb_project=job_wandb_project,
        wandb_url=wandb_url, paths=paths, progress=progress, gpu=gpu,
        config_snapshot=config_snapshot,
        data_interface=data_interface,
        eval_runs=eval_runs,
    )


async def _mlxp_details(
    job_id: str,
    job_name: str,
    state: str,
    elapsed: str,
    phase: str,
    variant: str | None,
    include_gpu: bool = False,
    pod_name: str | None = None,
    node: str | None = None,
    job_comment: str | None = None,
) -> JobDetails:
    """MLXP runs train via `kubectl apply` on a pod. All paths live on DDN."""
    settings = get_settings()
    exp_dir = f"{settings.experiments_dir}/{variant}" if variant else settings.experiments_dir
    metadata = _metadata_fields(job_comment)
    output_namespace = metadata.get("output_namespace") or None
    ckpt_dir = (
        metadata.get("checkpoint_dir")
        or (
            f"{exp_dir}/checkpoints/{output_namespace or job_name}"
            if phase in ("train", "resume") else None
        )
    )
    eval_dir = (
        metadata.get("eval_dir")
        or (
            f"{exp_dir}/eval_results/{output_namespace}"
            if phase == "eval" and output_namespace else
            (f"{exp_dir}/eval_results" if phase == "eval" else None)
        )
    )
    paths = Paths(
        stdout=f"kubectl logs -n {settings.namespace} -l job-name={job_id}",
        stderr=f"kubectl logs -n {settings.namespace} -l job-name={job_id}  (k8s merges stdout+stderr)",
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        eval_checkpoint=None,
        eval_dir=eval_dir,
        isaac_logs_glob=(
            f"{metadata.get('job_log_dir')}/server_*.log"
            if phase == "eval" and metadata.get("job_log_dir")
            else (f"{exp_dir}/logs/{job_id}/server_*.log" if phase == "eval" else None)
        ),
    )
    if phase == "eval":
        paths.eval_checkpoint = metadata.get("checkpoint_path") or None
    config_snapshot = await _mlxp_config_snapshot(metadata)
    data_interface = await _mlxp_data_interface_snapshot(
        metadata,
        config_snapshot,
        variant=variant,
    )
    job_wandb_project = (
        metadata.get("wandb_project")
        or metadata.get("submit_wandb_project")
        or (config_snapshot.wandb_project if config_snapshot else None)
    )
    # mlxp body pins WANDB_RUN_ID = job_name (resolved from the Job's
    # display-name annotation by get_job). The k8s job_id has no wandb
    # run behind it.
    wandb_url = (
        await _wandb_url(job_name, project=job_wandb_project)
        if phase in ("train", "resume")
        else None
    )
    progress = await _mlxp_progress(job_name, variant, phase, metadata, project=job_wandb_project)
    gpu = await _mlxp_gpu_usage(pod_name, node, state) if include_gpu else None
    eval_runs = await _mlxp_eval_runs(paths.eval_dir) if phase == "eval" and paths.eval_dir else []

    return JobDetails(
        cluster="mlxp", job_id=job_id, job_name=job_name,
        phase=phase, variant=variant, resume_of=metadata.get("resume_of") or None,
        state=state, elapsed=elapsed,
        wandb_project=job_wandb_project,
        wandb_url=wandb_url, paths=paths, progress=progress, gpu=gpu,
        config_snapshot=config_snapshot,
        data_interface=data_interface,
        eval_runs=eval_runs,
    )


def _meta_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "y"}


async def _slurm_config_snapshot(
    host: str,
    meta: dict[str, str],
    *,
    cluster: str | None = None,
) -> ConfigSnapshot | None:
    path = meta.get("config_snapshot_path")
    meta_path = meta.get("config_snapshot_meta_path")
    rel = meta.get("config_snapshot_rel")
    extra_args_path = meta.get("submit_extra_args_path")
    extra_args_rel = meta.get("submit_extra_args_rel")
    if not path and rel:
        path = f"$HOME/{rel}"
    if not path and not extra_args_path and not extra_args_rel:
        return None

    text: str | None = None
    error: str | None = None
    if path:
        if rel:
            cmd = f"cat $HOME/{shlex.quote(rel)} 2>/dev/null"
        else:
            cat_path = path
            if cat_path.startswith("$HOME/"):
                cmd = f"cat $HOME/{shlex.quote(cat_path[len('$HOME/'):])} 2>/dev/null"
            else:
                cmd = f"cat {shlex.quote(cat_path)} 2>/dev/null"
        r = await ssh_run(host, cmd, timeout=10.0)
        if r.returncode == 0:
            text = r.stdout
        else:
            error = (r.stderr or "config snapshot not found").strip()

    extra_args_text = await _slurm_optional_text(host, extra_args_path, extra_args_rel)
    sidecar_extra_args = _snapshot_extra_args(extra_args_text)
    if text is None and meta.get("phase") == "train":
        recovered = await _recover_slurm_training_snapshot(meta, sidecar_extra_args, cluster=cluster)
        if recovered:
            text = recovered
            error = None
    extra_args = _snapshot_extra_args(text) or sidecar_extra_args
    if extra_args_path is None and extra_args_rel:
        extra_args_path = f"$HOME/{extra_args_rel}"

    return ConfigSnapshot(
        path=path,
        meta_path=meta_path,
        text=text,
        extra_args_path=extra_args_path,
        extra_args=extra_args,
        wandb_project=meta.get("wandb_project")
        or meta.get("submit_wandb_project")
        or _snapshot_wandb_project(text),
        git_repo_path=meta.get("submit_git_repo_path") or None,
        git_repo_label=meta.get("submit_git_repo_label") or None,
        git_branch=meta.get("submit_git_branch") or None,
        git_commit=meta.get("submit_git_commit") or None,
        git_dirty_at_submit=_meta_bool(meta.get("submit_git_dirty_at_submit")),
        git_committed_dirty=_meta_bool(meta.get("submit_git_committed_dirty")),
        error=error,
    )


async def _recover_slurm_training_snapshot(
    meta: dict[str, str],
    extra_args: list[str],
    *,
    cluster: str | None = None,
) -> str | None:
    variant_name = meta.get("variant")
    job_name = meta.get("job_name")
    if not variant_name or not job_name:
        return None
    try:
        variant = await load_variant(variant_name)
        model = resolve_training_model(variant)
        train_num_gpus = _meta_int(meta, "train_num_gpus") or variant_int(variant, "TRAIN_NUM_GPUS", 2)
        train_max_steps = _meta_int(meta, "train_max_steps") or variant_int(variant, "MAX_STEPS", 30000)
        train_save_steps = _meta_int(meta, "train_save_steps") or variant_int(variant, "SAVE_STEPS", 1000)
        train_action_horizon = _meta_int(meta, "train_action_horizon")
        train_global_batch_size = _meta_int(meta, "train_global_batch_size")
        if train_global_batch_size is None:
            for key in ("TRAIN_GLOBAL_BATCH_SIZE", "GLOBAL_BATCH_SIZE"):
                raw = (variant.vars.get(key) or "").strip()
                if raw:
                    try:
                        train_global_batch_size = int(raw)
                        break
                    except ValueError:
                        pass
        git = _snapshot_git_from_meta(meta)
        text = render_training_config_snapshot(
            base_config=variant.raw,
            variant=variant_name,
            model=model.family,
            job_name=job_name,
            cluster=cluster or meta.get("cluster") or "",
            partition=meta.get("partition") or meta.get("submit_partition"),
            node=meta.get("node") or meta.get("submit_node"),
            extra_args=extra_args,
            train_num_gpus=train_num_gpus,
            train_global_batch_size=train_global_batch_size,
            train_max_steps=train_max_steps,
            train_save_steps=train_save_steps,
            train_action_horizon=train_action_horizon,
            wandb_project=meta.get("wandb_project") or meta.get("submit_wandb_project"),
            git=git,
        )
    except Exception:
        return None
    warning = (
        "# Recovered by train-eval-web because the original submission snapshot file is missing.\n"
        "# Source: Slurm sidecar metadata plus the current local base variant config.\n"
        "# If the original extra-args sidecar was deleted too, per-job extra args may be absent below.\n\n"
    )
    return warning + text


def _snapshot_git_from_meta(meta: dict[str, str]) -> SubmitGitInfo | None:
    repo_path = meta.get("submit_git_repo_path") or meta.get("submit_train_repo_dir")
    repo_label = meta.get("submit_git_repo_label") or meta.get("model_label")
    commit = meta.get("submit_git_commit")
    if not (repo_path or repo_label or commit):
        return None
    return SubmitGitInfo(
        repo_path=repo_path or "",
        repo_label=repo_label or "",
        commit=commit or None,
        commit_subject=meta.get("submit_git_commit_subject") or None,
        branch=meta.get("submit_git_branch") or None,
        dirty_before=bool(_meta_bool(meta.get("submit_git_dirty_at_submit"))),
        committed_dirty=bool(_meta_bool(meta.get("submit_git_committed_dirty"))),
        dirty_files=[],
    )


async def _slurm_data_interface_snapshot(
    host: str,
    meta: dict[str, str],
    config_snapshot: ConfigSnapshot | None,
    *,
    variant: str | None,
) -> DataInterfaceSummary | None:
    if not variant:
        return None
    source, path = _resolve_remote_modality_path(meta, config_snapshot)
    if not source and not path:
        return None
    if not path:
        return DataInterfaceSummary(
            variant=variant,
            source=source,
            error="TRAIN_MODALITY_CONFIG is set, but the submitted config snapshot path is unavailable",
        )
    text = await _slurm_optional_text(host, path, None)
    if text is None:
        return DataInterfaceSummary(
            variant=variant,
            source=source,
            path=path,
            error=f"modality.py not found on cluster: {path}",
        )
    return summarize_data_interface_text(
        variant_name=variant,
        source=source,
        path=path,
        text=text,
    )


def _resolve_remote_modality_path(
    meta: dict[str, str],
    config_snapshot: ConfigSnapshot | None,
) -> tuple[str | None, str | None]:
    path = (meta.get("train_modality_config") or "").strip() or None
    if path:
        return path.rsplit("/", 1)[-1], path

    source = _snapshot_shell_value(config_snapshot.text if config_snapshot else None, "TRAIN_MODALITY_CONFIG")
    if not source:
        return None, None
    if source.startswith("$HOME/") or source.startswith("/") or source.startswith("~/"):
        return source, source

    config_path = config_snapshot.path if config_snapshot else None
    if not config_path or "/" not in config_path:
        return source, None
    return source, f"{config_path.rsplit('/', 1)[0]}/{source}"


def _snapshot_shell_value(text: str | None, key: str) -> str | None:
    if not text:
        return None
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, flags=re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw:
        return None
    try:
        parts = shlex.split(raw, comments=False, posix=True)
        if parts:
            return parts[0]
    except ValueError:
        pass
    return raw.strip("'\"") or None


def _meta_int(meta: dict[str, str], key: str) -> int | None:
    raw = meta.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def _slurm_optional_text(host: str, path: str | None, rel: str | None = None) -> str | None:
    if not path and not rel:
        return None
    if rel:
        cmd = f"cat $HOME/{shlex.quote(rel)} 2>/dev/null"
    elif path and path.startswith("$HOME/"):
        cmd = f"cat $HOME/{shlex.quote(path[len('$HOME/'):])} 2>/dev/null"
    elif path and path.startswith("~/"):
        cmd = f"cat $HOME/{shlex.quote(path[len('~/'):])} 2>/dev/null"
    elif path:
        cmd = f"cat {shlex.quote(path)} 2>/dev/null"
    else:
        return None
    r = await ssh_run(host, cmd, timeout=10.0)
    return r.stdout if r.returncode == 0 else None


def _snapshot_extra_args(text: str | None) -> list[str]:
    if not text:
        return []
    match = re.search(r"^SUBMIT_EXTRA_ARGS=\(\n([\s\S]*?)^\)$", text, flags=re.MULTILINE)
    if not match:
        return []
    args: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line, comments=False, posix=True)
        except ValueError:
            parts = [line.strip("'\"")]
        args.extend(parts)
    return args


async def _mlxp_config_snapshot(meta: dict[str, str]) -> ConfigSnapshot | None:
    path = meta.get("config_snapshot_path")
    if not path:
        return None
    meta_path = meta.get("config_snapshot_meta_path")

    text, error = await _mlxp_read_file(path)

    return ConfigSnapshot(
        path=path,
        meta_path=meta_path,
        text=text,
        wandb_project=meta.get("wandb_project")
        or meta.get("submit_wandb_project")
        or _snapshot_wandb_project(text),
        git_repo_path=meta.get("submit_git_repo_path") or None,
        git_repo_label=meta.get("submit_git_repo_label") or None,
        git_branch=meta.get("submit_git_branch") or None,
        git_commit=meta.get("submit_git_commit") or None,
        git_dirty_at_submit=_meta_bool(meta.get("submit_git_dirty_at_submit")),
        git_committed_dirty=_meta_bool(meta.get("submit_git_committed_dirty")),
        error=error,
    )


async def _mlxp_data_interface_snapshot(
    meta: dict[str, str],
    config_snapshot: ConfigSnapshot | None,
    *,
    variant: str | None,
) -> DataInterfaceSummary | None:
    if not variant:
        return None
    source, path = _resolve_remote_modality_path(meta, config_snapshot)
    if not source and not path:
        return None
    if not path:
        return DataInterfaceSummary(
            variant=variant,
            source=source,
            error="TRAIN_MODALITY_CONFIG is set, but the submitted config snapshot path is unavailable",
        )
    text, _ = await _mlxp_read_file(path)
    if text is None:
        return DataInterfaceSummary(
            variant=variant,
            source=source,
            path=path,
            error=f"modality.py not found on MLXP: {path}",
        )
    return summarize_data_interface_text(
        variant_name=variant,
        source=source,
        path=path,
        text=text,
    )


async def _mlxp_read_file(path: str) -> tuple[str | None, str | None]:
    import asyncio
    import shutil
    from .mlxp_data_pod import ensure_listing_pod

    if shutil.which("kubectl") is None:
        return None, "kubectl not found on PATH"
    settings = get_settings()
    try:
        pod = await ensure_listing_pod()
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            "exec",
            "-n",
            settings.namespace,
            pod,
            "--",
            "cat",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=12.0)
        if proc.returncode == 0:
            return stdout.decode(errors="replace"), None
        return None, stderr.decode(errors="replace").strip() or "file not found"
    except Exception as exc:
        return None, str(exc)


_EVAL_RUNS_SCRIPT = r'''
import json
import os
import re
from pathlib import Path


def int_or_none(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def seed_from_run_label(label):
    m = re.search(r"seed([0-9]+)", label)
    return int(m.group(1)) if m else None


root = Path(os.environ["EVAL_DIR"])
rows = []
if root.is_dir():
    for path in sorted(root.glob("**/results.json"), key=lambda p: str(p)):
        rel = path.relative_to(root).parts
        if len(rel) < 3:
            continue
        run_label = rel[-2]
        if not run_label.startswith("run_"):
            continue
        eval_set = rel[-3]
        task = "/".join(rel[:-3]) or None
        try:
            data = json.load(open(path))
        except Exception:
            continue
        config = data.get("config") or {}
        summary = data.get("summary") or {}
        seed = int_or_none(config.get("seed"))
        if seed is None:
            seed = seed_from_run_label(run_label)
        rows.append({
            "task": task,
            "eval_set": eval_set,
            "run": run_label,
            "seed": seed,
            "success_count": int_or_none(summary.get("success_count")),
            "total_episodes": int_or_none(summary.get("total_episodes") or summary.get("episode_count")),
            "success_rate": float_or_none(summary.get("success_rate")),
            "path": str(path),
        })
print(json.dumps(rows))
'''


async def _slurm_eval_runs(host: str, eval_dir: str) -> list[EvalRun]:
    cmd = (
        f"EVAL_DIR={_remote_path_expr(eval_dir)} "
        "python3 - <<'PY'\n"
        + _EVAL_RUNS_SCRIPT
        + "\nPY"
    )
    try:
        r = await ssh_run(host, cmd, timeout=20.0)
    except Exception:
        return []
    if r.returncode != 0:
        return []
    try:
        raw = json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return [EvalRun.model_validate(item) for item in raw]


def _remote_path_expr(path: str) -> str:
    return path if path.startswith("$HOME/") else shlex.quote(path)


async def _mlxp_eval_runs(eval_dir: str) -> list[EvalRun]:
    import asyncio
    import shutil
    from .mlxp_data_pod import ensure_listing_pod

    if shutil.which("kubectl") is None:
        return []
    settings = get_settings()
    try:
        pod = await ensure_listing_pod()
    except Exception:
        return []
    cmd = (
        f"EVAL_DIR={shlex.quote(eval_dir)} "
        "python3 - <<'PY'\n"
        + _EVAL_RUNS_SCRIPT
        + "\nPY"
    )
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-lc", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
    except Exception:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return []
    if proc.returncode != 0:
        return []
    try:
        raw = json.loads(stdout.decode(errors="replace") or "[]")
    except json.JSONDecodeError:
        return []
    return [EvalRun.model_validate(item) for item in raw]


def _parse_gpu_usage(stdout: str, node: str | None) -> GpuUsage | None:
    samples: dict[int, dict[str, Any]] = {}
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            if len(parts) >= 5:
                name = parts[1] or None
                utilization = int(parts[2])
                used_mib = int(parts[3])
                total_mib = int(parts[4])
            elif len(parts) >= 4:
                name = None
                utilization = int(parts[1])
                used_mib = int(parts[2])
                total_mib = int(parts[3])
            else:
                name = None
                utilization = None
                used_mib = int(parts[1])
                total_mib = int(parts[2])
        except ValueError:
            continue
        sample = samples.setdefault(
            index,
            {
                "name": name,
                "utilization_values": [],
                "used_mib": used_mib,
                "total_mib": total_mib,
            },
        )
        if name:
            sample["name"] = name
        if utilization is not None:
            sample["utilization_values"].append(utilization)
        sample["used_mib"] = max(int(sample["used_mib"]), used_mib)
        sample["total_mib"] = total_mib
    devices: list[GpuDeviceUsage] = []
    for index in sorted(samples):
        sample = samples[index]
        values = sample["utilization_values"]
        utilization = round(sum(values) / len(values)) if values else None
        used_mib = int(sample["used_mib"])
        total_mib = int(sample["total_mib"])
        devices.append(
            GpuDeviceUsage(
                index=index,
                name=sample["name"],
                utilization_gpu_percent=utilization,
                used_mib=used_mib,
                total_mib=total_mib,
                used_gb=round(used_mib / 1024, 1),
                total_gb=round(total_mib / 1024, 1),
            )
        )
    if not devices:
        return None
    used_mib_total = sum(d.used_mib for d in devices)
    total_mib_total = sum(d.total_mib for d in devices)
    utilization_values = [
        d.utilization_gpu_percent
        for d in devices
        if d.utilization_gpu_percent is not None
    ]
    return GpuUsage(
        node=node,
        utilization_gpu_percent=(
            round(sum(utilization_values) / len(utilization_values), 1)
            if utilization_values else None
        ),
        used_gb=round(used_mib_total / 1024, 1),
        total_gb=round(total_mib_total / 1024, 1),
        devices=devices,
    )


async def _first_slurm_node(host: str, nodelist: str) -> str | None:
    node_expr = nodelist.strip()
    if not node_expr or node_expr in {"None assigned", "N/A", "(None)"}:
        return None
    r = await ssh_run(
        host,
        f"PATH=/opt/slurm/bin:$PATH scontrol show hostnames {shlex.quote(node_expr)} 2>/dev/null | head -1",
        timeout=8.0,
    )
    node = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    return node or node_expr


async def _slurm_gpu_usage(
    host: str,
    job_id: str,
    nodelist: str,
    state: str,
) -> GpuUsage:
    node_expr = nodelist.strip()
    fallback_node = None if node_expr in {"", "None assigned", "N/A", "(None)"} else node_expr
    if not state.upper().startswith(("RUNNING", "COMPLETING")):
        return GpuUsage(node=fallback_node, error="GPU memory is only available while the job is running")

    node = await _first_slurm_node(host, nodelist)

    smi_query = (
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )
    query = f"for i in 1 2 3; do {smi_query}; [ \"$i\" = 3 ] || sleep 1; done"

    # Preferred: execute inside the job allocation. This works even when
    # compute-node SSH is not available directly from the login node.
    srun_cmd = (
        f"PATH=/opt/slurm/bin:$PATH timeout 8 "
        f"srun --jobid={shlex.quote(job_id)} --overlap --ntasks=1 "
        f"bash -lc {shlex.quote(query)} 2>&1"
    )
    r = await ssh_run(host, srun_cmd, timeout=10.0)
    usage = _parse_gpu_usage(r.stdout, node)
    if usage:
        return usage

    # Fallback for clusters where login -> compute SSH is permitted.
    if node:
        ssh_cmd = (
            "timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "
            "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            f"{shlex.quote(node)} {shlex.quote(query)} 2>&1"
        )
        r = await ssh_run(host, ssh_cmd, timeout=10.0)
        usage = _parse_gpu_usage(r.stdout, node)
        if usage:
            return usage

    err = (r.stderr or r.stdout or "").strip().splitlines()
    message = err[-1] if err else "GPU memory is unavailable"
    return GpuUsage(node=node, error=message[:240])


async def _mlxp_gpu_usage(
    pod_name: str | None,
    node: str | None,
    state: str,
) -> GpuUsage:
    import asyncio
    import shutil

    fallback_node = node or pod_name or None
    if not state.upper().startswith(("RUNNING", "COMPLETING")):
        return GpuUsage(node=fallback_node, error="GPU memory is only available while the job is running")
    if not pod_name:
        return GpuUsage(node=fallback_node, error="MLXP pod is not available yet")
    if shutil.which("kubectl") is None:
        return GpuUsage(node=fallback_node, error="kubectl not found on PATH")
    settings = get_settings()

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            "exec",
            "-n",
            settings.namespace,
            pod_name,
            "--",
            "bash",
            "-lc",
            (
                "for i in 1 2 3; do "
                "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
                "--format=csv,noheader,nounits; "
                '[ "$i" = 3 ] || sleep 1; '
                "done"
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=12.0)
    except Exception as exc:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return GpuUsage(node=fallback_node, error=f"MLXP GPU sample failed: {exc}")

    if proc.returncode != 0:
        msg = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
        lines = msg.splitlines()
        return GpuUsage(
            node=fallback_node,
            error=(lines[-1] if lines else "MLXP GPU sample failed")[:240],
        )

    usage = _parse_gpu_usage(stdout.decode(errors="replace"), fallback_node)
    return usage or GpuUsage(node=fallback_node, error="MLXP GPU sample returned no devices")


async def _mlxp_progress(
    run_id: str,
    variant: str | None,
    phase: str,
    metadata: dict[str, str] | None = None,
    project: str | None = None,
) -> Progress:
    """Progress for an MLXP training job.

    Primary source: the run's wandb summary (its `_step` field is updated
    every logging tick — i.e. every 10 training steps for gr00t-n16).
    Fallback: highest `checkpoint-N` dir on DDN (SAVE_STEPS granularity).

    `run_id` is the wandb run id — the job_name (display name) for MLXP,
    not the k8s job_id which has no wandb run behind it.
    """
    import asyncio

    progress = Progress(phase=phase)
    if not variant:
        return progress

    metadata = metadata or {}
    if phase == "eval":
        return await _mlxp_eval_progress(variant, metadata, progress)

    try:
        if metadata.get("train_max_steps"):
            progress.max_steps = int(metadata["train_max_steps"])
        else:
            v = await load_variant(variant)
            if "MAX_STEPS" in v.vars:
                progress.max_steps = int(v.vars["MAX_STEPS"])
    except Exception:
        pass

    if phase not in ("train", "resume"):
        return progress

    # 1. wandb — fine-grained (per logging tick).
    step = await _wandb_step(run_id, project=project)
    if step is not None:
        progress.current_step = step
        if progress.max_steps:
            progress.percent = round(100.0 * step / progress.max_steps, 1)
            progress.current_label = f"step {step:,}/{progress.max_steps:,}"
        else:
            progress.current_label = f"step {step:,}"
        return progress

    # 2. checkpoint dir count — coarse (SAVE_STEPS granularity).
    import shutil
    if shutil.which("kubectl") is None:
        return progress
    settings = get_settings()
    ckpt_root = f"{settings.experiments_dir}/{variant}/checkpoints"
    checkpoint_dir = metadata.get("checkpoint_dir")
    output_namespace = metadata.get("output_namespace")
    ckpt_dirs = [
        d for d in [
            checkpoint_dir,
            f"{ckpt_root}/{output_namespace}" if output_namespace else None,
            f"{ckpt_root}/{run_id}",
            f"{ckpt_root}/{run_id}/{run_id}",
            ckpt_root,
        ]
        if d
    ]
    from .mlxp_data_pod import ensure_listing_pod
    try:
        pod = await ensure_listing_pod()
    except Exception:
        return progress

    dirs = " ".join(shlex.quote(d) for d in ckpt_dirs)
    cmd = (
        f"for d in {dirs}; do "
        'ls -d "$d"/checkpoint-* 2>/dev/null; '
        "done | sed 's:.*checkpoint-::' | sort -n | tail -1"
    )
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-c", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except Exception:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return progress

    latest = stdout.decode(errors="replace").strip()
    if not latest.isdigit():
        return progress
    cur = int(latest)
    progress.current_step = cur
    if progress.max_steps:
        progress.percent = round(100.0 * cur / progress.max_steps, 1)
        progress.current_label = f"step {cur:,}/{progress.max_steps:,}"
    else:
        progress.current_label = f"step {cur:,}"
    return progress


async def _mlxp_eval_progress(
    variant: str,
    metadata: dict[str, str],
    progress: Progress,
) -> Progress:
    import asyncio
    import shutil

    try:
        v = await load_variant(variant)
        eval_sets, n_runs, n_eps, tasks = eval_shape(v, metadata)
    except Exception:
        return progress
    total = max(len(tasks) * len(eval_sets) * n_runs, 0)
    progress.total_runs = total or None
    if total <= 0:
        return progress
    if n_eps > 0:
        progress.max_steps = total * n_eps

    if shutil.which("kubectl") is None:
        return progress
    settings = get_settings()
    eval_dir = metadata.get("eval_dir") or f"{settings.experiments_dir}/{variant}/eval_results"
    from .mlxp_data_pod import ensure_listing_pod
    try:
        pod = await ensure_listing_pod()
    except Exception:
        return progress

    cmd = f"find {shlex.quote(eval_dir)} -type f -name results.json 2>/dev/null | wc -l"
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-c", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except Exception:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return progress
    try:
        completed = int(stdout.decode(errors="replace").strip())
    except ValueError:
        completed = 0
    progress.completed_runs = completed
    if n_eps > 0 and progress.max_steps:
        progress.current_step = min(completed * n_eps, progress.max_steps)
        progress.percent = round(100.0 * progress.current_step / progress.max_steps, 1)
        progress.current_label = f"{completed}/{total} runs · {progress.current_step}/{progress.max_steps} episodes"
    else:
        progress.percent = round(100.0 * completed / total, 1)
        progress.current_label = f"{completed}/{total} runs"
    return progress


def _append_wandb_workspace(url: str, workspace: str | None) -> str:
    if not workspace:
        return url
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("nw", workspace)
    return urlunparse(parsed._replace(query=urlencode(query)))


async def _wandb_workspace(entity: str) -> str | None:
    """Return W&B's browser workspace selector for the entity, if known."""
    if WANDB_WORKSPACE_OVERRIDE:
        return WANDB_WORKSPACE_OVERRIDE
    if entity in _wandb_workspace_cache:
        return _wandb_workspace_cache[entity] or None

    import asyncio

    def _query() -> str | None:
        try:
            import wandb
            api = wandb.Api(timeout=10)
            viewer = api.viewer
            username = getattr(viewer, "username", None)
            if username == entity:
                return f"nwuser{entity}"
            teams = getattr(viewer, "teams", None)
            if callable(teams):
                teams = teams()
            if isinstance(teams, dict):
                teams = [edge.get("node") for edge in teams.get("edges", [])]
            teams = teams or []
            for team in teams:
                name = team.get("name") if isinstance(team, dict) else getattr(team, "name", None)
                if name == entity:
                    return f"nwteam{entity}"
        except Exception:
            return None
        return None

    workspace = await asyncio.to_thread(_query)
    _wandb_workspace_cache[entity] = workspace or ""
    return workspace


_wandb_workspace_cache: dict[str, str] = {}


async def _wandb_url(run_id: str, project: str | None = None) -> str | None:
    import asyncio

    entity = await _wandb_entity()
    if not entity:
        return None
    resolved_project = project or get_project()
    workspace = await _wandb_workspace(entity)
    fallback = _append_wandb_workspace(
        f"https://wandb.ai/{quote(entity, safe='')}/{quote(resolved_project, safe='')}/runs/{quote(run_id, safe='')}",
        workspace,
    )

    def _query() -> str | None:
        try:
            import wandb
            api = wandb.Api(timeout=10)
            run = api.run(f"{entity}/{resolved_project}/{run_id}")
            return _append_wandb_workspace(run.url or fallback, workspace)
        except Exception:
            return fallback

    return await asyncio.to_thread(_query)


_wandb_entity_cache: str | None = None


async def _wandb_entity() -> str | None:
    """Resolve the wandb entity once. Order: env override → API default."""
    global _wandb_entity_cache
    if WANDB_ENTITY_OVERRIDE:
        return WANDB_ENTITY_OVERRIDE
    if _wandb_entity_cache is not None:
        return _wandb_entity_cache or None
    import asyncio

    def _default() -> str:
        try:
            import wandb
            return wandb.Api(timeout=10).default_entity or ""
        except Exception:
            return ""

    _wandb_entity_cache = await asyncio.to_thread(_default)
    return _wandb_entity_cache or None


async def _wandb_step(run_id: str, project: str | None = None) -> int | None:
    """Return the run's latest step (None if API unreachable / run not found)."""
    import asyncio

    entity = await _wandb_entity()
    if not entity:
        return None
    resolved_project = project or get_project()

    def _query() -> int | None:
        try:
            import wandb
            api = wandb.Api(timeout=10)
            run = api.run(f"{entity}/{resolved_project}/{run_id}")
            # train/global_step is the actual training-loop step. wandb's
            # built-in `_step` counts wandb.log() calls, which is
            # global_step / logging_steps — off by ~10× for gr00t-n16.
            s = run.summary.get("train/global_step")
            if s is None:
                s = run.summary.get("global_step")
            if s is None:
                s = run.summary.get("_step")
            return int(s) if s is not None else None
        except Exception:
            return None

    return await asyncio.to_thread(_query)


_BODY_SUFFIX_RE = re.compile(r"/lib/(train|eval)_body(?:_n16)?\.sh$")


def _latest_logs_dir_script(variant: str) -> str:
    """Shell snippet that prints the candidate exp dir with newest logs/.

    Keep this as a loop instead of a shell pipeline built with `;`: without
    grouping, only the last command is piped, which can return
    "<mtime> <path>" instead of just the path when the first candidate wins.
    """
    rels = " ".join(shlex.quote(rel) for rel in exp_dir_rel_candidates(variant))
    return (
        "best_mtime=-1; best_path=''; "
        f"for rel in {rels}; do "
        'c="$HOME/$rel"; '
        'if [ -d "$c/logs" ]; then '
        'm=$(stat -c %Y "$c/logs" 2>/dev/null || echo 0); '
        'case "$m" in ""|*[!0-9]*) m=0;; esac; '
        'if [ "$m" -gt "$best_mtime" ]; then '
        'best_mtime="$m"; best_path="$c"; '
        "fi; "
        "fi; "
        "done; "
        'printf "%s\\n" "$best_path"'
    )


def _existing_exp_dirs_script(variant: str) -> str:
    """Shell snippet that prints existing candidate exp dirs, in order."""
    rels = " ".join(shlex.quote(rel) for rel in exp_dir_rel_candidates(variant))
    return (
        f"for rel in {rels}; do "
        'c="$HOME/$rel"; '
        'if [ -d "$c" ]; then printf "%s\\n" "$c"; fi; '
        "done"
    )

async def _resolve_eval_checkpoint(
    host: str,
    stdout_path: str,
    exp_dir: str,
    submitted_checkpoint: str | None,
) -> str | None:
    """Return the checkpoint path for an eval job.

    Completed/running eval logs are the most accurate source because the body
    only logs `Checkpoint:` after verifying the directory exists. For pending
    or pre-log jobs, fall back to the submit sidecar's explicit checkpoint.
    """
    stdout_q = shlex.quote(stdout_path)
    r = await ssh_run(
        host,
        f"grep -E 'Checkpoint: ' {stdout_q} 2>/dev/null | tail -1",
        timeout=10.0,
    )
    line = r.stdout.strip()
    marker = "Checkpoint: "
    if marker in line:
        checkpoint = line.split(marker, 1)[1].strip()
        if checkpoint:
            return checkpoint

    if submitted_checkpoint:
        checkpoint = submitted_checkpoint.strip()
        if checkpoint:
            return checkpoint

    return None


async def _resolve_exp_dir(host: str, job_id: str, variant: str) -> str:
    """Find the variant's experiment dir on the cluster, per-job.

    Two parallel submission paths write to different roots:
      - web app  → `~/.train-eval-web/experiments/<v>`
      - bash CLI → `~/train-eval-scripts/experiments/<v>`

    Whichever the live job is actually using is the one with a fresh
    `logs/` subdir. We pick by mtime — that handles the case where the
    scontrol Command path doesn't match the running job's REPO_ROOT
    (the user can `bash ./submit` from train-eval-scripts but with a
    body script copied into .train-eval-web, and vice versa).
    """
    # Pick the candidate whose logs/ has the most recent mtime. Returns
    # one line: the winning candidate path, or empty if neither has logs/.
    r = await ssh_run(host, _latest_logs_dir_script(variant), timeout=10.0)
    chosen = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if chosen:
        return chosen

    # No logs/ in either candidate → derive from scontrol's Command path
    # (works while the job is in slurm's recent memory).
    r = await ssh_run(host, f"scontrol show job {job_id} 2>/dev/null | grep -m1 '^   Command='", timeout=10.0)
    if r.returncode == 0 and r.stdout.strip():
        line = r.stdout.strip()
        cmd_path = line.split("=", 1)[1].strip() if "=" in line else ""
        m = _BODY_SUFFIX_RE.search(cmd_path)
        if m:
            repo_root = cmd_path[: m.start()]
            return f"{repo_root}/experiments/{variant}"

    # Last resort: existence probe.
    r = await ssh_run(host, _existing_exp_dirs_script(variant), timeout=10.0)
    lines = r.stdout.strip().splitlines()
    if lines:
        return lines[0]
    return f"$HOME/{exp_dir_rel_candidates(variant)[0]}"


_TQDM_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")


async def _compute_progress(cluster: str, job_id: str, phase: str, variant: str | None,
                             stdout: str, stderr: str,
                             ckpt_dir: str | None, eval_dir: str | None,
                             slurm_meta: dict[str, str] | None = None,
                             wandb_project: str | None = None) -> Progress:
    progress = Progress(phase=phase)
    if not variant:
        return progress

    if phase in ("train", "resume"):
        host = (await load_cluster(cluster)).ssh_alias
        # 1) Live-running jobs: parse latest tqdm step from stderr.
        r = await ssh_run(
            host,
            f"tail -c 4096 {stderr} 2>/dev/null | tr '\\r' '\\n' | grep -oE '[0-9]+/[0-9]+ \\[' | tail -1",
            timeout=10.0,
        )
        m = _TQDM_STEP_RE.search(r.stdout or "")
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            progress.current_step = cur
            progress.max_steps = total
            if total > 0:
                progress.percent = round(100.0 * cur / total, 1)
            progress.current_label = f"step {cur:,}/{total:,}"
            return progress

        # 2) Pending / between-runs / pre-tqdm: derive from the highest
        #    checkpoint dir on disk. Useful for jobs that were preempted and
        #    are queued for requeue — we still want to show how far they got.
        max_steps: int | None = None
        try:
            meta_max_steps = (slurm_meta or {}).get("train_max_steps")
            if meta_max_steps:
                max_steps = int(meta_max_steps)
            else:
                v = await load_variant(variant)
                if "MAX_STEPS" in v.vars:
                    max_steps = int(v.vars["MAX_STEPS"])
        except Exception:
            pass
        progress.max_steps = max_steps

        if not ckpt_dir:
            return progress
        r = await ssh_run(
            host,
            f"ls -d {ckpt_dir}/checkpoint-* 2>/dev/null | sed 's:.*checkpoint-::' | sort -n | tail -1",
            timeout=10.0,
        )
        latest = r.stdout.strip()
        if latest.isdigit():
            cur = int(latest)
            progress.current_step = cur
            if max_steps:
                progress.percent = round(100.0 * cur / max_steps, 1)
            progress.current_label = (
                f"step {cur:,}/{max_steps:,}" if max_steps else f"step {cur:,}"
            )

    elif phase == "eval":
        try:
            v = await load_variant(variant)
            eval_sets, n_runs, n_eps, tasks = eval_shape(v, slurm_meta)
        except FileNotFoundError:
            eval_sets = [s for s in ((slurm_meta or {}).get("eval_sets") or "").split() if s]
            n_runs = _meta_int(slurm_meta or {}, "eval_n_runs") or 0
            n_eps = _meta_int(slurm_meta or {}, "eval_n_episodes") or 0
            tasks = []
        total = max(len(tasks) * len(eval_sets) * n_runs, 0)
        progress.total_runs = total or None

        if not (eval_dir and total > 0):
            return progress

        host = (await load_cluster(cluster)).ssh_alias
        # Completed runs = number of results.json files written by the body.
        r = await ssh_run(
            host,
            f"find {eval_dir} -type f -name results.json 2>/dev/null | wc -l",
            timeout=10.0,
        )
        try:
            completed = int(r.stdout.strip())
        except ValueError:
            completed = 0
        progress.completed_runs = completed

        # Episode counter inside this job. Prefer client stdout because native
        # Isaac vectorization can complete multiple environments per server
        # reset; server reset counts are only a fallback.
        active_eps = 0
        active_envs = 0
        job_completed_runs = 0
        job_log_dir = (slurm_meta or {}).get("job_log_dir") if slurm_meta else None
        if n_eps > 0:
            env = await load_cluster(cluster)
            log_dir_q = shlex.quote(env.vars["LOG_DIR"])
            job_id_q = shlex.quote(job_id)
            stdout_cmd = (
                f"pattern={log_dir_q}/*_{job_id_q}.out; "
                "episodes=$(grep -h '^Episode .* completed' $pattern 2>/dev/null | wc -l); "
                "runs=$(grep -h '^Results saved to:' $pattern 2>/dev/null | wc -l); "
                "echo \"$episodes $runs\""
            )
            r = await ssh_run(host, stdout_cmd, timeout=10.0)
            try:
                parts = r.stdout.strip().split()
                active_eps = int(parts[0]) if parts else 0
                job_completed_runs = int(parts[1]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                active_eps = 0
                job_completed_runs = 0

            job_logs_q = shlex.quote(job_log_dir or f"{eval_dir.rsplit('/', 1)[0]}/logs/{job_id}")
            ep_cmd = (
                f"job_logs={job_logs_q}; "
                "if [ -d \"$job_logs\" ]; then pattern=\"$job_logs/server_*.log\"; "
                "else echo '0 0'; exit 0; fi; "
                "total=0; envs_started=0; "
                "for f in $pattern; do "
                "[ -f \"$f\" ] || continue; "
                "envs=$(grep -m1 'Num envs:' \"$f\" 2>/dev/null | sed -E 's/.*Num envs: ([0-9]+).*/\\1/'); "
                "case \"$envs\" in ''|*[!0-9]*) envs=1 ;; esac; "
                "c=$(grep -c 'Resetting environment with seed:' \"$f\" 2>/dev/null || echo 0); "
                "if [ \"$c\" -gt 0 ]; then c=$((c - 1)); fi; "
                f"if [ \"$c\" -gt {n_eps} ]; then c={n_eps}; fi; "
                "total=$((total + c * envs)); envs_started=$((envs_started + envs)); "
                "done; "
                "echo \"$total $envs_started\""
            )
            r = await ssh_run(host, ep_cmd, timeout=10.0)
            try:
                parts = r.stdout.strip().split()
                active_eps = max(active_eps, int(parts[0]) if parts else 0)
                active_envs = int(parts[1]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                active_envs = 0

        # Promote eval into the unified step-based shape so the frontend
        # ETA + progress bar work the same way as training:
        #   current_step = completed_runs · N_EPISODES + incomplete episodes in this job
        #   max_steps    = total_runs    · N_EPISODES
        if n_eps > 0:
            progress.max_steps = total * n_eps
            active_incomplete_eps = max(0, active_eps - job_completed_runs * n_eps)
            progress.current_step = min(completed * n_eps + active_incomplete_eps, progress.max_steps)
            progress.percent = round(100.0 * progress.current_step / progress.max_steps, 1)
            episode_label = f"{progress.current_step}/{progress.max_steps} episodes"
            progress.current_label = f"{completed}/{total} runs · {episode_label}"
        else:
            progress.percent = round(100.0 * completed / total, 1)
            progress.current_label = f"{completed}/{total} runs"
    return progress
