"""Submit a job to a remote cluster.

Flow:
  1. rsync the local `configs/` + `lib/` into `~/.train-eval-web/` on the cluster
  2. Resolve cluster + variant configs (locally) to derive partition, time, GPUs, body script path.
  3. Build an `sbatch` command targeting the cluster-side body script.
  4. Run it over ssh, parse "Submitted batch job <id>" out of stdout.
"""

import re
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import cluster_settings
from .clusters import load_cluster
from .data_interface import rewrite_action_horizon
from .eval_harness import harness_for
from .job_identity import comment_field_fragment
from .output_namespace import make_output_namespace, validate_output_namespace
from .partitions import is_background_partition
from .paths import CLUSTER_STAGING_REL, CONFIGS_DIR, LIB_DIR
from .resource_presets import slurm_resources_for
from .ssh import rsync_to, ssh_run
from .submission_snapshot import (
    apply_dataset_override,
    metadata_json,
    prepare_slurm_training_git,
    render_eval_config_preview,
    render_training_config_snapshot,
    resolve_train_git_commit_override,
    snapshot_metadata,
    snapshot_suffix,
    shell_array_assignment,
    slurm_training_repo_path,
    training_repo_label,
)
from .training_models import (
    TrainingModel,
    action_horizon_mode_for_variant,
    resolve_training_model,
    rewrites_modality_action_horizon,
)
from .train_overrides import (
    resolve_train_action_horizon as resolve_action_horizon_override,
    validate_global_batch_divisible,
)
from .variants import load_variant
from .variant_values import variant_int
from .wandb_config import get_project as wandb_project


class SubmitRequest(BaseModel):
    cluster: str
    variant: str
    phase: Literal["train", "eval"]
    # Submitted config.sh must carry a non-empty TRAIN_NOTE. None or an empty
    # string falls back to the variant's config value, mirroring job_name.
    train_note: str | None = None
    # Internal job action: retry a timed-out training job from its existing
    # checkpoint. This is intentionally separate from the submit-page phase.
    resume: bool = False
    resume_of: str | None = None
    resubmit_action: Literal["resume", "retry"] | None = None
    # Slurm-only: partition name (None → fall back to cluster.env default).
    partition: str | None = None
    # MLXP-only: which k8s node to pin via nodeAffinity (each rlwrld team
    # member is assigned a specific GPU node in the GPU Resource Schedule
    # sheet). None falls back to the saved MLXP settings default.
    node: str | None = None
    # MLXP-only: scheduling class (mlxp/job-class label). None → dedicated.
    job_class: Literal["dedicated", "normal", "background"] | None = None
    # Per-submit dataset override. Two shapes accepted:
    #   - single string  → replaces DATASET_NAME in single-task variants
    #   - list of "name|cfg|weight" entries → replaces DATASETS array
    # None means "use whatever the variant config.sh says".
    dataset_override: str | list[str] | None = None
    extra_args: list[str] = Field(default_factory=list)
    # Per-submission GPU override. For train it controls torchrun/world size;
    # for eval it controls scheduler allocation and eval worker count.
    train_num_gpus: int | None = Field(default=None, ge=1)
    # Train-only: per-submission overrides. None means "use config.sh".
    train_global_batch_size: int | None = Field(default=None, ge=1)
    train_max_steps: int | None = Field(default=None, ge=1)
    train_save_steps: int | None = Field(default=None, ge=1)
    train_action_horizon: int | None = Field(default=None, ge=1)
    # Optional model-code git commit. If set, the backend verifies the commit
    # exists in the selected model repo and runs from a detached worktree at
    # that exact revision.
    train_git_commit: str | None = None
    # Eval-only: override Isaac's native vectorized env count per GPU.
    eval_num_envs_per_gpu: int | None = Field(default=None, ge=1)
    # Eval-only: per-submission overrides for eval_allex.py and eval matrix.
    eval_n_episodes: int | None = Field(default=None, ge=1)
    eval_n_runs: int | None = Field(default=None, ge=1)
    eval_sets: list[str] | None = None
    eval_overwrite_results: bool = False
    # Eval-only: absolute path to the checkpoint dir on the cluster.
    # Eval submissions must provide this explicitly.
    checkpoint_path: str | None = None
    # Eval-only: DexJoCo task (yaml stem / env_name) chosen via the task picker.
    dexjoco_task: str | None = None
    # Internal eval resume: remote eval_results dir from the timed-out job.
    # Seed the staged eval_results before sbatch so completed runs are skipped.
    seed_eval_results_from: str | list[str] | None = None
    # Internal resume/output control. New user submissions leave this unset;
    # the submitter generates one immutable namespace for checkpoints/results.
    output_namespace: str | None = None
    # Optional override for the auto-generated job_name. Must match
    # `{train|eval}_<anything>_<YYYYMMDD>_<HHMMSS>` so the parser
    # keeps working. None → server builds the default.
    job_name: str | None = None
    # Train-only: set after explicit user approval when the repo is dirty.
    commit_dirty_changes: bool = False


def make_default_job_name(phase: str, variant: str) -> str:
    return f"{phase}_{variant}_{datetime.now():%Y%m%d_%H%M%S}"


def resolve_job_name(req_job_name: str | None, phase: str, variant: str) -> str:
    """Return user-provided job_name if non-empty, else build the default.

    No format validation — caller may pass any string. Note that names that
    don't match `{phase}_<slug>_<YYYYMMDD>_<HHMMSS>` will resolve to
    ("unknown", None) in parse_phase_and_variant, so phase/variant won't be
    derivable from the name.
    """
    if req_job_name is None:
        return make_default_job_name(phase, variant)
    name = req_job_name.strip()
    if not name:
        return make_default_job_name(phase, variant)
    return name


def resolve_train_note(requested_note: str | None, variant) -> str:
    requested = requested_note.strip() if requested_note is not None else ""
    raw = requested or variant.vars.get("TRAIN_NOTE", "")
    note = raw.strip()
    if not note:
        raise ValueError("TRAIN_NOTE is required")
    if "\n" in note or "\r" in note:
        raise ValueError("TRAIN_NOTE must be a single line")
    return note


def slurm_comment_metadata(
    *,
    phase: str,
    variant: str,
    model_id: str,
    output_namespace: str | None = None,
    resume_of: str | None = None,
    resubmit_action: str | None = None,
) -> str:
    """Small scheduler Comment fallback.

    Full submission metadata lives in the per-job sidecar after sbatch
    succeeds. Slurm's Comment field has a tighter controller-side limit, so
    keep it short enough for resubmits where paths already have long names.
    """
    return comment_field_fragment(
        {
            "phase": phase,
            "variant": variant,
            "model_id": model_id,
            "output_namespace": output_namespace,
            "resume_of": (resume_of or "").strip() or None,
            "resubmit_action": resubmit_action,
        }
    )


def normalize_eval_sets(eval_sets: list[str] | None) -> list[str] | None:
    if eval_sets is None:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for raw in eval_sets:
        item = raw.strip()
        if not item or item in seen:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", item):
            raise ValueError(f"invalid eval set {item!r}; use letters, numbers, dot, underscore, or hyphen")
        seen.add(item)
        out.append(item)
    if not out:
        raise ValueError("eval_sets must contain at least one eval set")
    return out


def require_eval_checkpoint_path(req: SubmitRequest) -> str:
    path = (req.checkpoint_path or "").strip()
    if req.phase == "eval" and not path:
        raise ValueError("checkpoint_path is required for eval")
    return path


@dataclass(frozen=True)
class TrainSettings:
    num_gpus: int
    global_batch_size: int | None
    max_steps: int
    save_steps: int


def resolve_train_settings(variant, model_family: str, *, num_gpus_override: int | None,
                           global_batch_override: int | None, max_steps_override: int | None,
                           save_steps_override: int | None) -> TrainSettings:
    """Resolve train-time override values exactly once for submit and preview."""
    train_num_gpus = num_gpus_override or variant_int(variant, "TRAIN_NUM_GPUS", 2)
    train_max_steps = max_steps_override or variant_int(variant, "MAX_STEPS", 30000)
    train_save_steps = save_steps_override or variant_int(variant, "SAVE_STEPS", 1000)
    train_global_batch_size = global_batch_override

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

    validate_global_batch_divisible(model_family, train_global_batch_size, train_num_gpus)

    return TrainSettings(
        num_gpus=train_num_gpus,
        global_batch_size=train_global_batch_size,
        max_steps=train_max_steps,
        save_steps=train_save_steps,
    )


def resolve_train_action_horizon(
    req: SubmitRequest,
    variant,
    model: TrainingModel,
    action_horizon_mode: str | None = None,
) -> int | None:
    return resolve_action_horizon_override(
        variant=variant,
        model=model,
        action_horizon_mode=action_horizon_mode,
        requested=req.train_action_horizon,
    )


def resolve_train_git_commit(req: SubmitRequest, variant) -> str | None:
    return resolve_train_git_commit_override(
        req.train_git_commit,
        variant.vars,
    )


class SubmitResponse(BaseModel):
    job_id: str
    job_name: str
    partition: str
    sbatch_cmd: str
    rsync_stdout: str
    sbatch_stdout: str


MAX_EVAL_NUM_ENVS_PER_GPU = 1


@dataclass(frozen=True)
class ConfigSnapshotPaths:
    suffix: str
    rel: str
    meta_rel: str
    modality_rel: str
    path: str
    meta_path: str
    modality_path: str


def config_snapshot_paths(variant: str, job_name: str) -> ConfigSnapshotPaths:
    suffix = snapshot_suffix(job_name)
    rel = f"{CLUSTER_STAGING_REL}/experiments/{variant}/config_{suffix}.sh"
    meta_rel = f"{CLUSTER_STAGING_REL}/experiments/{variant}/config_{suffix}.meta.json"
    modality_rel = f"{CLUSTER_STAGING_REL}/experiments/{variant}/modality_{suffix}.py"
    return ConfigSnapshotPaths(
        suffix=suffix,
        rel=rel,
        meta_rel=meta_rel,
        modality_rel=modality_rel,
        path=f"$HOME/{rel}",
        meta_path=f"$HOME/{meta_rel}",
        modality_path=f"$HOME/{modality_rel}",
    )


async def _rsync_text(host: str, text: str, remote_rel: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix="_submit_snapshot", delete=False) as fp:
        fp.write(text)
        tmp_path = fp.name
    try:
        parent = str(Path(remote_rel).parent)
        if parent and parent != ".":
            mkdir = await ssh_run(host, f"mkdir -p $HOME/{shlex.quote(parent)}", timeout=10.0)
            if mkdir.returncode != 0:
                raise RuntimeError(f"mkdir for snapshot failed: {mkdir.stderr or mkdir.stdout}")
        r = await rsync_to(host, tmp_path, remote_rel)
        if r.returncode != 0:
            raise RuntimeError(f"rsync snapshot failed: {r.stderr}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def submit(req: SubmitRequest) -> SubmitResponse:
    if req.resume and req.phase != "train":
        raise ValueError("resume=true is only valid for phase=train")
    eval_num_envs_per_gpu = req.eval_num_envs_per_gpu
    eval_sets = normalize_eval_sets(req.eval_sets)
    eval_checkpoint = require_eval_checkpoint_path(req) if req.phase == "eval" else None
    if req.phase != "eval" and any((
        eval_num_envs_per_gpu is not None,
        req.eval_n_episodes is not None,
        req.eval_n_runs is not None,
        eval_sets is not None,
        req.eval_overwrite_results,
        bool(req.checkpoint_path and req.checkpoint_path.strip()),
    )):
        raise ValueError("eval overrides are only valid for phase=eval")
    if req.phase != "train" and any((
        req.train_global_batch_size is not None,
        req.train_max_steps is not None,
        req.train_save_steps is not None,
        req.train_action_horizon is not None,
    )):
        raise ValueError("train overrides are only valid for phase=train")
    if (
        req.phase == "eval"
        and eval_num_envs_per_gpu is not None
        and eval_num_envs_per_gpu > MAX_EVAL_NUM_ENVS_PER_GPU
    ):
        raise ValueError(
            "eval_num_envs_per_gpu > 1 is disabled: the ALLEX target reset "
            "path is not vector-env safe"
        )

    cluster = await load_cluster(req.cluster)
    variant = await load_variant(req.variant)
    train_note = resolve_train_note(req.train_note, variant)

    if req.phase == "eval":
        harness_for(variant).validate_submit(req)

    # ── Resolve partition + model + body script + walltime ──
    model = resolve_training_model(variant)
    action_horizon_mode = action_horizon_mode_for_variant(model, variant)
    body_script, walltime = model.body_for_phase(req.phase)
    training_repo = slurm_training_repo_path(cluster.vars, model)

    partition = req.partition or cluster.vars["PARTITION"]
    sbatch_flags: list[str] = []
    if is_background_partition(partition):
        sbatch_flags.append("--requeue")
    exclude_nodes = (cluster.vars.get("SBATCH_EXCLUDE") or cluster.vars.get("SLURM_EXCLUDE_NODES") or "").strip()
    if exclude_nodes:
        sbatch_flags.append(f"--exclude={shlex.quote(exclude_nodes)}")

    train_settings = resolve_train_settings(
        variant,
        model.family,
        num_gpus_override=req.train_num_gpus,
        global_batch_override=req.train_global_batch_size,
        max_steps_override=req.train_max_steps,
        save_steps_override=req.train_save_steps,
    )
    train_action_horizon = resolve_train_action_horizon(
        req,
        variant,
        model,
        action_horizon_mode,
    ) if req.phase == "train" else None
    train_git_commit = resolve_train_git_commit(req, variant)
    gpus = str(train_settings.num_gpus)
    slurm_resources = slurm_resources_for(
        cluster=req.cluster,
        partition=partition,
        phase=req.phase,
        num_gpus=train_settings.num_gpus,
    )

    # Unified shape across slurm + MLXP. The cluster/partition are shown in
    # table columns; the job name carries phase, variant, and timestamp.
    job_name = resolve_job_name(req.job_name, req.phase, req.variant)
    output_namespace = (
        validate_output_namespace(req.output_namespace)
        if req.output_namespace
        else (None if req.resume else make_output_namespace(job_name, req.variant))
    )
    host = cluster.ssh_alias
    submitted_wandb_project = wandb_project()
    exp_dir_remote = f"$HOME/{CLUSTER_STAGING_REL}/experiments/{req.variant}"
    checkpoint_dir = (
        f"{exp_dir_remote}/checkpoints/{output_namespace}"
        if req.phase == "train" and output_namespace
        else None
    )
    eval_dir = (
        f"{exp_dir_remote}/eval_results/{output_namespace}"
        if req.phase == "eval" and output_namespace
        else (f"{exp_dir_remote}/eval_results" if req.phase == "eval" else None)
    )
    results_path = f"{eval_dir}/results.json" if eval_dir else None
    job_log_dir = f"{exp_dir_remote}/logs/{output_namespace}" if output_namespace else ""
    snapshot_paths = config_snapshot_paths(req.variant, job_name)

    snapshot_rel = snapshot_paths.rel
    snapshot_meta_rel = snapshot_paths.meta_rel
    snapshot_path = snapshot_paths.path
    snapshot_meta_path = snapshot_paths.meta_path
    snapshot_modality_rel: str | None = None
    snapshot_modality_path: str | None = None
    snapshot_modality_text: str | None = None
    snapshot_text: str | None = None
    snapshot_meta_text: str | None = None
    submit_extra_args_rel: str | None = None
    submit_extra_args_path: str | None = None
    submit_extra_args_text: str | None = None
    submit_git = await prepare_slurm_training_git(
        host=host,
        repo_path=training_repo,
        repo_label=training_repo_label(model),
        job_name=job_name,
        commit_dirty_changes=req.commit_dirty_changes if req.phase == "train" else False,
        require_clean=(
            req.phase == "train"
            and not req.resume
            and not (req.resume_of and req.resume_of.strip())
        ),
        requested_commit=train_git_commit,
    )

    if req.phase == "train":
        if req.extra_args:
            safe_job_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", job_name).strip("_") or snapshot_paths.suffix
            submit_extra_args_rel = f"{CLUSTER_STAGING_REL}/jobs/{safe_job_name}.extra_args.sh"
            submit_extra_args_path = f"$HOME/{submit_extra_args_rel}"
            submit_extra_args_text = (
                "# Generated by train-eval-web for this submission.\n"
                + shell_array_assignment("SUBMIT_EXTRA_ARGS", req.extra_args)
                + "\n"
            )
        if rewrites_modality_action_horizon(action_horizon_mode) and train_action_horizon is not None:
            modality_rel = (variant.vars.get("TRAIN_MODALITY_CONFIG") or "").strip()
            if not modality_rel:
                raise ValueError(f"variant {variant.name}: TRAIN_MODALITY_CONFIG missing")
            if Path(modality_rel).is_absolute() or ".." in Path(modality_rel).parts:
                raise ValueError("TRAIN_MODALITY_CONFIG must be relative to the experiment directory")
            modality_path = CONFIGS_DIR / "experiments" / req.variant / modality_rel
            if not modality_path.is_file():
                raise FileNotFoundError(f"modality config not found: {modality_path}")
            snapshot_modality_rel = snapshot_paths.modality_rel
            snapshot_modality_path = snapshot_paths.modality_path
            snapshot_modality_text = rewrite_action_horizon(
                modality_path.read_text(),
                train_action_horizon,
            )
        snapshot_text = render_training_config_snapshot(
            base_config=variant.raw,
            variant=req.variant,
            model=model.family,
            job_name=job_name,
            cluster=req.cluster,
            partition=partition,
            dataset_override=req.dataset_override,
            extra_args=req.extra_args,
            train_num_gpus=train_settings.num_gpus,
            train_global_batch_size=train_settings.global_batch_size,
            train_max_steps=train_settings.max_steps,
            train_save_steps=train_settings.save_steps,
            train_action_horizon=train_action_horizon,
            train_modality_config=Path(snapshot_modality_rel).name if snapshot_modality_rel else None,
            train_git_commit=train_git_commit,
            train_note=train_note,
            wandb_project=submitted_wandb_project,
            git=submit_git,
        )
        snapshot_meta_text = metadata_json(snapshot_metadata(
            job_name=job_name,
            cluster=req.cluster,
            variant=req.variant,
            path=snapshot_path,
            meta_path=snapshot_meta_path,
            partition=partition,
            dataset_override=req.dataset_override,
            extra_args=req.extra_args,
            train_num_gpus=train_settings.num_gpus,
            train_global_batch_size=train_settings.global_batch_size,
            train_max_steps=train_settings.max_steps,
            train_save_steps=train_settings.save_steps,
            train_action_horizon=train_action_horizon,
            train_modality_config=snapshot_modality_path,
            train_git_commit=train_git_commit,
            train_note=train_note,
            wandb_project=submitted_wandb_project,
            git=submit_git,
        ))
    else:
        snapshot_text = render_eval_config_preview(
            base_config=variant.raw,
            variant=req.variant,
            job_name=job_name,
            cluster=req.cluster,
            partition=partition,
            dataset_override=req.dataset_override,
            eval_n_episodes=req.eval_n_episodes,
            eval_n_runs=req.eval_n_runs,
            eval_sets=eval_sets,
            eval_overwrite_results=req.eval_overwrite_results,
            checkpoint_path=eval_checkpoint,
            extra_args=req.extra_args,
            train_num_gpus=train_settings.num_gpus,
            train_git_commit=train_git_commit,
            train_note=train_note,
            dexjoco_task=req.dexjoco_task,
        )
        snapshot_meta_text = metadata_json(snapshot_metadata(
            job_name=job_name,
            cluster=req.cluster,
            phase="eval",
            variant=req.variant,
            path=snapshot_path,
            meta_path=snapshot_meta_path,
            partition=partition,
            dataset_override=req.dataset_override,
            extra_args=req.extra_args,
            train_num_gpus=train_settings.num_gpus,
            train_git_commit=train_git_commit,
            train_note=train_note,
            wandb_project=submitted_wandb_project,
            git=submit_git,
        ))

    # ── Sync code to cluster staging ──
    # Body scripts expect $REPO_ROOT/{clusters,experiments,lib}/ at the staging
    # root, so flatten configs/ on the way out: configs/clusters → clusters/,
    # configs/experiments → experiments/.
    staging = f"$HOME/{CLUSTER_STAGING_REL}"
    mkdir_result = await ssh_run(host, f"mkdir -p {staging}/clusters {staging}/experiments {staging}/models {staging}/lib {staging}/jobs")
    if mkdir_result.returncode != 0:
        raise RuntimeError(f"mkdir on cluster failed: {mkdir_result.stderr}")

    rsync_results = []
    # (local source with trailing slash, remote target dir name)
    sync_targets = [
        (str(CONFIGS_DIR / "clusters") + "/",    "clusters"),
        (str(CONFIGS_DIR / "experiments") + "/", "experiments"),
        (str(CONFIGS_DIR / "models") + "/",      "models"),
        (str(LIB_DIR) + "/",                      "lib"),
    ]
    for local, remote_name in sync_targets:
        remote = f"{CLUSTER_STAGING_REL}/{remote_name}"
        excludes = (
            [
                "checkpoints/",
                "eval_results/",
                "logs/",
                "results.json",
                "config_*.sh",
                "config_*.meta.json",
                "modality_*.py",
                "extra_args_*.sh",
            ]
            if remote_name == "experiments"
            else None
        )
        r = await rsync_to(host, local, remote, delete=True, exclude=excludes)
        if r.returncode != 0:
            raise RuntimeError(f"rsync failed for {local}: {r.stderr}")
        rsync_results.append(r)

    await _rsync_text(
        host,
        cluster_settings.load_env_text(req.cluster),
        f"{CLUSTER_STAGING_REL}/clusters/{req.cluster}.env",
    )

    # Apply dataset override to the staged config.sh, if requested.
    if req.dataset_override is not None:
        modified = apply_dataset_override(variant.raw, req.dataset_override)
        if modified != variant.raw:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_config.sh", delete=False,
            ) as fp:
                fp.write(modified)
                tmp_path = fp.name
            try:
                remote_cfg = f"{CLUSTER_STAGING_REL}/experiments/{req.variant}/config.sh"
                r = await rsync_to(host, tmp_path, remote_cfg)
                if r.returncode != 0:
                    raise RuntimeError(f"rsync override failed: {r.stderr}")
                rsync_results.append(r)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    if snapshot_rel and snapshot_text and snapshot_meta_rel and snapshot_meta_text:
        await _rsync_text(host, snapshot_text, snapshot_rel)
        await _rsync_text(host, snapshot_meta_text, snapshot_meta_rel)
    if snapshot_modality_rel and snapshot_modality_text:
        await _rsync_text(host, snapshot_modality_text, snapshot_modality_rel)
    if submit_extra_args_rel and submit_extra_args_text:
        await _rsync_text(host, submit_extra_args_text, submit_extra_args_rel)

    if req.phase == "eval" and req.seed_eval_results_from:
        dst_rel = (
            f"{CLUSTER_STAGING_REL}/experiments/{req.variant}/eval_results/{output_namespace}"
            if output_namespace
            else f"{CLUSTER_STAGING_REL}/experiments/{req.variant}/eval_results"
        )
        seed_sources = (
            [req.seed_eval_results_from]
            if isinstance(req.seed_eval_results_from, str)
            else req.seed_eval_results_from
        )
        for source in seed_sources:
            src = source.strip().rstrip("/")
            if not src:
                continue
            seed_cmd = (
                f"src={shlex.quote(src)}; dst=$HOME/{shlex.quote(dst_rel)}; "
                "mkdir -p \"$dst\" && "
                "if [ \"$src\" != \"$dst\" ] && [ -d \"$src\" ]; then "
                "rsync -a --ignore-existing \"$src\"/ \"$dst\"/; "
                "fi"
            )
            seeded = await ssh_run(host, seed_cmd, timeout=120.0)
            if seeded.returncode != 0:
                raise RuntimeError(f"seeding eval results failed: {seeded.stderr or seeded.stdout}")

    # ── Build sbatch command ──
    log_dir = cluster.vars["LOG_DIR"]
    resume_expected = "1" if req.resume else "0"

    body_path = f"$HOME/{CLUSTER_STAGING_REL}/lib/{body_script}"
    repo_root_remote = f"$HOME/{CLUSTER_STAGING_REL}"

    # Persist only identity/link metadata in sacct's Comment field so the
    # details page can recover custom job names even if the sidecar is missing.
    # Full paths and git fields are written to the sidecar below; keeping them
    # out of --comment avoids Slurm's "parameter too long" rejection on resume.
    comment = slurm_comment_metadata(
        phase=req.phase,
        variant=req.variant,
        model_id=model.id,
        output_namespace=output_namespace,
        resume_of=req.resume_of,
        resubmit_action=req.resubmit_action,
    )

    sbatch_parts = [
        "/opt/slurm/bin/sbatch",
        f"--job-name={shlex.quote(job_name)}",
        f"--partition={shlex.quote(partition)}",
        "--nodes=1",
        f"--gpus-per-node={shlex.quote(gpus)}",
        *(
            [
                f"--cpus-per-task={slurm_resources.cpus_per_task}",
                f"--mem={shlex.quote(slurm_resources.memory)}",
            ]
            if slurm_resources is not None
            else []
        ),
        f"--time={shlex.quote(walltime)}",
        f"--output={log_dir}/{job_name}_%j.out",
        f"--error={log_dir}/{job_name}_%j.err",
        f"--comment={shlex.quote(comment)}",
        f"--export=ALL,VARIANT={shlex.quote(req.variant)},CLUSTER={shlex.quote(req.cluster)},"
        f"REPO_ROOT={repo_root_remote},RESUME_EXPECTED={resume_expected},"
        f"SUBMIT_PARTITION={shlex.quote(partition)},"
        f"SUBMIT_TRAIN_REPO_DIR={shlex.quote(training_repo)},"
        f"SUBMIT_WANDB_PROJECT={shlex.quote(submitted_wandb_project)}"
        + f",SUBMIT_CONFIG_FILE=$HOME/{shlex.quote(snapshot_rel)}"
        + (
            f",SUBMIT_OUTPUT_NAMESPACE={shlex.quote(output_namespace)}"
            if output_namespace else ""
        )
        # Pin wandb run id to the slurm display name so the URL is stable
        # and matches MLXP's run-id format.
        + f",WANDB_RUN_ID={shlex.quote(job_name)}"
        + f",SUBMIT_TRAIN_NUM_GPUS={train_settings.num_gpus}"
        + (
            f",SUBMIT_TRAIN_MAX_STEPS={train_settings.max_steps},"
            f"SUBMIT_TRAIN_SAVE_STEPS={train_settings.save_steps}"
            if req.phase == "train" else ""
        )
        + (
            f",SUBMIT_TRAIN_GLOBAL_BATCH_SIZE={req.train_global_batch_size}"
            if req.phase == "train" and req.train_global_batch_size is not None else ""
        )
        + (
            f",SUBMIT_TRAIN_ACTION_HORIZON={train_action_horizon}"
            if req.phase == "train" and train_action_horizon is not None else ""
        )
        + (
            f",SUBMIT_ACTION_HORIZON_MODE={shlex.quote(action_horizon_mode)}"
            if req.phase == "train" else ""
        )
        + (
            f",SUBMIT_EXTRA_ARGS_FILE={submit_extra_args_path}"
            if req.phase == "train" and submit_extra_args_path else ""
        )
        + (
            f",SUBMIT_GIT_COMMIT={shlex.quote(submit_git.commit)}"
            if submit_git and submit_git.commit else ""
        )
        + (
            f",SUBMIT_EVAL_NUM_ENVS_PER_GPU={eval_num_envs_per_gpu}"
            if req.phase == "eval" and eval_num_envs_per_gpu is not None else ""
        )
        + (
            f",SUBMIT_EVAL_N_EPISODES={req.eval_n_episodes}"
            if req.phase == "eval" and req.eval_n_episodes is not None else ""
        )
        + (
            f",SUBMIT_EVAL_N_RUNS={req.eval_n_runs}"
            if req.phase == "eval" and req.eval_n_runs is not None else ""
        )
        + (
            f",SUBMIT_EVAL_SETS={shlex.quote(' '.join(eval_sets))}"
            if req.phase == "eval" and eval_sets is not None else ""
        )
        + (
            ",SUBMIT_EVAL_OVERWRITE_RESULTS=1"
            if req.phase == "eval" and req.eval_overwrite_results else ""
        )
        + (
            f",EVAL_CHECKPOINT={shlex.quote(eval_checkpoint)}"
            if req.phase == "eval" and eval_checkpoint else ""
        )
        + (
            f",SUBMIT_DEXJOCO_TASK={shlex.quote(req.dexjoco_task)}"
            if req.phase == "eval" and req.dexjoco_task else ""
        ),
        *sbatch_flags,
        *([] if req.phase == "train" else [shlex.quote(a) for a in req.extra_args]),
        body_path,
    ]
    # Fallback to which-sbatch if /opt/slurm/bin/sbatch missing:
    sbatch_cmd = (
        "SBATCH_BIN=$(command -v sbatch 2>/dev/null || echo /opt/slurm/bin/sbatch); "
        + " ".join(sbatch_parts).replace("/opt/slurm/bin/sbatch", "$SBATCH_BIN", 1)
    )

    sb = await ssh_run(host, sbatch_cmd, timeout=30.0)
    if sb.returncode != 0:
        raise RuntimeError(f"sbatch failed: {sb.stderr or sb.stdout}")

    m = re.search(r"Submitted batch job (\d+)", sb.stdout)
    if not m:
        raise RuntimeError(f"could not parse sbatch output: {sb.stdout!r}")
    job_id = m.group(1)

    # Persistent sidecar so the details page can recover phase/variant for
    # this job_id forever. Slurm's --comment is unreliable: it's on the
    # live controller (scontrol) but most slurmdbd setups (kakao's
    # included) don't archive it to sacct.
    meta_dir = "$HOME/.train-eval-web/jobs"
    meta = (
        f"phase={req.phase}\n"
        f"variant={req.variant}\n"
        f"model_id={model.id}\n"
        f"model_label={model.label}\n"
        f"submit_train_repo_dir={training_repo}\n"
        f"wandb_project={submitted_wandb_project}\n"
        f"train_note={train_note}\n"
        f"job_name={job_name}\n"
        + (
            f"output_namespace={output_namespace}\n"
            if output_namespace else ""
        )
        + (
            f"job_log_dir={job_log_dir}\n"
            if job_log_dir else ""
        )
        + (
            f"resume_of={req.resume_of.strip()}\n"
            if req.resume_of and req.resume_of.strip()
            else ""
        )
        + (
            f"resubmit_action={req.resubmit_action}\n"
            if req.resubmit_action
            else ""
        )
        + ("resume=true\n" if req.resume else "")
        + (
            f"eval_num_envs_per_gpu={eval_num_envs_per_gpu}\n"
            if req.phase == "eval" and eval_num_envs_per_gpu is not None
            else ""
        )
        + (
            f"eval_num_gpus={train_settings.num_gpus}\n"
            if req.phase == "eval"
            else ""
        )
        + (
            f"train_num_gpus={train_settings.num_gpus}\n"
            f"train_max_steps={train_settings.max_steps}\n"
            f"train_save_steps={train_settings.save_steps}\n"
            if req.phase == "train"
            else ""
        )
        + (
            f"checkpoint_dir={checkpoint_dir}\n"
            if req.phase == "train" and checkpoint_dir
            else ""
        )
        + (
            f"train_action_horizon={train_action_horizon}\n"
            if req.phase == "train" and train_action_horizon is not None
            else ""
        )
        + (
            f"train_modality_config={snapshot_modality_path}\n"
            if req.phase == "train" and snapshot_modality_path
            else ""
        )
        + (
            f"train_global_batch_size={req.train_global_batch_size}\n"
            if req.phase == "train" and req.train_global_batch_size is not None
            else ""
        )
        + (
            f"config_snapshot_path={snapshot_path}\n"
            f"config_snapshot_rel={snapshot_rel}\n"
            f"config_snapshot_meta_path={snapshot_meta_path}\n"
            f"config_snapshot_meta_rel={snapshot_meta_rel}\n"
            if snapshot_path and snapshot_rel and snapshot_meta_path and snapshot_meta_rel
            else ""
        )
        + (
            f"submit_extra_args_path={submit_extra_args_path}\n"
            f"submit_extra_args_rel={submit_extra_args_rel}\n"
            if req.phase == "train" and submit_extra_args_path and submit_extra_args_rel
            else ""
        )
        + (
            f"submit_git_repo_path={submit_git.repo_path}\n"
            f"submit_git_repo_label={submit_git.repo_label}\n"
            f"submit_git_branch={submit_git.branch or ''}\n"
            f"submit_git_commit={submit_git.commit}\n"
            f"submit_git_commit_subject={submit_git.commit_subject or ''}\n"
            f"submit_git_dirty_at_submit={'true' if submit_git.dirty_before else 'false'}\n"
            f"submit_git_committed_dirty={'true' if submit_git.committed_dirty else 'false'}\n"
            if submit_git
            else ""
        )
        + (
            f"exclude_nodes={exclude_nodes}\n"
            if exclude_nodes
            else ""
        )
        + (
            f"eval_n_episodes={req.eval_n_episodes}\n"
            if req.phase == "eval" and req.eval_n_episodes is not None
            else ""
        )
        + (
            f"eval_n_runs={req.eval_n_runs}\n"
            if req.phase == "eval" and req.eval_n_runs is not None
            else ""
        )
        + (
            f"eval_sets={' '.join(eval_sets)}\n"
            if req.phase == "eval" and eval_sets is not None
            else ""
        )
        + (
            "eval_overwrite_results=true\n"
            if req.phase == "eval" and req.eval_overwrite_results
            else ""
        )
        + (
            f"checkpoint_path={eval_checkpoint}\n"
            if req.phase == "eval" and eval_checkpoint
            else ""
        )
        + (
            f"eval_dir={eval_dir}\n"
            f"results_path={results_path}\n"
            if req.phase == "eval" and eval_dir and results_path
            else ""
        )
    )
    meta_cmd = (
        f"mkdir -p {meta_dir} && "
        f"cat > {meta_dir}/{job_id}.meta <<'EOF'\n{meta}EOF"
    )
    await ssh_run(host, meta_cmd, timeout=15.0)

    return SubmitResponse(
        job_id=job_id,
        job_name=job_name,
        partition=partition,
        sbatch_cmd=sbatch_cmd,
        rsync_stdout="\n".join(r.stdout for r in rsync_results),
        sbatch_stdout=sb.stdout,
    )
