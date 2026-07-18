"""Render a k8s Job YAML for a gr00t training variant and `kubectl apply` it.

Mirrors the slurm submit flow conceptually:
  - load the variant config (DATASETS, MAX_STEPS, …)
  - render a Job YAML that runs gr00t_finetune.py against the user's MLXP DDN
  - apply it with `kubectl apply`, parse the returned Job name

Different from slurm:
  - no partition picker (k8s scheduler does its thing); user picks `num_gpus`
    and we map to CPU/memory per the Notion guide's table
  - the body script is inlined into the Job spec's `args` (no separate
    train_body.sh file synced to the cluster — DDN already has the gr00t repo)
  - logs/status come from `kubectl logs` / `kubectl get pod`, not slurm tools
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import shlex
import shutil
import tarfile
import time
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .data_interface import rewrite_action_horizon
from .job_identity import comment_field_fragment
from .mlxp_config import (
    MlxpSettings,
    get_settings,
    labels,
)
from . import paths
from .output_namespace import make_output_namespace, validate_output_namespace
from .paths import EXPERIMENTS_DIR, LIB_DIR
from .submission_snapshot import (
    ensure_trailing_newline,
    is_safe_relpath,
    metadata_json,
    prepare_mlxp_training_git,
    render_eval_config_preview,
    render_training_config_snapshot,
    resolve_modality_config,
    resolve_train_git_commit_override,
    snapshot_metadata,
    snapshot_suffix,
    training_repo_label,
)
from .training_models import (
    TrainingModel,
    action_horizon_mode_for_variant,
    load_training_model,
    mlxp_repo_path,
    passes_action_horizon_cli,
    resolve_training_model,
    rewrites_modality_action_horizon,
)
from .train_overrides import DEFAULT_TRAIN_NUM_GPUS, resolve_train_action_horizon
from .wandb_config import get_project as _wandb_project
from .variants import DEFAULT_DATA_CONFIG, load_variant


# Per-GPU resource map (from the Notion MLXP guide section 3.1).
# Node total: CPU=112, memory=1760Gi, GPU=8.
_GPU_RESOURCES = {
    1: ("14",  "220Gi"),
    2: ("28",  "440Gi"),
    4: ("56",  "880Gi"),
    8: ("100", "1500Gi"),
}

_K8S_NAME_RE = re.compile(r"[^a-z0-9-]+")


def _k8s_name_segment(value: str) -> str:
    segment = _K8S_NAME_RE.sub("-", value.lower()).strip("-")
    segment = re.sub(r"-{2,}", "-", segment)
    return segment or "job"


def _mlxp_job_id(settings: MlxpSettings, job_name: str) -> str:
    """Kubernetes metadata.name following the MLXP guide:
    `<user>-<job-name>`, with DNS-label sanitation and 63-char max length.

    A short digest of the original job_name is always appended so two distinct
    job_names that sanitize to the same DNS label do not collide (which would
    make `kubectl apply` silently replace an existing Job). The digest is over
    the full pre-sanitation name, which carries the `_HHMMSS` submit timestamp.
    """
    prefix = _k8s_name_segment(settings.user)
    body = _k8s_name_segment(job_name)
    digest = hashlib.sha1(job_name.encode()).hexdigest()[:8]
    # The default job_name already starts with the username (make_default_job_name
    # -> "<user>_<phase>_..."), so prepending the user again produced a doubled
    # "youngwoong-youngwoong-..." k8s name. Only add the prefix when the body
    # doesn't already carry it, keeping the required <user>-scoped name.
    if body == prefix or body.startswith(f"{prefix}-"):
        name = body
    else:
        name = f"{prefix}-{body}"
    keep = 63 - len(digest) - 1
    return f"{name[:keep].rstrip('-')}-{digest}"


def _hf_cache_exports(settings: MlxpSettings) -> str:
    hf_home = shlex.quote(settings.hf_home)
    return f"""\
export HF_HOME={hf_home}
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
"""


def _uv_bootstrap_block(settings: MlxpSettings) -> str:
    """Install uv into the DDN user base when it isn't already on PATH."""
    uv_userbase = shlex.quote(f"{settings.ddn_user_home}/.local")
    return f"""\
if ! command -v uv >/dev/null 2>&1; then
    PYTHONUSERBASE={uv_userbase} python3 -m pip install --user uv
fi"""


def _strip_resume_state_block(ckpt_dir: str, max_steps: str) -> str:
    """Shell snippet that strips resume-only trainer state from each step dir.

    Byte-identical between the n1.5 and n1.6 body scripts; keeps the same
    deployable core files the checkpoint-copy feature keeps.
    """
    return f"""\
# Training complete — strip resume-only trainer state from each step dir,
# keeping the same deployable core files the checkpoint-copy feature keeps.
if [ -d "{ckpt_dir}/checkpoint-{max_steps}" ]; then
    echo "[mlxp] removing resume-only trainer state under {ckpt_dir}"
    for step_dir in "{ckpt_dir}"/checkpoint-*/; do
        [ -d "$step_dir" ] || continue
        rm -rf "$step_dir"global_step* "$step_dir"optimizer* "$step_dir"scheduler.pt \\
               "$step_dir"rng_state_*.pth "$step_dir"trainer_state.json \\
               "$step_dir"latest "$step_dir"zero_to_fp32.py || true
    done
fi"""


def _mlxp_isaac_assets_block(settings: MlxpSettings) -> str:
    """Link DDN-stored ALLEX assets into the Isaac repo inside the eval image."""
    asset_root = f"{settings.workspace_dir.rstrip('/')}/rlwrld_isaac"
    quoted_asset_root = shlex.quote(asset_root)
    return f"""\
ALLEX_ASSET_ROOT={quoted_asset_root}
if [ "$ALLEX_ASSET_ROOT" != "$ISAAC_DIR" ]; then
    if [ ! -d "$ALLEX_ASSET_ROOT/objects" ]; then
        echo "[mlxp] missing ALLEX objects: $ALLEX_ASSET_ROOT/objects" >&2
        exit 1
    fi
    if [ ! -d "$ALLEX_ASSET_ROOT/source/allex_sim/allex_sim/assets" ]; then
        echo "[mlxp] missing ALLEX assets: $ALLEX_ASSET_ROOT/source/allex_sim/allex_sim/assets" >&2
        exit 1
    fi
    mkdir -p "$ISAAC_DIR/source/allex_sim/allex_sim"
    rm -rf "$ISAAC_DIR/objects" "$ISAAC_DIR/source/allex_sim/allex_sim/assets"
    ln -sfnT "$ALLEX_ASSET_ROOT/objects" "$ISAAC_DIR/objects"
    ln -sfnT "$ALLEX_ASSET_ROOT/source/allex_sim/allex_sim/assets" "$ISAAC_DIR/source/allex_sim/allex_sim/assets"
    echo "[mlxp] linked ALLEX assets from $ALLEX_ASSET_ROOT"
fi
if [ ! -e "$ISAAC_DIR/source/allex_sim/allex_sim/assets/ALLEX_simple.usd" ]; then
    echo "[mlxp] missing ALLEX_simple.usd in $ISAAC_DIR/source/allex_sim/allex_sim/assets" >&2
    exit 1
fi
"""

def mlxp_training_repo_path(model: str | TrainingModel) -> str:
    resolved = model if isinstance(model, TrainingModel) else load_training_model(model)
    settings = get_settings()
    return mlxp_repo_path(
        resolved,
        {
            "DDN_USER_HOME": settings.ddn_user_home,
            "DDN_MOUNT": settings.ddn_mount,
        },
    )


class MlxpSubmitRequest(BaseModel):
    variant: str
    phase: Literal["train", "eval"] = "train"
    train_note: str | None = None
    num_gpus: int = DEFAULT_TRAIN_NUM_GPUS
    global_batch_size: int | None = None
    max_steps: int | None = None
    save_steps: int | None = None
    num_workers: int | None = None
    action_horizon: int | None = Field(default=None, ge=1)
    train_git_commit: str | None = None
    # The k8s node to pin via nodeAffinity. Leave None to fall back to the
    # configured default node. Only used for job_class=dedicated.
    node: str | None = None
    # MLXP scheduling class (metadata.labels."mlxp/job-class"). dedicated pins
    # the node via hostname-In affinity and keeps priority on it; normal and
    # background go to the queue (no node pinning, preemptible + auto-resume).
    job_class: Literal["dedicated", "normal", "background"] = "normal"
    dataset_override: str | list[str] | None = None
    extra_args: list[str] = Field(default_factory=list)
    eval_num_envs_per_gpu: int | None = Field(default=None, ge=1)
    eval_n_episodes: int | None = Field(default=None, ge=1)
    eval_n_runs: int | None = Field(default=None, ge=1)
    eval_sets: list[str] | None = None
    # Eval-only: multitask task-subset selection (task SHORT labels).
    eval_tasks: list[str] | None = None
    eval_overwrite_results: bool = False
    # Eval-only: DexJoCo task (yaml stem / env_name) chosen via the task picker.
    # Falls back to the variant's own DEXJOCO_TASK when omitted.
    dexjoco_task: str | None = None
    checkpoint_path: str | None = None
    # Optional override for the auto-generated display job_name. Validated
    # against the unified regex in submit.resolve_job_name.
    job_name: str | None = None
    output_namespace: str | None = None
    commit_dirty_changes: bool = False


class MlxpSubmitResponse(BaseModel):
    job_id: str             # 6-char k8s Job name, used in /jobs/<cluster>/<id>
    job_name: str           # human-readable {phase}_{variant}_{ts}, same shape as slurm
    apply_stdout: str


async def submit_mlxp(req: MlxpSubmitRequest) -> MlxpSubmitResponse:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    if req.num_gpus not in _GPU_RESOURCES:
        raise ValueError(f"num_gpus must be one of {list(_GPU_RESOURCES)}, got {req.num_gpus}")
    # submit_mlxp resolves several fields (git commit, action horizon, output
    # namespace, clamped eval envs) into the request as it runs. Work on a
    # private copy so the caller's request is never mutated — slurm's submit()
    # never mutates req, and a caller that reuses/retries the request must see
    # its original values.
    req = req.model_copy(deep=True)
    if req.phase == "eval" and not (req.checkpoint_path or "").strip():
        raise ValueError("checkpoint_path is required for MLXP eval")
    # Clamp defensively, matching slurm: resubmit/edit paths recover
    # eval_num_envs_per_gpu from old metadata and must not hard-fail on a stale
    # out-of-range value. Fresh submits are already capped at 1 by the schema.
    from .submit import clamp_eval_num_envs
    req.eval_num_envs_per_gpu = clamp_eval_num_envs(req.eval_num_envs_per_gpu)
    if req.phase != "eval" and any((
        req.eval_num_envs_per_gpu is not None,
        req.eval_n_episodes is not None,
        req.eval_n_runs is not None,
        req.eval_sets is not None,
        req.eval_tasks is not None,
        req.eval_overwrite_results,
        bool(req.checkpoint_path and req.checkpoint_path.strip()),
    )):
        raise ValueError("eval overrides are only valid for phase=eval")
    if req.phase != "train" and any((
        req.global_batch_size is not None,
        req.max_steps is not None,
        req.save_steps is not None,
        req.action_horizon is not None,
    )):
        raise ValueError("train overrides are only valid for phase=train")

    variant = await load_variant(req.variant)
    settings = get_settings()
    cpu, mem = _GPU_RESOURCES[req.num_gpus]
    model = resolve_training_model(variant)
    action_horizon_mode = action_horizon_mode_for_variant(model, variant)
    req.train_git_commit = resolve_train_git_commit_override(
        req.train_git_commit,
        variant.vars,
    )
    if req.phase == "train" and model.family == "n1.6":
        req.action_horizon = resolve_train_action_horizon(
            variant=variant,
            model=model,
            action_horizon_mode=action_horizon_mode,
            requested=req.action_horizon,
        )
    # Resolve train overrides ONCE. The config snapshot, the job-comment
    # metadata, and the executed torchrun command must all see the same values;
    # the body renderer previously re-derived the batch from TRAIN_BATCH_SIZE
    # and could run a different global batch than the snapshot recorded.
    train_settings = None
    if req.phase == "train":
        from .submit import resolve_train_settings
        train_settings = resolve_train_settings(
            variant,
            model.family,
            num_gpus_override=req.num_gpus,
            global_batch_override=req.global_batch_size,
            max_steps_override=req.max_steps,
            save_steps_override=req.save_steps,
            num_workers_override=req.num_workers,
        )
    # job_id is the k8s Job resource name. MLXP's guide requires
    # `<user>-<job-name>`; job_name stays as the display name carried in
    # annotations with the same shape as slurm's job_name.
    from .submit import resolve_job_name, resolve_train_note
    job_name = resolve_job_name(req.job_name, req.phase, req.variant)
    job_id = _mlxp_job_id(settings, job_name)
    train_note = resolve_train_note(req.train_note, variant)
    req.output_namespace = validate_output_namespace(
        req.output_namespace or make_output_namespace(job_name, req.variant)
    )
    repo_path = mlxp_training_repo_path(model)
    submit_git = await prepare_mlxp_training_git(
        repo_path=repo_path,
        repo_label=training_repo_label(model),
        job_name=job_name,
        commit_dirty_changes=req.commit_dirty_changes if req.phase == "train" else False,
        require_clean=(req.phase == "train"),
        requested_commit=req.train_git_commit,
    )

    if req.job_class == "dedicated":
        node = req.node or settings.default_node
        if not (node or "").strip():
            raise ValueError("job_class=dedicated requires a node (request or settings default)")
    else:
        # Queue classes leave placement to the MLXP scheduler — never pin.
        node = ""
    if req.phase == "eval":
        snapshot = _build_eval_snapshot_payload(
            variant=variant,
            req=req,
            job_id=job_id,
            job_name=job_name,
            node=node,
            submit_git=submit_git,
            model=model,
            settings=settings,
            train_note=train_note,
        )
    else:
        snapshot = _build_snapshot_payload(
            variant=variant,
            req=req,
            job_id=job_id,
            job_name=job_name,
            node=node,
            submit_git=submit_git,
            model=model,
            settings=settings,
            train_note=train_note,
            action_horizon_mode=action_horizon_mode,
            train_settings=train_settings,
        )
    await _write_snapshot_to_ddn(snapshot)
    model_output_dir: str | None = None
    if req.phase == "eval":
        body_script = _render_eval_body_script(variant, req, job_name, snapshot, model, repo_path, settings)
    else:
        body_script = _render_body_script(variant, req, job_name, snapshot, model, repo_path, settings, train_settings)
        # Org policy: expose the per-job checkpoint output dir as MODEL_OUTPUT_DIR
        # so the rendered training command consumes the env rather than a spliced
        # literal. Same path the body scripts compute from settings.experiments_dir.
        model_output_dir = paths.checkpoint_dir(
            f"{settings.experiments_dir}/{variant.name}", req.output_namespace
        )
    spec = _render_job_yaml(
        job_id,
        job_name,
        body_script,
        req.num_gpus,
        cpu,
        mem,
        settings.wandb_secret,
        node,
        req.job_class,
        _job_comment(req, variant, snapshot, model, train_settings),
        train_note,
        settings,
        model_output_dir=model_output_dir,
    )
    yaml_text = yaml.safe_dump(spec, sort_keys=False)

    proc = await asyncio.create_subprocess_exec(
        "kubectl", "apply", "-f", "-", "--validate=false", "-n", settings.namespace,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=yaml_text.encode())
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl apply failed: {stderr.decode(errors='replace').strip()}")

    return MlxpSubmitResponse(
        job_id=job_id,
        job_name=job_name,
        apply_stdout=stdout.decode(errors="replace").strip(),
    )


def _build_snapshot_payload(*, variant, req: MlxpSubmitRequest, job_id: str, job_name: str,
                            node: str, submit_git, model: TrainingModel,
                            settings: MlxpSettings, train_note: str,
                            action_horizon_mode: str, train_settings) -> dict:
    train_num_gpus = train_settings.num_gpus
    train_max_steps = train_settings.max_steps
    train_save_steps = train_settings.save_steps
    train_num_workers = train_settings.num_workers
    train_global_batch_size = train_settings.global_batch_size
    suffix = req.output_namespace or f"{snapshot_suffix(job_name)}_{job_id}"
    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    path = paths.config_path(exp_dir, suffix)
    meta_path = paths.meta_path(exp_dir, suffix)
    config_text = render_training_config_snapshot(
        base_config=variant.raw,
        variant=variant.name,
        model=model.family,
        job_name=job_name,
        cluster="mlxp",
        node=node,
        dataset_override=req.dataset_override,
        extra_args=req.extra_args,
        train_num_gpus=train_num_gpus,
        train_global_batch_size=train_global_batch_size,
        train_max_steps=train_max_steps,
        train_save_steps=train_save_steps,
        train_num_workers=train_num_workers,
        train_action_horizon=req.action_horizon,
        train_modality_config=(
            f"modality_{suffix}.py"
            if rewrites_modality_action_horizon(action_horizon_mode) and req.action_horizon is not None
            else None
        ),
        train_git_commit=req.train_git_commit,
        train_note=train_note,
        wandb_project=_wandb_project(),
        git=submit_git,
    )
    meta = snapshot_metadata(
        job_id=job_id,
        job_name=job_name,
        cluster="mlxp",
        variant=variant.name,
        path=path,
        meta_path=meta_path,
        node=node,
        dataset_override=req.dataset_override,
        extra_args=req.extra_args,
        train_num_gpus=train_num_gpus,
        train_global_batch_size=train_global_batch_size,
        train_max_steps=train_max_steps,
        train_save_steps=train_save_steps,
        train_num_workers=train_num_workers,
        train_action_horizon=req.action_horizon,
        train_modality_config=(
            f"{exp_dir}/modality_{suffix}.py"
            if rewrites_modality_action_horizon(action_horizon_mode) and req.action_horizon is not None
            else None
        ),
        train_git_commit=req.train_git_commit,
        train_note=train_note,
        wandb_project=_wandb_project(),
        git=submit_git,
    )
    meta["output_namespace"] = req.output_namespace
    meta["checkpoint_dir"] = paths.checkpoint_dir(exp_dir, req.output_namespace)
    payload = {
        "job_id": job_id,
        "job_name": job_name,
        "output_namespace": req.output_namespace,
        "phase": "train",
        "path": path,
        "meta_path": meta_path,
        "config_text": config_text,
        "meta_text": metadata_json(meta),
        "git_commit": submit_git.commit,
        "git_commit_subject": submit_git.commit_subject,
        "git_branch": submit_git.branch,
        "git_repo_path": submit_git.repo_path,
        "git_repo_label": submit_git.repo_label,
        "git_dirty_at_submit": submit_git.dirty_before,
        "git_committed_dirty": submit_git.committed_dirty,
        "action_horizon_mode": action_horizon_mode,
    }
    if rewrites_modality_action_horizon(action_horizon_mode) and req.action_horizon is not None:
        _, modality_path = resolve_modality_config(variant)
        payload["modality_path"] = f"{exp_dir}/modality_{suffix}.py"
        payload["modality_text"] = rewrite_action_horizon(
            modality_path.read_text(),
            req.action_horizon,
        )
    return payload


def _build_eval_snapshot_payload(*, variant, req: MlxpSubmitRequest, job_id: str, job_name: str,
                                 node: str, submit_git, model: TrainingModel,
                                 settings: MlxpSettings, train_note: str) -> dict:
    from .dexjoco_rollout import rollout_for_variant
    from .eval_harness import harness_for
    from .submit import normalize_eval_sets, normalize_eval_tasks

    eval_sets = normalize_eval_sets(req.eval_sets)
    eval_tasks = normalize_eval_tasks(req.eval_tasks)
    suffix = req.output_namespace or f"{snapshot_suffix(job_name)}_{job_id}"
    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    path = paths.config_path(exp_dir, suffix)
    meta_path = paths.meta_path(exp_dir, suffix)
    checkpoint_path = (req.checkpoint_path or "").strip()
    dexjoco_task = (req.dexjoco_task or "").strip() or None
    rollout = (
        rollout_for_variant(variant)
        if harness_for(variant).name == "dexjoco"
        else None
    )
    config_text = render_eval_config_preview(
        base_config=variant.raw,
        variant=variant.name,
        model=model.family,
        job_name=job_name,
        cluster="mlxp",
        node=node,
        dataset_override=req.dataset_override,
        eval_n_episodes=req.eval_n_episodes,
        eval_n_runs=req.eval_n_runs,
        eval_sets=eval_sets,
        eval_tasks=eval_tasks,
        eval_overwrite_results=req.eval_overwrite_results,
        checkpoint_path=checkpoint_path,
        extra_args=req.extra_args,
        data_dir=settings.datasets_dir,
        eval_num_gpus=req.num_gpus,
        eval_unset_cuda_visible_devices_for_server=1,
        train_git_commit=req.train_git_commit,
        train_note=train_note,
        dexjoco_task=req.dexjoco_task,
    )
    meta = snapshot_metadata(
        job_id=job_id,
        job_name=job_name,
        cluster="mlxp",
        phase="eval",
        variant=variant.name,
        path=path,
        meta_path=meta_path,
        node=node,
        dataset_override=req.dataset_override,
        extra_args=req.extra_args,
        train_git_commit=req.train_git_commit,
        train_note=train_note,
        wandb_project=_wandb_project(),
        eval_rollout=rollout.metadata() if rollout else None,
        git=submit_git,
    )
    meta["output_namespace"] = req.output_namespace
    meta["eval_dir"] = paths.eval_dir(exp_dir, req.output_namespace)
    meta["results_path"] = paths.results_path(meta["eval_dir"])
    meta["job_log_dir"] = paths.job_log_dir(exp_dir, req.output_namespace)
    meta["eval"] = {
        "checkpoint_path": checkpoint_path,
        "num_envs_per_gpu": req.eval_num_envs_per_gpu,
        "n_episodes": req.eval_n_episodes,
        "n_runs": req.eval_n_runs,
        "eval_sets": eval_sets,
        "eval_tasks": eval_tasks,
        "overwrite_results": req.eval_overwrite_results,
        "unset_cuda_visible_devices_for_server": 1,
        "dexjoco_task": dexjoco_task,
        "rollout": rollout.metadata() if rollout else None,
    }
    return {
        "job_id": job_id,
        "job_name": job_name,
        "output_namespace": req.output_namespace,
        "phase": "eval",
        "path": path,
        "meta_path": meta_path,
        "config_text": config_text,
        "meta_text": metadata_json(meta),
        "eval_sets": eval_sets,
        "eval_tasks": eval_tasks,
        "dexjoco_task": dexjoco_task,
        "checkpoint_path": checkpoint_path,
        "git_commit": submit_git.commit,
        "git_commit_subject": submit_git.commit_subject,
        "git_branch": submit_git.branch,
        "git_repo_path": submit_git.repo_path,
        "git_repo_label": submit_git.repo_label,
        "git_dirty_at_submit": submit_git.dirty_before,
        "git_committed_dirty": submit_git.committed_dirty,
    }


def _shell_words(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _mlxp_worktree_path(snapshot: dict) -> str:
    # Both snapshot builders always set job_id (submit_mlxp passes it in).
    settings = get_settings()
    return f"{settings.experiments_dir}/.worktrees/{snapshot['job_id']}"


def _repo_checkout_preamble(repo_path: str, snapshot: dict) -> str:
    """Pin the runtime checkout to the commit captured at submit time.

    MLXP jobs can sit pending for hours. Running directly from the shared DDN
    checkout means a queued job may execute code that changed after submit.
    A detached worktree keeps the runtime code equal to the recorded commit.
    """
    commit = snapshot.get("git_commit")
    if not isinstance(commit, str) or not commit:
        repo = shlex.quote(repo_path)
        return f"REPO_SRC={repo}\ncd \"$REPO_SRC\"\n"

    worktree_path = _mlxp_worktree_path(snapshot)
    repo = shlex.quote(repo_path)
    worktree = shlex.quote(worktree_path)
    commit_q = shlex.quote(commit)
    return f"""\
REPO_SRC={repo}
REPO_WORKTREE={worktree}
REPO_COMMIT={commit_q}
mkdir -p "$(dirname "$REPO_WORKTREE")"
git -c safe.directory="$REPO_SRC" -C "$REPO_SRC" worktree prune || true
if [ ! -e "$REPO_WORKTREE/.git" ]; then
    if [ -e "$REPO_WORKTREE" ]; then
        echo "[mlxp] refusing to use non-git worktree path: $REPO_WORKTREE" >&2
        exit 1
    fi
    for attempt in 1 2 3 4 5; do
        if git -c safe.directory="$REPO_SRC" -C "$REPO_SRC" worktree add --detach "$REPO_WORKTREE" "$REPO_COMMIT"; then
            break
        fi
        rc=$?
        if [ "$attempt" = "5" ]; then
            exit "$rc"
        fi
        sleep $((attempt * 2))
    done
fi
cd "$REPO_WORKTREE"
CURRENT_COMMIT="$(git -c safe.directory="$REPO_WORKTREE" rev-parse HEAD)"
if [ "$CURRENT_COMMIT" != "$REPO_COMMIT" ]; then
    echo "[mlxp] worktree commit mismatch: expected $REPO_COMMIT got $CURRENT_COMMIT" >&2
    exit 1
fi
echo "[mlxp] running submitted code commit $CURRENT_COMMIT from $REPO_WORKTREE"
"""


def _repo_runtime_preamble(repo_path: str, snapshot: dict) -> str:
    return f"""\
{_repo_checkout_preamble(repo_path, snapshot)}
if [ -d "$REPO_SRC/.venv" ] && [ ! -e .venv ]; then
    ln -s "$REPO_SRC/.venv" .venv
fi
export PYTHONPATH="$PWD${{PYTHONPATH:+:$PYTHONPATH}}"
"""


def _snapshot_preamble(snapshot: dict) -> str:
    config_text = snapshot["config_text"]
    meta_text = snapshot["meta_text"]
    path = snapshot["path"]
    meta_path = snapshot["meta_path"]
    modality_path = snapshot.get("modality_path")
    modality_text = snapshot.get("modality_text")
    modality_block = ""
    if isinstance(modality_path, str) and isinstance(modality_text, str):
        modality_text = ensure_trailing_newline(modality_text)
        modality_block = f"""
mkdir -p {shlex.quote(modality_path.rsplit('/', 1)[0])}
cat > {shlex.quote(modality_path)} <<'TRAIN_EVAL_MODALITY_SNAPSHOT'
{modality_text}TRAIN_EVAL_MODALITY_SNAPSHOT
"""
    return f"""\
mkdir -p {shlex.quote(path.rsplit('/', 1)[0])}
cat > {shlex.quote(path)} <<'TRAIN_EVAL_CONFIG_SNAPSHOT'
{config_text}TRAIN_EVAL_CONFIG_SNAPSHOT
cat > {shlex.quote(meta_path)} <<'TRAIN_EVAL_CONFIG_META'
{meta_text}TRAIN_EVAL_CONFIG_META
{modality_block}
"""


def _snapshot_tar(snapshot: dict) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tf:
        for path_key, text_key in (
            ("path", "config_text"),
            ("meta_path", "meta_text"),
            ("modality_path", "modality_text"),
        ):
            if path_key not in snapshot or text_key not in snapshot:
                continue
            data = snapshot[text_key].encode()
            info = tarfile.TarInfo(snapshot[path_key].lstrip("/"))
            info.size = len(data)
            info.mode = 0o644
            info.mtime = int(time.time())
            tf.addfile(info, io.BytesIO(data))
    return payload.getvalue()


async def _write_snapshot_to_ddn(snapshot: dict) -> None:
    from .mlxp_data_pod import ensure_listing_pod

    pod = await ensure_listing_pod()
    settings = get_settings()
    snapshot_dir = snapshot["path"].rsplit("/", 1)[0]
    proc = await asyncio.create_subprocess_exec(
        "kubectl",
        "exec",
        "-i",
        "-n",
        settings.namespace,
        pod,
        "--",
        "bash",
        "-lc",
        f"mkdir -p {shlex.quote(snapshot_dir)} && tar -xf - -C /",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=_snapshot_tar(snapshot)),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("writing MLXP config snapshot timed out")
    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        out = stdout.decode(errors="replace").strip()
        raise RuntimeError(f"writing MLXP config snapshot failed: {err or out}")


def _render_body_script(
    variant,
    req: MlxpSubmitRequest,
    job_name: str,
    snapshot: dict,
    model: TrainingModel,
    repo_path: str,
    settings: MlxpSettings,
    train_settings,
) -> str:
    """Render the inline bash the container runs.

    Resolves the variant's dataset list, then dispatches to the right gr00t
    entrypoint based on MODEL_VERSION:
      - n1.5 → gr00t_finetune.py with /tmp/data_config.yaml
      - n1.6 → launch_finetune.py with --dataset-path + --modality-config-path

    `job_name` flows into WANDB_RUN_ID and --experiment-name / --run_name.
    """
    family = model.family

    # ── Resolve dataset name list (model-agnostic) ──
    names: list[str] = []
    override = req.dataset_override
    if override is not None:
        if isinstance(override, list):
            # Either "name" or "name|cfg|weight" entries.
            names = [e.split("|", 1)[0] for e in override]
            override_full = override  # preserve N1.5 cfg/weight if present
        else:
            names = [override]
            override_full = [override]
    else:
        if variant.arrays.get("TRAIN_DATASET_NAMES"):
            names = list(variant.arrays["TRAIN_DATASET_NAMES"])
            override_full = None
        elif variant.arrays.get("DATASETS"):
            names = [e.split("|", 1)[0] for e in variant.arrays["DATASETS"]]
            override_full = None
        elif variant.vars.get("DATASET_NAME"):
            names = [variant.vars["DATASET_NAME"]]
            override_full = None
        else:
            raise ValueError(
                f"variant {variant.name} has no DATASET_NAME / DATASETS / TRAIN_DATASET_NAMES"
            )

    # Values come from the SAME TrainSettings the snapshot recorded, so the
    # executed command can never diverge from the recorded config (the batch
    # was previously re-derived from TRAIN_BATCH_SIZE, ignoring the variant's
    # TRAIN_GLOBAL_BATCH_SIZE).
    max_steps = str(train_settings.max_steps)
    save_steps = str(train_settings.save_steps)
    num_workers = str(train_settings.num_workers)
    global_batch = int(train_settings.global_batch_size)
    train_extra = _shell_words(variant.arrays.get("TRAIN_EXTRA_ARGS") or [])
    user_extra = _shell_words(req.extra_args)

    # submit_mlxp fills req.output_namespace before any render runs.
    output_namespace = req.output_namespace
    ckpt_dir = paths.checkpoint_dir(f"{settings.experiments_dir}/{variant.name}", output_namespace)
    run_log_dir = f"{ckpt_dir}/logs"
    wandb_project = shlex.quote(_wandb_project())

    if family == "n1.6":
        return _render_body_n16(
            variant=variant, req=req, job_name=job_name, names=names,
            max_steps=max_steps, save_steps=save_steps, num_workers=num_workers,
            global_batch=global_batch,
            train_extra=train_extra, user_extra=user_extra, ckpt_dir=ckpt_dir,
            snapshot=snapshot, model=model, repo_path=repo_path, settings=settings,
        )
    if family == "n1.7":
        return _render_body_n17(
            variant=variant, req=req, job_name=job_name, names=names,
            max_steps=max_steps, save_steps=save_steps, num_workers=num_workers,
            global_batch=global_batch,
            train_extra=train_extra, user_extra=user_extra, ckpt_dir=ckpt_dir,
            snapshot=snapshot, model=model, repo_path=repo_path, settings=settings,
        )
    if family == "gam":
        return _render_body_gam(
            variant=variant, req=req, job_name=job_name,
            max_steps=max_steps, save_steps=save_steps, num_workers=num_workers,
            global_batch=global_batch, ckpt_dir=ckpt_dir,
            snapshot=snapshot, model=model, repo_path=repo_path, settings=settings,
        )
    if family != "n1.5":
        raise ValueError(f"unsupported MLXP model family: {family}")

    # n1.5 takes a per-GPU batch; divisibility was validated by
    # resolve_train_settings at submit time.
    batch_size = str(global_batch // req.num_gpus)

    # ── N1.5: build the data_config.yaml rows ──
    if override_full is not None and isinstance(override, list) and any("|" in e for e in override):
        datasets_decl = override_full
    elif override_full is not None and isinstance(override, str):
        cfg = variant.vars.get("DATA_CONFIG", DEFAULT_DATA_CONFIG)
        datasets_decl = [f"{override}|{cfg}|1.0"]
    elif variant.arrays.get("DATASETS"):
        datasets_decl = variant.arrays["DATASETS"]
    else:
        cfg = variant.vars.get("DATA_CONFIG", DEFAULT_DATA_CONFIG)
        datasets_decl = [f"{names[0]}|{cfg}|1.0"]

    data_config_yaml = _n15_data_config_yaml(
        variant,
        datasets_decl,
        use_file=override is None,
        settings=settings,
    )

    # No leading indentation — keeps the embedded heredoc YAML well-formed.
    return f"""\
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT={wandb_project}
# Pin the wandb run-id to the k8s Job name so requeues continue the same
# run (HF Trainer otherwise spawns a fresh run on each container start).
export WANDB_RUN_ID="{job_name}"
export WANDB_RESUME=allow
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
{_hf_cache_exports(settings)}

{_repo_runtime_preamble(repo_path, snapshot)}
source .venv/bin/activate

{_snapshot_preamble(snapshot)}
mkdir -p {ckpt_dir}
RUN_LOG_DIR={shlex.quote(run_log_dir)}
mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/training.log") 2>&1
echo "[mlxp] run namespace: {output_namespace}"

# Render data_config.yaml from variant config.
cat > /tmp/data_config.yaml <<'YAML_EOF'
{data_config_yaml}
YAML_EOF

# Auto-resume from latest checkpoint if any.
RESUME_FLAG=""
if compgen -G "{ckpt_dir}/checkpoint-*" > /dev/null; then
    echo "[mlxp] existing checkpoint detected — will resume"
    RESUME_FLAG="--resume"
fi

torchrun --nproc_per_node={req.num_gpus} scripts/gr00t_finetune.py \\
    --num-gpus {req.num_gpus} \\
    --batch-size {batch_size} \\
    --learning_rate 1e-4 \\
    --output-dir "$MODEL_OUTPUT_DIR" \\
    --data-config /tmp/data_config.yaml \\
    --max-steps {max_steps} \\
    --save-steps {save_steps} \\
    --dataloader_num_workers {num_workers} \\
    --dataloader-prefetch-factor 10 \\
    --video-backend torchcodec \\
    --report-to wandb \\
    --pin_memory \\
    --run_name "{variant.name}" \\
    --seed 42 \\
    $RESUME_FLAG {train_extra} {user_extra}

{_strip_resume_state_block(ckpt_dir, max_steps)}
"""


def _n15_data_config_yaml(
    variant,
    datasets_decl: list[str],
    *,
    use_file: bool,
    settings: MlxpSettings,
) -> str:
    rel = (variant.vars.get("TRAIN_DATA_CONFIG") or "data_config.yaml").strip()
    path = EXPERIMENTS_DIR / variant.name / rel
    if use_file and is_safe_relpath(rel, {".yaml", ".yml"}) and path.is_file():
        return (
            path.read_text()
            .replace("${DATA_DIR}", settings.datasets_dir)
            .replace("$DATA_DIR", settings.datasets_dir)
        )

    yaml_rows = []
    for entry in datasets_decl:
        parts = entry.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"bad DATASETS entry (need name|cfg|weight): {entry!r}")
        name, cfg, weight = parts
        yaml_rows.append(
            f"    - path: {settings.datasets_dir}/{name}\n"
            f"      embodiment_tag: new_embodiment\n"
            f"      data_config: {cfg}\n"
            f"      weight: {weight}"
        )
    return "train:\n  datasets:\n" + "\n".join(yaml_rows)


def _render_body_n16(*, variant, req: MlxpSubmitRequest, job_name: str,
                     names: list[str], max_steps: str, save_steps: str,
                     num_workers: str,
                     global_batch: int, train_extra: str, user_extra: str,
                     ckpt_dir: str, snapshot: dict, model: TrainingModel, repo_path: str,
                     settings: MlxpSettings) -> str:
    """Body script for GR00T N1.6 (launch_finetune.py).

    Unlike N1.5, N1.6 takes --dataset-path (multiple) + --modality-config-path
    (a Python file). We inline the modality config from the local variant
    directory so MLXP doesn't need a rsync step.
    """
    _, modality_path = resolve_modality_config(variant)
    modality_text = snapshot.get("modality_text")
    if not isinstance(modality_text, str):
        modality_text = modality_path.read_text()
    modality_text = ensure_trailing_newline(modality_text)

    dataset_paths_arg = " \\\n        ".join(
        f"{settings.datasets_dir}/{n}" for n in names
    )
    # Optional per-dataset embodiment tags (mixed-embodiment training). Build a
    # name->tag map from the parallel config arrays and resolve tags for the
    # datasets actually being trained (which may be a submit-time subset via
    # dataset_override). Emitted only when every dataset resolves to a tag, so a
    # missing/unmapped dataset fails loudly rather than silently mis-tagging.
    cfg_names = variant.arrays.get("TRAIN_DATASET_NAMES")
    cfg_tags = variant.arrays.get("TRAIN_DATASET_EMBODIMENT_TAGS")
    embodiment_tags_line = ""
    if cfg_names and cfg_tags and len(cfg_names) == len(cfg_tags):
        tag_by_name = dict(zip(cfg_names, cfg_tags))
        if all(n in tag_by_name for n in names):
            resolved = " ".join(tag_by_name[n] for n in names)
            embodiment_tags_line = f"--embodiment-tags {resolved} \\\n    "
    run_log_dir = f"{ckpt_dir}/logs"
    wandb_project = shlex.quote(_wandb_project())
    uv_bin_dir = shlex.quote(f"{settings.ddn_user_home}/.local/bin")
    output_namespace = ckpt_dir.rstrip("/").rsplit("/", 1)[-1]
    action_horizon_mode = str(snapshot.get("action_horizon_mode") or model.action_horizon_mode)
    action_horizon_arg = (
        f" --action-horizon {req.action_horizon}"
        if passes_action_horizon_cli(action_horizon_mode) and req.action_horizon is not None
        else ""
    )

    return f"""\
set -euo pipefail
export PATH="{uv_bin_dir}:$HOME/.local/bin:$PATH"
export WANDB_PROJECT={wandb_project}
export WANDB_RUN_ID="{job_name}"
export WANDB_RESUME=allow
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMNI_KIT_ACCEPT_EULA=Y
{_hf_cache_exports(settings)}

{_uv_bootstrap_block(settings)}

{_repo_runtime_preamble(repo_path, snapshot)}
UV_RUN_ARGS=""
if [ -e .venv ]; then
    UV_RUN_ARGS="--no-sync"
fi

{_snapshot_preamble(snapshot)}
mkdir -p {ckpt_dir}
RUN_LOG_DIR={shlex.quote(run_log_dir)}
mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/training.log") 2>&1

cat > /tmp/modality_config.py <<'PY_EOF'
{modality_text}
PY_EOF

RESUME_FLAG=""
if compgen -G "{ckpt_dir}/checkpoint-*" > /dev/null; then
    echo "[mlxp] existing checkpoint detected — will resume"
    RESUME_FLAG="--resume"
fi

uv run $UV_RUN_ARGS torchrun --nproc_per_node={req.num_gpus} gr00t/experiment/launch_finetune.py \\
    --base-model-path nvidia/GR00T-N1.6-3B \\
    --dataset-path \\
        {dataset_paths_arg} \\
    --embodiment-tag NEW_EMBODIMENT \\
    {embodiment_tags_line}--modality-config-path /tmp/modality_config.py \\
    --num-gpus {req.num_gpus} \\
    --output-dir "$(dirname "$MODEL_OUTPUT_DIR")" \\
    --global-batch-size {global_batch} \\
    --learning-rate 1e-4 \\
    --max-steps {max_steps} \\
    --save-steps {save_steps} \\
    --save-total-limit 5 \\
    --dataloader-num-workers {num_workers} \\
    --experiment-name "{output_namespace}" \\
    --use-wandb \\
    --wandb-project {wandb_project} \\
    --color-jitter-params brightness 0.2 contrast 0.2 saturation 0.2 hue 0.1 \\
    $RESUME_FLAG{action_horizon_arg} {train_extra} {user_extra}

{_strip_resume_state_block(ckpt_dir, max_steps)}
"""


def _render_body_n17(*, variant, req: MlxpSubmitRequest, job_name: str,
                     names: list[str], max_steps: str, save_steps: str,
                     num_workers: str,
                     global_batch: int, train_extra: str, user_extra: str,
                     ckpt_dir: str, snapshot: dict, model: TrainingModel, repo_path: str,
                     settings: MlxpSettings) -> str:
    """Body script for GR00T N1.7.

    Two launchers share this renderer:
      - single-dataset (default): gr00t/experiment/launch_finetune.py with a
        singular --dataset-path + an inlined --modality-config-path (.py), no
        --action-horizon (that flag does not exist on the single launcher);
      - multi-dataset (variant sets TRAIN_DATA_YAML): launch_finetune_multi.py
        with --data-yaml (inlined, ${DATA_DIR} expanded) plus an optional
        --action-horizon. Its data-YAML rows reference modality `.py` files that
        must live under the repo's configs/ dir.

    The base model is always passed explicitly (nvidia/GR00T-N1.7-3B): the single
    launcher requires it and the multi launcher would otherwise default to 2B.
    """
    run_log_dir = f"{ckpt_dir}/logs"
    wandb_project = shlex.quote(_wandb_project())
    uv_bin_dir = shlex.quote(f"{settings.ddn_user_home}/.local/bin")
    output_namespace = ckpt_dir.rstrip("/").rsplit("/", 1)[-1]

    data_yaml_rel = (variant.vars.get("TRAIN_DATA_YAML") or "").strip()
    if data_yaml_rel:
        # Multi-dataset: inline the variant's data YAML (expand ${DATA_DIR}).
        if not is_safe_relpath(data_yaml_rel, {".yaml", ".yml"}):
            raise ValueError(
                f"variant {variant.name}: TRAIN_DATA_YAML must be a .yaml/.yml file, got {data_yaml_rel!r}"
            )
        data_yaml_path = EXPERIMENTS_DIR / variant.name / data_yaml_rel
        if not data_yaml_path.is_file():
            raise FileNotFoundError(f"data yaml not found: {data_yaml_path}")
        data_yaml_text = ensure_trailing_newline(
            data_yaml_path.read_text()
            .replace("${DATA_DIR}", settings.datasets_dir)
            .replace("$DATA_DIR", settings.datasets_dir)
        )
        action_horizon = (variant.vars.get("TRAIN_ACTION_HORIZON") or "").strip()
        action_horizon_arg = f" --action-horizon {action_horizon}" if action_horizon else ""
        stage_block = f"""cat > /tmp/data_config.yaml <<'YAML_EOF'
{data_yaml_text}YAML_EOF"""
        launch_block = f"""uv run $UV_RUN_ARGS torchrun --nproc_per_node={req.num_gpus} gr00t/experiment/launch_finetune_multi.py \\
    --base-model-path nvidia/GR00T-N1.7-3B \\
    --data-yaml /tmp/data_config.yaml \\
    --num-gpus {req.num_gpus} \\
    --output-dir "$(dirname "$MODEL_OUTPUT_DIR")" \\
    --global-batch-size {global_batch} \\
    --learning-rate 1e-4 \\
    --max-steps {max_steps} \\
    --save-steps {save_steps} \\
    --save-total-limit 5 \\
    --dataloader-num-workers {num_workers} \\
    --experiment-name "{output_namespace}" \\
    --use-wandb \\
    --wandb-project {wandb_project} \\
    --color-jitter-params brightness 0.2 contrast 0.2 saturation 0.2 hue 0.1 \\
    $RESUME_FLAG{action_horizon_arg} {train_extra} {user_extra}"""
    else:
        # Single-dataset: inline the modality .py and pass one --dataset-path.
        _, modality_path = resolve_modality_config(variant)
        modality_text = snapshot.get("modality_text")
        if not isinstance(modality_text, str):
            modality_text = modality_path.read_text()
        modality_text = ensure_trailing_newline(modality_text)
        dataset_path_arg = f"{settings.datasets_dir}/{names[0]}"
        stage_block = f"""cat > /tmp/modality_config.py <<'PY_EOF'
{modality_text}PY_EOF"""
        launch_block = f"""uv run $UV_RUN_ARGS torchrun --nproc_per_node={req.num_gpus} gr00t/experiment/launch_finetune.py \\
    --base-model-path nvidia/GR00T-N1.7-3B \\
    --dataset-path {dataset_path_arg} \\
    --embodiment-tag NEW_EMBODIMENT \\
    --modality-config-path /tmp/modality_config.py \\
    --num-gpus {req.num_gpus} \\
    --output-dir "$(dirname "$MODEL_OUTPUT_DIR")" \\
    --global-batch-size {global_batch} \\
    --learning-rate 1e-4 \\
    --max-steps {max_steps} \\
    --save-steps {save_steps} \\
    --save-total-limit 5 \\
    --dataloader-num-workers {num_workers} \\
    --experiment-name "{output_namespace}" \\
    --use-wandb \\
    --wandb-project {wandb_project} \\
    --color-jitter-params brightness 0.2 contrast 0.2 saturation 0.2 hue 0.1 \\
    $RESUME_FLAG {train_extra} {user_extra}"""

    return f"""\
set -euo pipefail
export PATH="{uv_bin_dir}:$HOME/.local/bin:$PATH"
export WANDB_PROJECT={wandb_project}
export WANDB_RUN_ID="{job_name}"
export WANDB_RESUME=allow
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMNI_KIT_ACCEPT_EULA=Y
{_hf_cache_exports(settings)}

{_uv_bootstrap_block(settings)}

{_repo_runtime_preamble(repo_path, snapshot)}
UV_RUN_ARGS=""
if [ -e .venv ]; then
    UV_RUN_ARGS="--no-sync"
fi

{_snapshot_preamble(snapshot)}
mkdir -p {ckpt_dir}
RUN_LOG_DIR={shlex.quote(run_log_dir)}
mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/training.log") 2>&1

{stage_block}

RESUME_FLAG=""
if compgen -G "{ckpt_dir}/checkpoint-*" > /dev/null; then
    echo "[mlxp] existing checkpoint detected — will resume"
    RESUME_FLAG="--resume"
fi

{launch_block}

{_strip_resume_state_block(ckpt_dir, max_steps)}
"""


def _render_body_gam(*, variant, req: MlxpSubmitRequest, job_name: str,
                     max_steps: str, save_steps: str, num_workers: str,
                     global_batch: int, ckpt_dir: str, snapshot: dict,
                     model: TrainingModel, repo_path: str,
                     settings: MlxpSettings) -> str:
    """Body script for the GAM family (slurm-only lib/train_body_gam.sh port).

    GAM does not use torchrun / launch_finetune.py. This reproduces
    lib/train_body_gam.sh in the MLXP pod: after the repo checkout preamble
    (cwd = pinned worktree, .venv symlinked, PYTHONPATH set) it stages the
    variant's gam_config.yaml into the pod and hands the launch to the fork's
    wrapper `dexjoco/train_dexjoco.sh`, driven entirely by GAM_* env vars. The
    wrapper writes a single-file checkpoint-final.pt into GAM_RESULTS_DIR; that
    file is the completion/skip marker. gr00t-only knobs (action horizon,
    TRAIN_EXTRA_ARGS, user extra_args) do not apply and are ignored.
    """
    # The variant's second file is the GAM OmegaConf config; inline it into the
    # pod the same way n1.6 inlines its modality .py (no rsync step on MLXP).
    second_rel = (variant.vars.get("TRAIN_MODALITY_CONFIG") or "gam_config.yaml").strip()
    if not is_safe_relpath(second_rel, {".yaml", ".yml"}):
        raise ValueError(
            f"variant {variant.name}: GAM TRAIN_MODALITY_CONFIG must be a .yaml/.yml file, got {second_rel!r}"
        )
    second_path = EXPERIMENTS_DIR / variant.name / second_rel
    if not second_path.is_file():
        raise FileNotFoundError(f"GAM config not found: {second_path}")
    gam_config_text = ensure_trailing_newline(second_path.read_text())

    run_log_dir = f"{ckpt_dir}/logs"
    wandb_project = shlex.quote(_wandb_project())
    output_namespace = ckpt_dir.rstrip("/").rsplit("/", 1)[-1]
    datasets_dir = settings.datasets_dir

    return f"""\
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT={wandb_project}
export WANDB_RUN_ID="{job_name}"
export WANDB_RESUME=allow
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMNI_KIT_ACCEPT_EULA=Y
{_hf_cache_exports(settings)}

{_repo_runtime_preamble(repo_path, snapshot)}

{_snapshot_preamble(snapshot)}
mkdir -p {ckpt_dir}
RUN_LOG_DIR={shlex.quote(run_log_dir)}
mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/training.log") 2>&1
echo "[mlxp] run namespace: {output_namespace}"

# checkpoint-final.pt is the wrapper's completion marker (see the contract).
if [ -f "{ckpt_dir}/checkpoint-final.pt" ]; then
    echo "[mlxp] {ckpt_dir}/checkpoint-final.pt exists — training already complete; skipping."
    exit 0
fi

# Stage the GAM training config (variant second file) into the pod.
cat > /tmp/gam_config.yaml <<'GAM_CONFIG_EOF'
{gam_config_text}GAM_CONFIG_EOF

# DA3 backbone + GAM init checkpoint are untracked assets that live only in the
# main checkout, not the per-job worktree the wrapper defaults to. Point them at
# the main repo so stage_1 finds checkpoints/track4world_da3.pth and training
# starts from the pretrained init.
export DA3_ROOT={shlex.quote(repo_path)}
export GAM_INIT_CKPT={shlex.quote(f"{repo_path}/checkpoints/pretrained-gam.pt")}

# GAM_WANDB_RUN_ID == the display job name: the wrapper enables --wandb with
# id==job_name (project pinned to dexjoco) so the backend resolves the run link.
GAM_CONFIG_YAML=/tmp/gam_config.yaml \\
GAM_DATA_ROOT={shlex.quote(datasets_dir)} \\
GAM_RESULTS_DIR="$MODEL_OUTPUT_DIR" \\
GAM_NUM_GPUS={req.num_gpus} \\
GAM_GLOBAL_BATCH_SIZE={global_batch} \\
GAM_MAX_STEPS={max_steps} \\
GAM_SAVE_STEPS={save_steps} \\
GAM_NUM_WORKERS={num_workers} \\
GAM_WANDB_RUN_ID={shlex.quote(job_name)} \\
    bash dexjoco/train_dexjoco.sh

if [ ! -f "{ckpt_dir}/checkpoint-final.pt" ]; then
    echo "[mlxp] ERROR: GAM wrapper exited 0 but {ckpt_dir}/checkpoint-final.pt is missing" >&2
    exit 1
fi
echo "[mlxp] training complete: {ckpt_dir}/checkpoint-final.pt"
"""


def _mlxp_dexjoco_lib_block(runtime_root: str) -> str:
    """Stage lib/dexjoco/ (the policy-server adapter + helpers) into the runtime.

    eval_body_dexjoco.sh runs ``$REPO_ROOT/lib/dexjoco/gr00t_dexjoco_server.py``;
    the MLXP pod has no rsync step, so inline every ``.py`` under lib/dexjoco/
    (mirrors how the Isaac path inlines isaac_server_runner.py).
    """
    dexjoco_src = LIB_DIR / "dexjoco"
    if not dexjoco_src.is_dir():
        raise FileNotFoundError(f"lib/dexjoco not found: {dexjoco_src}")
    py_files = sorted(dexjoco_src.glob("*.py"))
    if not any(p.name == "gr00t_dexjoco_server.py" for p in py_files):
        raise FileNotFoundError(f"gr00t_dexjoco_server.py not found under {dexjoco_src}")
    dest_dir = f"{runtime_root}/lib/dexjoco"
    blocks = [f"mkdir -p {shlex.quote(dest_dir)}"]
    for idx, p in enumerate(py_files):
        text = ensure_trailing_newline(p.read_text())
        marker = f"TEW_DEXJOCO_LIB_{idx}_EOF"
        blocks.append(
            f"cat > {shlex.quote(f'{dest_dir}/{p.name}')} <<'{marker}'\n{text}{marker}"
        )
    return "\n".join(blocks) + "\n"


def _render_eval_body_script(
    variant,
    req: MlxpSubmitRequest,
    job_name: str,
    snapshot: dict,
    model: TrainingModel,
    repo_path: str,
    settings: MlxpSettings,
) -> str:
    """Render MLXP eval by staging the same eval body script used on Slurm.

    Two harnesses are supported, differing only in what extra runtime files +
    cluster-env vars each needs:
      - Isaac (eval_body.sh): stages isaac_server_runner.py and links the
        DDN-stored ALLEX assets into the eval image.
      - DexJoCo (eval_body_dexjoco.sh): stages lib/dexjoco/ and points the body
        at the DDN-resident dexjoco repo + micromamba envs (see MlxpSettings).
    """
    is_dexjoco = model.eval_body_script == "eval_body_dexjoco.sh"
    eval_body_path = LIB_DIR / model.eval_body_script
    common_path = LIB_DIR / "_common.sh"
    if not eval_body_path.is_file():
        raise FileNotFoundError(f"eval body script not found: {eval_body_path}")
    common_text = ensure_trailing_newline(common_path.read_text())
    eval_body_text = ensure_trailing_newline(eval_body_path.read_text())

    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    output_namespace = str(snapshot["output_namespace"])
    eval_dir = paths.eval_dir(exp_dir, output_namespace)
    results_path = paths.results_path(eval_dir)
    runtime_root = f"{settings.experiments_dir}/.runtime/{snapshot['job_id']}"
    config_path = snapshot["path"]
    modality_block = ""
    if variant.vars.get("TRAIN_MODALITY_CONFIG"):
        modality_rel, modality_path = resolve_modality_config(variant)
        modality_target = f"{exp_dir}/{modality_rel}"
        modality_text = ensure_trailing_newline(modality_path.read_text())
        modality_block = f"""
mkdir -p {shlex.quote(modality_target.rsplit('/', 1)[0])}
cat > {shlex.quote(modality_target)} <<'TEW_MODALITY_EOF'
{modality_text}TEW_MODALITY_EOF
"""

    env_lines = [
        "export CLUSTER=mlxp",
        "export PARTITION=mlxp",
        f"export REPO_ROOT={shlex.quote(runtime_root)}",
        f"export GROOT_DIR={shlex.quote(repo_path)}",
        f"export GROOT_N16_DIR={shlex.quote(repo_path)}",
        f"export PHYSIXEL_DIR={shlex.quote(repo_path)}",
        f"export TRAIN_REPO_DIR={shlex.quote(repo_path)}",
        f"export DATA_DIR={shlex.quote(settings.datasets_dir)}",
        f"export LOG_DIR={shlex.quote(f'{exp_dir}/logs')}",
    ]
    if is_dexjoco:
        # DexJoCo body sources these from clusters/mlxp.env (same names as the
        # kakao cluster.env). The MuJoCo client runs in the `dexjoco` micromamba
        # env; the pi0.5 baseline server runs in `openpi`.
        env_lines += [
            f"export DEXJOCO_DIR={shlex.quote(settings.dexjoco_dir)}",
            f"export MICROMAMBA_BIN={shlex.quote(settings.micromamba_bin)}",
            f"export MAMBA_ROOT_PREFIX={shlex.quote(settings.mamba_root_prefix)}",
            f"export DEXJOCO_EVAL_ENV={shlex.quote(settings.dexjoco_eval_env)}",
            f"export DEXJOCO_OPENPI_ENV={shlex.quote(settings.dexjoco_openpi_env)}",
        ]
    else:
        env_lines.append(f"export ISAAC_DIR={shlex.quote(settings.isaac_dir)}")
    cluster_env = "\n".join(env_lines) + "\n"

    eval_exports = [
        f"export REPO_ROOT={shlex.quote(runtime_root)}",
        "export CLUSTER=mlxp",
        f"export VARIANT={shlex.quote(variant.name)}",
        f"export SLURM_JOB_ID={shlex.quote(snapshot['job_id'])}",
        f"export SLURM_JOB_NAME={shlex.quote(job_name)}",
        f"export SUBMIT_PARTITION={shlex.quote(req.node or settings.default_node)}",
        f"export SUBMIT_EXP_DIR={shlex.quote(exp_dir)}",
        f"export SUBMIT_OUTPUT_NAMESPACE={shlex.quote(output_namespace)}",
        f"export SUBMIT_EVAL_DIR={shlex.quote(eval_dir)}",
        f"export SUBMIT_RESULTS_PATH={shlex.quote(results_path)}",
        f"export SUBMIT_CONFIG_FILE={shlex.quote(config_path)}",
        f"export SUBMIT_DATA_DIR={shlex.quote(settings.datasets_dir)}",
        f"export SUBMIT_TRAIN_NUM_GPUS={req.num_gpus}",
        "export SUBMIT_EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER=1",
        f"export EVAL_CHECKPOINT={shlex.quote((req.checkpoint_path or '').strip())}",
    ]
    if req.eval_num_envs_per_gpu is not None:
        eval_exports.append(f"export SUBMIT_EVAL_NUM_ENVS_PER_GPU={req.eval_num_envs_per_gpu}")
    if req.eval_n_episodes is not None:
        eval_exports.append(f"export SUBMIT_EVAL_N_EPISODES={req.eval_n_episodes}")
    if req.eval_n_runs is not None:
        eval_exports.append(f"export SUBMIT_EVAL_N_RUNS={req.eval_n_runs}")
    if snapshot.get("eval_sets"):
        eval_exports.append(f"export SUBMIT_EVAL_SETS={shlex.quote(' '.join(snapshot['eval_sets']))}")
    if snapshot.get("eval_tasks"):
        eval_exports.append(f"export SUBMIT_EVAL_TASKS={shlex.quote(' '.join(snapshot['eval_tasks']))}")
    if req.eval_overwrite_results:
        eval_exports.append("export SUBMIT_EVAL_OVERWRITE_RESULTS=1")

    # Harness-specific staging + post-stage env, spliced into the heredoc below.
    if is_dexjoco:
        stage_block = _mlxp_dexjoco_lib_block(runtime_root) + modality_block
    else:
        isaac_runner_path = LIB_DIR / "isaac_server_runner.py"
        if not isaac_runner_path.is_file():
            raise FileNotFoundError(f"Isaac server runner not found: {isaac_runner_path}")
        isaac_runner_text = ensure_trailing_newline(isaac_runner_path.read_text())
        stage_block = f"""\
cat > {shlex.quote(runtime_root)}/lib/isaac_server_runner.py <<'TEW_ISAAC_RUNNER_EOF'
{isaac_runner_text}TEW_ISAAC_RUNNER_EOF
chmod +x {shlex.quote(runtime_root)}/lib/isaac_server_runner.py
{modality_block}
export ISAAC_DIR={shlex.quote(settings.isaac_dir)}
{_mlxp_isaac_assets_block(settings)}"""

    uv_bin_dir = shlex.quote(f"{settings.ddn_user_home}/.local/bin")

    return f"""\
set -euo pipefail
export PATH="{uv_bin_dir}:$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
{_hf_cache_exports(settings)}

{_uv_bootstrap_block(settings)}

{_repo_runtime_preamble(repo_path, snapshot)}
TRAIN_REPO_WORKTREE="$PWD"

{_snapshot_preamble(snapshot)}
mkdir -p {shlex.quote(runtime_root)}/clusters {shlex.quote(runtime_root)}/lib {shlex.quote(exp_dir)}/logs {shlex.quote(eval_dir)}
cat > {shlex.quote(runtime_root)}/clusters/mlxp.env <<'TEW_CLUSTER_ENV_EOF'
{cluster_env}TEW_CLUSTER_ENV_EOF
cat > {shlex.quote(runtime_root)}/lib/_common.sh <<'TEW_COMMON_EOF'
{common_text}TEW_COMMON_EOF
cat > {shlex.quote(runtime_root)}/lib/{model.eval_body_script} <<'TEW_EVAL_BODY_EOF'
{eval_body_text}TEW_EVAL_BODY_EOF
chmod +x {shlex.quote(runtime_root)}/lib/{model.eval_body_script}
{stage_block}
export SUBMIT_TRAIN_REPO_DIR="$TRAIN_REPO_WORKTREE"
{chr(10).join(eval_exports)}

bash {shlex.quote(runtime_root)}/lib/{model.eval_body_script}
"""


def _job_comment(req: MlxpSubmitRequest, variant, snapshot: dict, model: TrainingModel,
                 train_settings=None) -> str:
    settings = get_settings()
    output_namespace = str(snapshot["output_namespace"])
    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    # Use the resolved model identity (same as slurm's meta) rather than raw
    # variant vars: a variant that sets only TRAIN_MODEL — or nothing — would
    # otherwise record a divergent/empty model_id that the details page reads.
    fields: dict[str, str | None] = {
        "phase": req.phase,
        "variant": req.variant,
        "model_id": model.id,
        "model_label": model.label,
        "wandb_project": _wandb_project(),
        "output_namespace": output_namespace,
    }
    if req.phase == "train":
        fields["train_num_gpus"] = str(train_settings.num_gpus)
        fields["train_max_steps"] = str(train_settings.max_steps)
        fields["train_save_steps"] = str(train_settings.save_steps)
        fields["train_num_workers"] = str(train_settings.num_workers)
        if req.global_batch_size is not None:
            fields["train_global_batch_size"] = str(req.global_batch_size)
        if req.action_horizon is not None:
            fields["train_action_horizon"] = str(req.action_horizon)
        fields["checkpoint_dir"] = paths.checkpoint_dir(exp_dir, output_namespace)
    else:
        if req.eval_num_envs_per_gpu is not None:
            fields["eval_num_envs_per_gpu"] = str(req.eval_num_envs_per_gpu)
        if req.eval_n_episodes is not None:
            fields["eval_n_episodes"] = str(req.eval_n_episodes)
        if req.eval_n_runs is not None:
            fields["eval_n_runs"] = str(req.eval_n_runs)
        if snapshot.get("eval_sets"):
            fields["eval_sets"] = " ".join(snapshot["eval_sets"])
        if snapshot.get("eval_tasks"):
            fields["eval_tasks"] = " ".join(snapshot["eval_tasks"])
        if req.eval_overwrite_results:
            fields["eval_overwrite_results"] = "true"
        if snapshot.get("dexjoco_task"):
            fields["dexjoco_task"] = str(snapshot["dexjoco_task"])
        if req.checkpoint_path:
            fields["checkpoint_path"] = req.checkpoint_path.strip()
        rollout = snapshot.get("eval", {}).get("rollout", {})
        if rollout:
            fields["dexjoco_inference_mode"] = str(rollout.get("inference_mode") or "")
            fields["dexjoco_action_horizon"] = str(rollout.get("action_horizon") or "")
            fields["dexjoco_replan_ratio"] = str(rollout.get("replan_ratio") or "")
        eval_dir = paths.eval_dir(exp_dir, output_namespace)
        fields["eval_dir"] = eval_dir
        fields["results_path"] = paths.results_path(eval_dir)
        fields["job_log_dir"] = paths.job_log_dir(exp_dir, output_namespace)
    fields["config_snapshot_path"] = snapshot["path"]
    fields["config_snapshot_meta_path"] = snapshot["meta_path"]
    fields["submit_git_repo_path"] = snapshot.get("git_repo_path")
    fields["submit_git_repo_label"] = snapshot.get("git_repo_label")
    fields["submit_git_branch"] = snapshot.get("git_branch")
    fields["submit_git_commit"] = snapshot.get("git_commit")
    fields["submit_git_commit_subject"] = snapshot.get("git_commit_subject")
    fields["submit_git_dirty_at_submit"] = "true" if snapshot.get("git_dirty_at_submit") else "false"
    fields["submit_git_committed_dirty"] = "true" if snapshot.get("git_committed_dirty") else "false"
    return comment_field_fragment(fields)


def _render_job_yaml(job_id: str, job_name: str, body: str, num_gpus: int, cpu: str, mem: str,
                     wandb_secret: str, node: str, job_class: str, comment: str, train_note: str,
                     settings: MlxpSettings, model_output_dir: str | None = None) -> dict:
    # Per the MLXP guideline: dedicated requires the job-class label AND a
    # hostname-In affinity; queue classes (normal/background) must not pin a
    # node and instead constrain to the team zone via nodeSelector.
    if job_class == "dedicated":
        placement: dict = {
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [{
                            "matchExpressions": [{
                                "key": "kubernetes.io/hostname",
                                "operator": "In",
                                "values": [node],
                            }],
                        }],
                    },
                },
            },
        }
    else:
        placement = {"nodeSelector": {"mlx.navercorp.com/zone": settings.zone}}
    job_labels = {**labels(settings), "mlxp/job-class": job_class}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_id,
            "namespace": settings.namespace,
            "labels": job_labels,
            # display-name carries the human-readable {phase}_{variant}_{ts}
            # with underscores — invalid in k8s resource names but fine here.
            # comment mirrors slurm's sacct Comment field so the details page
            # can recover phase/variant even when the user picked a custom
            # job_name that doesn't match the unified regex.
            "annotations": {
                "train-eval-web/display-name": job_name,
                "train-eval-web/comment": comment,
                "train-eval-web/train-note": train_note,
            },
        },
        "spec": {
            "ttlSecondsAfterFinished": 604800,  # 7 days — Jobs page keeps showing them
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": job_labels,
                    "annotations": {
                        "mlx.navercorp.com/zone": settings.zone,
                        "sidecar.istio.io/inject": "false",
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": settings.image_pull_secret}],
                    "volumes": [
                        {"name": "ddn", "persistentVolumeClaim": {"claimName": settings.ddn_pvc}},
                        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "256Gi"}},
                    ],
                    **placement,
                    "containers": [{
                        "name": "main",
                        "image": settings.image,
                        "imagePullPolicy": "Always",
                        "command": ["/bin/bash", "-c"],
                        "args": [body],
                        "env": [
                            {
                                "name": "WANDB_API_KEY",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": wandb_secret,
                                        "key": "api-key",
                                        "optional": True,
                                    },
                                },
                            },
                            *(
                                [{"name": "MODEL_OUTPUT_DIR", "value": model_output_dir}]
                                if model_output_dir else []
                            ),
                        ],
                        "volumeMounts": [
                            {"name": "ddn",  "mountPath": settings.ddn_mount},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                        ],
                        "resources": {
                            "requests": {"cpu": cpu, "memory": mem, "nvidia.com/gpu": str(num_gpus)},
                            "limits":   {"cpu": cpu, "memory": mem, "nvidia.com/gpu": str(num_gpus)},
                        },
                    }],
                },
            },
        },
    }
