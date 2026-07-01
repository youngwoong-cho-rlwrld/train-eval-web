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
from .train_overrides import resolve_train_action_horizon, validate_global_batch_divisible
from .variant_values import variant_int
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
    `<user>-<job-name>`, with DNS-label sanitation and 63-char max length."""
    prefix = _k8s_name_segment(settings.user)
    body = _k8s_name_segment(job_name)
    name = f"{prefix}-{body}"
    if len(name) <= 63:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
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
    num_gpus: int = 2
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
    wandb_secret: str | None = None
    eval_num_envs_per_gpu: int | None = Field(default=None, ge=1)
    eval_n_episodes: int | None = Field(default=None, ge=1)
    eval_n_runs: int | None = Field(default=None, ge=1)
    eval_sets: list[str] | None = None
    eval_overwrite_results: bool = False
    checkpoint_path: str | None = None
    # Optional override for the auto-generated display job_name. Validated
    # against the unified regex in submit.resolve_job_name.
    job_name: str | None = None
    output_namespace: str | None = None
    commit_dirty_changes: bool = False


class MlxpSubmitResponse(BaseModel):
    job_id: str             # 6-char k8s Job name, used in /jobs/<cluster>/<id>
    job_name: str           # human-readable {phase}_{variant}_{ts}, same shape as slurm
    pod_name: str | None = None
    yaml: str
    apply_stdout: str


async def submit_mlxp(req: MlxpSubmitRequest) -> MlxpSubmitResponse:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    if req.num_gpus not in _GPU_RESOURCES:
        raise ValueError(f"num_gpus must be one of {list(_GPU_RESOURCES)}, got {req.num_gpus}")
    if req.phase == "eval" and not (req.checkpoint_path or "").strip():
        raise ValueError("checkpoint_path is required for MLXP eval")
    if req.phase == "eval" and req.eval_num_envs_per_gpu is not None and req.eval_num_envs_per_gpu > 1:
        raise ValueError(
            "eval_num_envs_per_gpu > 1 is disabled: the ALLEX target reset "
            "path is not vector-env safe"
        )
    if req.phase != "eval" and any((
        req.eval_num_envs_per_gpu is not None,
        req.eval_n_episodes is not None,
        req.eval_n_runs is not None,
        req.eval_sets is not None,
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
    if req.phase == "train":
        validate_global_batch_divisible(model.family, req.global_batch_size, req.num_gpus)
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
        )
    await _write_snapshot_to_ddn(snapshot)
    if req.phase == "eval":
        body_script = _render_eval_body_script(variant, req, job_name, snapshot, model, repo_path, settings)
    else:
        body_script = _render_body_script(variant, req, job_name, snapshot, model, repo_path, settings)
    spec = _render_job_yaml(
        job_id,
        job_name,
        body_script,
        req.num_gpus,
        cpu,
        mem,
        req.wandb_secret or settings.wandb_secret,
        node,
        req.job_class,
        _job_comment(req, variant, snapshot),
        train_note,
        settings,
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
        pod_name=None,
        yaml=yaml_text,
        apply_stdout=stdout.decode(errors="replace").strip(),
    )


def _build_snapshot_payload(*, variant, req: MlxpSubmitRequest, job_id: str, job_name: str,
                            node: str, submit_git, model: TrainingModel,
                            settings: MlxpSettings, train_note: str,
                            action_horizon_mode: str) -> dict:
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
    from .submit import normalize_eval_sets

    eval_sets = normalize_eval_sets(req.eval_sets)
    suffix = req.output_namespace or f"{snapshot_suffix(job_name)}_{job_id}"
    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    path = paths.config_path(exp_dir, suffix)
    meta_path = paths.meta_path(exp_dir, suffix)
    checkpoint_path = (req.checkpoint_path or "").strip()
    config_text = render_eval_config_preview(
        base_config=variant.raw,
        variant=variant.name,
        job_name=job_name,
        cluster="mlxp",
        node=node,
        dataset_override=req.dataset_override,
        eval_n_episodes=req.eval_n_episodes,
        eval_n_runs=req.eval_n_runs,
        eval_sets=eval_sets,
        eval_overwrite_results=req.eval_overwrite_results,
        checkpoint_path=checkpoint_path,
        extra_args=req.extra_args,
        data_dir=settings.datasets_dir,
        eval_unset_cuda_visible_devices_for_server=1,
        train_git_commit=req.train_git_commit,
        train_note=train_note,
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
        "overwrite_results": req.eval_overwrite_results,
        "unset_cuda_visible_devices_for_server": 1,
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
        "checkpoint_path": checkpoint_path,
        "git_commit": submit_git.commit,
        "git_commit_subject": submit_git.commit_subject,
        "git_branch": submit_git.branch,
        "git_repo_path": submit_git.repo_path,
        "git_repo_label": submit_git.repo_label,
        "git_dirty_at_submit": submit_git.dirty_before,
        "git_committed_dirty": submit_git.committed_dirty,
    }


def _path_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "job"


def _shell_words(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _mlxp_worktree_path(snapshot: dict) -> str:
    settings = get_settings()
    job_id = snapshot.get("job_id")
    if isinstance(job_id, str) and job_id:
        leaf = job_id
    else:
        leaf = _path_slug(str(snapshot.get("job_name") or "job"))
    return f"{settings.experiments_dir}/.worktrees/{leaf}"


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

    max_steps = str(req.max_steps or variant_int(variant, "MAX_STEPS", 30000))
    save_steps = str(req.save_steps or variant_int(variant, "SAVE_STEPS", 1000))
    num_workers = str(req.num_workers or variant_int(variant, "TRAIN_NUM_WORKERS", 16))
    batch_size = variant.vars.get("TRAIN_BATCH_SIZE", "64")
    if family == "n1.5" and req.global_batch_size is not None:
        batch_size = str(req.global_batch_size // req.num_gpus)
    train_extra = _shell_words(variant.arrays.get("TRAIN_EXTRA_ARGS") or [])
    user_extra = _shell_words(req.extra_args)

    output_namespace = req.output_namespace or _path_slug(job_name)
    ckpt_dir = paths.checkpoint_dir(f"{settings.experiments_dir}/{variant.name}", output_namespace)
    run_log_dir = f"{ckpt_dir}/logs"
    wandb_project = shlex.quote(_wandb_project())

    if family == "n1.6":
        return _render_body_n16(
            variant=variant, req=req, job_name=job_name, names=names,
            max_steps=max_steps, save_steps=save_steps, num_workers=num_workers,
            batch_size=batch_size,
            train_extra=train_extra, user_extra=user_extra, ckpt_dir=ckpt_dir,
            snapshot=snapshot, model=model, repo_path=repo_path, settings=settings,
        )
    if family != "n1.5":
        raise ValueError(f"unsupported MLXP model family: {family}")

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
    --output-dir {ckpt_dir} \\
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
                     batch_size: str, train_extra: str, user_extra: str,
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
    global_batch = req.global_batch_size or int(batch_size) * req.num_gpus
    run_log_dir = f"{ckpt_dir}/logs"
    wandb_project = shlex.quote(_wandb_project())
    uv_bin_dir = shlex.quote(f"{settings.ddn_user_home}/.local/bin")
    output_namespace = ckpt_dir.rstrip("/").rsplit("/", 1)[-1]
    output_parent = ckpt_dir.rsplit("/", 1)[0]
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
    --modality-config-path /tmp/modality_config.py \\
    --num-gpus {req.num_gpus} \\
    --output-dir {output_parent} \\
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


def _render_eval_body_script(
    variant,
    req: MlxpSubmitRequest,
    job_name: str,
    snapshot: dict,
    model: TrainingModel,
    repo_path: str,
    settings: MlxpSettings,
) -> str:
    """Render MLXP eval by staging the same eval body script used on Slurm."""
    # DexJoCo eval drives an Isaac-side server from lib/dexjoco/, which MLXP does
    # not stage into the pod. DexJoCo eval is Slurm-only for now; fail fast rather
    # than launch a job that cannot find the server.
    if model.eval_body_script == "eval_body_dexjoco.sh":
        raise ValueError(
            "DexJoCo eval is not supported on MLXP yet "
            "(lib/dexjoco/ is not staged into the MLXP pod); run DexJoCo eval on a Slurm cluster"
        )
    eval_body_path = LIB_DIR / model.eval_body_script
    common_path = LIB_DIR / "_common.sh"
    isaac_runner_path = LIB_DIR / "isaac_server_runner.py"
    if not eval_body_path.is_file():
        raise FileNotFoundError(f"eval body script not found: {eval_body_path}")
    if not isaac_runner_path.is_file():
        raise FileNotFoundError(f"Isaac server runner not found: {isaac_runner_path}")
    common_text = ensure_trailing_newline(common_path.read_text())
    eval_body_text = ensure_trailing_newline(eval_body_path.read_text())
    isaac_runner_text = ensure_trailing_newline(isaac_runner_path.read_text())

    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    output_namespace = str(snapshot.get("output_namespace") or _path_slug(job_name))
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
        f"export ISAAC_DIR={shlex.quote(settings.isaac_dir)}",
        f"export DATA_DIR={shlex.quote(settings.datasets_dir)}",
        f"export LOG_DIR={shlex.quote(f'{exp_dir}/logs')}",
    ]
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
    if req.eval_overwrite_results:
        eval_exports.append("export SUBMIT_EVAL_OVERWRITE_RESULTS=1")

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
cat > {shlex.quote(runtime_root)}/lib/isaac_server_runner.py <<'TEW_ISAAC_RUNNER_EOF'
{isaac_runner_text}TEW_ISAAC_RUNNER_EOF
chmod +x {shlex.quote(runtime_root)}/lib/isaac_server_runner.py
{modality_block}
export ISAAC_DIR={shlex.quote(settings.isaac_dir)}
{_mlxp_isaac_assets_block(settings)}
export SUBMIT_TRAIN_REPO_DIR="$TRAIN_REPO_WORKTREE"
{chr(10).join(eval_exports)}

bash {shlex.quote(runtime_root)}/lib/{model.eval_body_script}
"""


def _job_comment(req: MlxpSubmitRequest, variant, snapshot: dict) -> str:
    settings = get_settings()
    output_namespace = str(snapshot.get("output_namespace") or req.output_namespace or _path_slug(snapshot.get("job_name") or "job"))
    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    fields: dict[str, str | None] = {
        "phase": req.phase,
        "variant": req.variant,
        "model_id": variant.vars.get("MODEL_ID") or variant.vars.get("MODEL_VERSION") or "",
        "wandb_project": _wandb_project(),
        "output_namespace": output_namespace,
    }
    if req.phase == "train":
        max_steps = req.max_steps or variant_int(variant, "MAX_STEPS", 30000)
        save_steps = req.save_steps or variant_int(variant, "SAVE_STEPS", 1000)
        num_workers = req.num_workers or variant_int(variant, "TRAIN_NUM_WORKERS", 16)
        fields["train_num_gpus"] = str(req.num_gpus)
        fields["train_max_steps"] = str(max_steps)
        fields["train_save_steps"] = str(save_steps)
        fields["train_num_workers"] = str(num_workers)
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
        if req.eval_overwrite_results:
            fields["eval_overwrite_results"] = "true"
        if req.checkpoint_path:
            fields["checkpoint_path"] = req.checkpoint_path.strip()
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
                     settings: MlxpSettings) -> dict:
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
                        "env": [{
                            "name": "WANDB_API_KEY",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": wandb_secret,
                                    "key": "api-key",
                                    "optional": True,
                                },
                            },
                        }],
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
