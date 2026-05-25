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

import asyncio
import io
import re
import shlex
import shutil
import tarfile
import time
import uuid
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .mlxp_config import (
    MlxpSettings,
    get_settings,
    labels,
)
from .paths import EXPERIMENTS_DIR, LIB_DIR
from .submission_snapshot import (
    metadata_json,
    prepare_mlxp_training_git,
    render_eval_config_preview,
    render_training_config_snapshot,
    snapshot_metadata,
    snapshot_suffix,
    training_repo_label,
)
from .training_models import (
    TrainingModel,
    load_training_model,
    mlxp_repo_path,
    resolve_training_model,
)
from .train_overrides import resolve_train_action_horizon
from .variant_values import variant_int
from .wandb_config import get_project as _wandb_project
from .variants import load_variant
from typing import Literal


# Per-GPU resource map (from the Notion MLXP guide section 3.1).
# Node total: CPU=112, memory=1760Gi, GPU=8.
_GPU_RESOURCES = {
    1: ("14",  "220Gi"),
    2: ("28",  "440Gi"),
    4: ("56",  "880Gi"),
    8: ("100", "1500Gi"),
}


def _hf_cache_exports(settings: MlxpSettings) -> str:
    hf_home = shlex.quote(settings.hf_home)
    return f"""\
export HF_HOME={hf_home}
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
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
    num_gpus: int = 2
    global_batch_size: int | None = None
    max_steps: int | None = None
    save_steps: int | None = None
    action_horizon: int | None = Field(default=None, ge=1)
    # The k8s node to pin via nodeAffinity. Leave None to fall back to the
    # configured default node.
    node: str | None = None
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
    if req.phase == "train" and model.family == "n1.6":
        req.action_horizon = resolve_train_action_horizon(
            variant=variant,
            model_family=model.family,
            requested=req.action_horizon,
        )
    if (
        req.phase == "train"
        and req.global_batch_size is not None
        and model.family == "n1.5"
        and req.global_batch_size % req.num_gpus != 0
    ):
        raise ValueError("global_batch_size must be divisible by num_gpus for n1.5 training")
    # job_id is the k8s Job resource name — 6-char alpha-leading so it's
    # DNS-safe and URL-short. job_name is the display name carried as an
    # annotation; same shape as slurm's job_name.
    from .submit import resolve_job_name
    job_id = "m" + uuid.uuid4().hex[:5]
    job_name = resolve_job_name(req.job_name, req.phase, req.variant)
    repo_path = mlxp_training_repo_path(model)
    submit_git = await prepare_mlxp_training_git(
        repo_path=repo_path,
        repo_label=training_repo_label(model),
        job_name=job_name,
        commit_dirty_changes=req.commit_dirty_changes,
        require_clean=(req.phase == "train"),
    )

    node = req.node or settings.default_node
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
        _job_comment(req, variant, snapshot),
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
                            settings: MlxpSettings) -> dict:
    train_num_gpus = req.num_gpus
    train_max_steps = req.max_steps or variant_int(variant, "MAX_STEPS", 30000)
    train_save_steps = req.save_steps or variant_int(variant, "SAVE_STEPS", 1000)
    per_gpu_batch = int(variant.vars.get("TRAIN_BATCH_SIZE", "64"))
    train_global_batch_size = req.global_batch_size or per_gpu_batch * train_num_gpus
    suffix = f"{snapshot_suffix(job_name)}_{job_id}"
    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    path = f"{exp_dir}/config_{suffix}.sh"
    meta_path = f"{exp_dir}/config_{suffix}.meta.json"
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
        train_action_horizon=req.action_horizon,
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
        train_action_horizon=req.action_horizon,
        wandb_project=_wandb_project(),
        git=submit_git,
    )
    return {
        "job_id": job_id,
        "job_name": job_name,
        "phase": "train",
        "path": path,
        "meta_path": meta_path,
        "config_text": config_text,
        "meta_text": metadata_json(meta),
        "git_commit": submit_git.commit,
        "git_branch": submit_git.branch,
        "git_repo_path": submit_git.repo_path,
        "git_repo_label": submit_git.repo_label,
        "git_dirty_at_submit": submit_git.dirty_before,
        "git_committed_dirty": submit_git.committed_dirty,
    }


def _build_eval_snapshot_payload(*, variant, req: MlxpSubmitRequest, job_id: str, job_name: str,
                                 node: str, submit_git, model: TrainingModel,
                                 settings: MlxpSettings) -> dict:
    from .submit import _normalize_eval_sets

    eval_sets = _normalize_eval_sets(req.eval_sets)
    suffix = f"{snapshot_suffix(job_name)}_{job_id}"
    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    path = f"{exp_dir}/config_{suffix}.sh"
    meta_path = f"{exp_dir}/config_{suffix}.meta.json"
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
        wandb_project=_wandb_project(),
        git=submit_git,
    )
    meta["eval"] = {
        "checkpoint_path": checkpoint_path,
        "num_envs_per_gpu": req.eval_num_envs_per_gpu,
        "n_episodes": req.eval_n_episodes,
        "n_runs": req.eval_n_runs,
        "eval_sets": eval_sets,
        "overwrite_results": req.eval_overwrite_results,
    }
    return {
        "job_id": job_id,
        "job_name": job_name,
        "phase": "eval",
        "path": path,
        "meta_path": meta_path,
        "config_text": config_text,
        "meta_text": metadata_json(meta),
        "eval_sets": eval_sets,
        "checkpoint_path": checkpoint_path,
        "git_commit": submit_git.commit,
        "git_branch": submit_git.branch,
        "git_repo_path": submit_git.repo_path,
        "git_repo_label": submit_git.repo_label,
        "git_dirty_at_submit": submit_git.dirty_before,
        "git_committed_dirty": submit_git.committed_dirty,
    }


def _path_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "job"


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


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
    return f"""\
mkdir -p {shlex.quote(path.rsplit('/', 1)[0])}
cat > {shlex.quote(path)} <<'TRAIN_EVAL_CONFIG_SNAPSHOT'
{config_text}TRAIN_EVAL_CONFIG_SNAPSHOT
cat > {shlex.quote(meta_path)} <<'TRAIN_EVAL_CONFIG_META'
{meta_text}TRAIN_EVAL_CONFIG_META
"""


def _snapshot_tar(snapshot: dict) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as tf:
        for path_key, text_key in (
            ("path", "config_text"),
            ("meta_path", "meta_text"),
        ):
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
    batch_size = variant.vars.get("TRAIN_BATCH_SIZE", "64")
    if family == "n1.5" and req.global_batch_size is not None:
        batch_size = str(req.global_batch_size // req.num_gpus)
    train_extra = " ".join(variant.arrays.get("TRAIN_EXTRA_ARGS") or [])
    user_extra = " ".join(req.extra_args)

    ckpt_dir = f"{settings.experiments_dir}/{variant.name}/checkpoints/{job_name}"
    run_log_dir = f"{ckpt_dir}/logs"
    wandb_project = shlex.quote(_wandb_project())

    if family == "n1.6":
        return _render_body_n16(
            variant=variant, req=req, job_name=job_name, names=names,
            max_steps=max_steps, save_steps=save_steps, batch_size=batch_size,
            train_extra=train_extra, user_extra=user_extra, ckpt_dir=ckpt_dir,
            snapshot=snapshot, repo_path=repo_path, settings=settings,
        )
    if family != "n1.5":
        raise ValueError(f"unsupported MLXP model family: {family}")

    # ── N1.5: build the data_config.yaml rows ──
    if override_full is not None and isinstance(override, list) and any("|" in e for e in override):
        datasets_decl = override_full
    elif override_full is not None and isinstance(override, str):
        cfg = variant.vars.get("DATA_CONFIG", "allex_thetwo_ck40_egostereo")
        datasets_decl = [f"{override}|{cfg}|1.0"]
    elif variant.arrays.get("DATASETS"):
        datasets_decl = variant.arrays["DATASETS"]
    else:
        cfg = variant.vars.get("DATA_CONFIG", "allex_thetwo_ck40_egostereo")
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
    --dataloader_num_workers 16 \\
    --dataloader-prefetch-factor 10 \\
    --video-backend torchcodec \\
    --report-to wandb \\
    --pin_memory \\
    --run_name "{variant.name}" \\
    --seed 42 \\
    $RESUME_FLAG {train_extra} {user_extra}
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
    if use_file and _safe_yaml_relpath(rel) and path.is_file():
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


def _safe_yaml_relpath(rel: str) -> bool:
    path = Path(rel)
    return (
        bool(rel)
        and not path.is_absolute()
        and path.name == rel
        and not rel.startswith(".")
        and ".." not in path.parts
        and path.suffix in {".yaml", ".yml"}
    )


def _render_body_n16(*, variant, req: MlxpSubmitRequest, job_name: str,
                     names: list[str], max_steps: str, save_steps: str,
                     batch_size: str, train_extra: str, user_extra: str,
                     ckpt_dir: str, snapshot: dict, repo_path: str,
                     settings: MlxpSettings) -> str:
    """Body script for GR00T N1.6 (launch_finetune.py).

    Unlike N1.5, N1.6 takes --dataset-path (multiple) + --modality-config-path
    (a Python file). We inline the modality config from the local variant
    directory so MLXP doesn't need a rsync step.
    """
    modality_rel = variant.vars.get("TRAIN_MODALITY_CONFIG")
    if not modality_rel:
        raise ValueError(f"variant {variant.name}: TRAIN_MODALITY_CONFIG missing")
    modality_path = EXPERIMENTS_DIR / variant.name / modality_rel
    if not modality_path.is_file():
        raise FileNotFoundError(f"modality config not found: {modality_path}")
    modality_text = modality_path.read_text()

    dataset_paths_arg = " \\\n        ".join(
        f"{settings.datasets_dir}/{n}" for n in names
    )
    global_batch = req.global_batch_size or int(batch_size) * req.num_gpus
    run_log_dir = f"{ckpt_dir}/logs"
    wandb_project = shlex.quote(_wandb_project())
    uv_userbase = shlex.quote(f"{settings.ddn_user_home}/.local")
    uv_bin_dir = shlex.quote(f"{settings.ddn_user_home}/.local/bin")
    action_horizon_arg = (
        f" --action-horizon {req.action_horizon}"
        if req.action_horizon is not None
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

if ! command -v uv >/dev/null 2>&1; then
    PYTHONUSERBASE={uv_userbase} python3 -m pip install --user uv
fi

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
    --output-dir {ckpt_dir} \\
    --global-batch-size {global_batch} \\
    --learning-rate 1e-4 \\
    --max-steps {max_steps} \\
    --save-steps {save_steps} \\
    --save-total-limit 5 \\
    --dataloader-num-workers 8 \\
    --experiment-name "{job_name}" \\
    --use-wandb \\
    --wandb-project {wandb_project} \\
    $RESUME_FLAG{action_horizon_arg} {train_extra} {user_extra}
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
    eval_body_path = LIB_DIR / model.eval_body_script
    common_path = LIB_DIR / "_common.sh"
    isaac_runner_path = LIB_DIR / "isaac_server_runner.py"
    if not eval_body_path.is_file():
        raise FileNotFoundError(f"eval body script not found: {eval_body_path}")
    if not isaac_runner_path.is_file():
        raise FileNotFoundError(f"Isaac server runner not found: {isaac_runner_path}")
    common_text = _ensure_trailing_newline(common_path.read_text())
    eval_body_text = _ensure_trailing_newline(eval_body_path.read_text())
    isaac_runner_text = _ensure_trailing_newline(isaac_runner_path.read_text())

    exp_dir = f"{settings.experiments_dir}/{variant.name}"
    runtime_root = f"{settings.experiments_dir}/.runtime/{snapshot['job_id']}"
    config_path = snapshot["path"]
    modality_block = ""
    modality_rel = variant.vars.get("TRAIN_MODALITY_CONFIG")
    if modality_rel:
        modality_path = EXPERIMENTS_DIR / variant.name / modality_rel
        if not modality_path.is_file():
            raise FileNotFoundError(f"modality config not found: {modality_path}")
        modality_target = f"{exp_dir}/{modality_rel}"
        modality_text = _ensure_trailing_newline(modality_path.read_text())
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
        f"export SUBMIT_CONFIG_FILE={shlex.quote(config_path)}",
        f"export SUBMIT_DATA_DIR={shlex.quote(settings.datasets_dir)}",
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

    uv_userbase = shlex.quote(f"{settings.ddn_user_home}/.local")
    uv_bin_dir = shlex.quote(f"{settings.ddn_user_home}/.local/bin")

    return f"""\
set -euo pipefail
export PATH="{uv_bin_dir}:$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
{_hf_cache_exports(settings)}

if ! command -v uv >/dev/null 2>&1; then
    PYTHONUSERBASE={uv_userbase} python3 -m pip install --user uv
fi

{_repo_runtime_preamble(repo_path, snapshot)}
TRAIN_REPO_WORKTREE="$PWD"

{_snapshot_preamble(snapshot)}
mkdir -p {shlex.quote(runtime_root)}/clusters {shlex.quote(runtime_root)}/lib {shlex.quote(exp_dir)}/logs {shlex.quote(exp_dir)}/eval_results
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
export SUBMIT_TRAIN_REPO_DIR="$TRAIN_REPO_WORKTREE"
{chr(10).join(eval_exports)}

bash {shlex.quote(runtime_root)}/lib/{model.eval_body_script}
"""


def _job_comment(req: MlxpSubmitRequest, variant, snapshot: dict) -> str:
    comment = (
        f"phase={req.phase};variant={req.variant};model_id={variant.vars.get('MODEL_ID') or variant.vars.get('MODEL_VERSION') or ''};"
        f"wandb_project={_wandb_project()}"
    )
    if req.phase == "train":
        max_steps = req.max_steps or variant_int(variant, "MAX_STEPS", 30000)
        save_steps = req.save_steps or variant_int(variant, "SAVE_STEPS", 1000)
        comment += (
            f";train_num_gpus={req.num_gpus};"
            f"train_max_steps={max_steps};train_save_steps={save_steps}"
        )
        if req.global_batch_size is not None:
            comment += f";train_global_batch_size={req.global_batch_size}"
        if req.action_horizon is not None:
            comment += f";train_action_horizon={req.action_horizon}"
    else:
        if req.eval_num_envs_per_gpu is not None:
            comment += f";eval_num_envs_per_gpu={req.eval_num_envs_per_gpu}"
        if req.eval_n_episodes is not None:
            comment += f";eval_n_episodes={req.eval_n_episodes}"
        if req.eval_n_runs is not None:
            comment += f";eval_n_runs={req.eval_n_runs}"
        if snapshot.get("eval_sets"):
            comment += f";eval_sets={' '.join(snapshot['eval_sets'])}"
        if req.eval_overwrite_results:
            comment += ";eval_overwrite_results=true"
        if req.checkpoint_path:
            comment += f";checkpoint_path={req.checkpoint_path.strip()}"
    comment += (
        f";config_snapshot_path={snapshot['path']}"
        f";config_snapshot_meta_path={snapshot['meta_path']}"
        f";submit_git_repo_path={snapshot['git_repo_path']}"
        f";submit_git_repo_label={snapshot['git_repo_label']}"
    )
    if snapshot.get("git_branch"):
        comment += f";submit_git_branch={snapshot['git_branch']}"
    if snapshot.get("git_commit"):
        comment += f";submit_git_commit={snapshot['git_commit']}"
    comment += (
        f";submit_git_dirty_at_submit={'true' if snapshot.get('git_dirty_at_submit') else 'false'}"
        f";submit_git_committed_dirty={'true' if snapshot.get('git_committed_dirty') else 'false'}"
    )
    return comment


def _render_job_yaml(job_id: str, job_name: str, body: str, num_gpus: int, cpu: str, mem: str,
                     wandb_secret: str, node: str, comment: str,
                     settings: MlxpSettings) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_id,
            "namespace": settings.namespace,
            "labels": labels(settings),
            # display-name carries the human-readable {phase}_{variant}_{ts}
            # with underscores — invalid in k8s resource names but fine here.
            # comment mirrors slurm's sacct Comment field so the details page
            # can recover phase/variant even when the user picked a custom
            # job_name that doesn't match the unified regex.
            "annotations": {
                "train-eval-web/display-name": job_name,
                "train-eval-web/comment": comment,
            },
        },
        "spec": {
            "ttlSecondsAfterFinished": 604800,  # 7 days — Jobs page keeps showing them
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": labels(settings),
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
