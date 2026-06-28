"""Training model registry.

Adding a model should be data/config, not another hardcoded branch in the
submitter. Each model lives in `configs/models/<model_id>.env`; experiments
select one with MODEL_ID=<model_id>. Existing configs that only set
MODEL_VERSION=n1.5/n1.6 still resolve to the matching model id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .cluster_settings import parse_env_text
from .paths import MODELS_DIR

ACTION_HORIZON_MODES = {"none", "modality", "cli", "modality_and_cli"}
SUPPORTED_FAMILIES = frozenset({"n1.5", "n1.6"})  # training families flags.py + variants.py know how to render; pi0.5 is eval-only (dexjoco harness)


@dataclass(frozen=True)
class TrainingModel:
    id: str
    label: str
    family: str
    flags_profile: str
    slurm_repo_var: str | None
    slurm_repo_dir: str | None
    mlxp_repo_dir: str | None
    train_body_script: str
    eval_body_script: str
    train_walltime: str
    eval_walltime: str
    action_horizon_mode: str

    def body_for_phase(self, phase: str) -> tuple[str, str]:
        if phase == "train":
            return self.train_body_script, self.train_walltime
        if phase == "eval":
            return self.eval_body_script, self.eval_walltime
        raise ValueError(f"unsupported phase: {phase}")


def model_id_for_variant(variant: Any) -> str:
    variant_vars = getattr(variant, "vars", {}) or {}
    return (
        variant_vars.get("MODEL_ID")
        or variant_vars.get("TRAIN_MODEL")
        or variant_vars.get("MODEL_VERSION")
        or "n1.5"
    ).strip()


def resolve_training_model(variant: Any) -> TrainingModel:
    return load_training_model(model_id_for_variant(variant))


def action_horizon_mode_for_variant(model: TrainingModel, variant: Any) -> str:
    variant_vars = getattr(variant, "vars", {}) or {}
    mode = (
        variant_vars.get("TRAIN_ACTION_HORIZON_MODE")
        or variant_vars.get("ACTION_HORIZON_MODE")
        or model.action_horizon_mode
    ).strip()
    return _validate_action_horizon_mode(model.id, mode)


def family_derives_per_gpu_batch_size(family: str) -> bool:
    """Whether a model family derives TRAIN_BATCH_SIZE = global_batch / num_gpus.

    n1.5 (gr00t_finetune.py) takes a per-GPU batch size, so the renderer splits
    the requested global batch across GPUs; n1.6/PhysiXel consume the global
    batch directly. Centralized here so adding/renaming families doesn't require
    editing config renderers.
    """
    return family == "n1.5"


def rewrites_modality_action_horizon(mode: str) -> bool:
    return mode in {"modality", "modality_and_cli"}


def passes_action_horizon_cli(mode: str) -> bool:
    return mode in {"cli", "modality_and_cli"}


def load_training_model(model_id: str) -> TrainingModel:
    model_id = (model_id or "n1.5").strip()
    path = MODELS_DIR / f"{model_id}.env"
    if not path.is_file():
        raise ValueError(
            f"training model {model_id!r} not found; add configs/models/{model_id}.env"
        )
    data = parse_env_text(path.read_text(), validate_keys=True)
    family = (data.get("MODEL_FAMILY") or data.get("MODEL_VERSION") or model_id).strip()
    flags_profile = (data.get("FLAGS_PROFILE") or family).strip()
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"model {model_id}: unsupported MODEL_FAMILY {family!r}")
    if flags_profile not in SUPPORTED_FAMILIES:
        raise ValueError(f"model {model_id}: unsupported FLAGS_PROFILE {flags_profile!r}")
    action_horizon_mode = (
        data.get("ACTION_HORIZON_MODE")
        or ("modality" if family == "n1.6" else "none")
    ).strip()
    action_horizon_mode = _validate_action_horizon_mode(model_id, action_horizon_mode)
    train_body_script = (data.get("TRAIN_BODY_SCRIPT") or "").strip()
    eval_body_script = (data.get("EVAL_BODY_SCRIPT") or "").strip()
    if not train_body_script or not eval_body_script:
        raise ValueError(f"model {model_id}: TRAIN_BODY_SCRIPT and EVAL_BODY_SCRIPT are required")
    return TrainingModel(
        id=model_id,
        label=(data.get("MODEL_LABEL") or model_id).strip(),
        family=family,
        flags_profile=flags_profile,
        slurm_repo_var=(data.get("SLURM_REPO_VAR") or "").strip() or None,
        slurm_repo_dir=(data.get("SLURM_REPO_DIR") or "").strip() or None,
        mlxp_repo_dir=(data.get("MLXP_REPO_DIR") or "").strip() or None,
        train_body_script=train_body_script,
        eval_body_script=eval_body_script,
        train_walltime=(data.get("TRAIN_WALLTIME") or "48:00:00").strip(),
        eval_walltime=(data.get("EVAL_WALLTIME") or "08:00:00").strip(),
        action_horizon_mode=action_horizon_mode,
    )


def slurm_repo_path(cluster_vars: dict[str, str], model: TrainingModel) -> str:
    if model.slurm_repo_dir:
        return expand_value(model.slurm_repo_dir, cluster_vars)
    if not model.slurm_repo_var:
        raise ValueError(f"model {model.id} missing SLURM_REPO_VAR or SLURM_REPO_DIR")
    path = (cluster_vars.get(model.slurm_repo_var) or "").strip()
    if not path:
        raise ValueError(f"cluster config missing {model.slurm_repo_var} for model {model.id}")
    return path


def mlxp_repo_path(model: TrainingModel, env: dict[str, str]) -> str:
    if not model.mlxp_repo_dir:
        raise ValueError(f"model {model.id} missing MLXP_REPO_DIR")
    return expand_value(model.mlxp_repo_dir, env)


_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand_value(value: str, env: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        return env.get(key, match.group(0))

    return _VAR_RE.sub(repl, value)


def _validate_action_horizon_mode(model_id: str, mode: str) -> str:
    if mode not in ACTION_HORIZON_MODES:
        raise ValueError(
            f"model {model_id}: unsupported ACTION_HORIZON_MODE {mode!r}"
        )
    return mode
