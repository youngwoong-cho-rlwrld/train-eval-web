"""Persisted MLXP settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from . import cluster_settings


_SETTINGS_DIR = Path.home() / ".train-eval-web"
_SETTINGS_FILE = _SETTINGS_DIR / "mlxp.json"

# This application's identity, written as the `tool` label on every Job/Pod it
# creates and used to scope job listing selectors. Not user config: changing it
# would orphan a user's existing jobs from the UI. The `owner` label is always
# the resolved user.
TOOL_LABEL = "train-eval-web"


_DEFAULT_DDN_MOUNT = "/data"


def _default_user() -> str:
    return os.environ.get("TRAIN_EVAL_MLXP_USER") or os.environ.get("USER") or "youngwoong"


def _defaults_for(user: str, ddn_mount: str, ddn_home: str) -> dict[str, Any]:
    """Derived defaults for every field, given the three roots the rest hang
    off: the job ``user``, the DDN mount, and the per-user DDN home
    (``{ddn_mount}/{user}`` unless overridden)."""
    return {
        "user": user,
        "namespace": "p-rlwrld",
        "gpus_per_node": 8,
        "ddn_mount": ddn_mount,
        "ddn_user_home": ddn_home,
        "datasets_dir": f"{ddn_home}/datasets",
        # Training outputs go to the org unified checkpoints root (per-user
        # folder), not the legacy per-user home. UNIFIED_EXPERIMENTS_DIR (or the
        # legacy TRAIN_EVAL_MLXP_EXPERIMENTS_DIR) still wins when set.
        "experiments_dir": f"{ddn_mount}/rlwrld-unified-checkpoints/{user}/experiments",
        "hf_home": f"{ddn_home}/.cache/huggingface",
        "workspace_dir": f"{ddn_home}/workspace",
        "isaac_dir": f"{ddn_home}/workspace/rlwrld_isaac",
        # DexJoCo (MuJoCo benchmark) eval deps on the DDN. Mirrors the kakao
        # cluster.env layout: a dexjoco repo (configs + openpi + eval client) and
        # a micromamba root holding the `dexjoco` (MuJoCo client) and `openpi`
        # (pi0.5 server) envs. Defaults follow the same tree used on kakao.
        "dexjoco_dir": f"{ddn_home}/workspace/dexjoco",
        "micromamba_bin": f"{ddn_home}/bin/micromamba",
        "mamba_root_prefix": f"{ddn_home}/micromamba",
        "dexjoco_eval_env": "dexjoco",
        "dexjoco_openpi_env": "openpi",
        "data_pod_name": f"{user}-data-pod",
        "ddn_pvc": "ddn-rlwrld-shared",
        "image": "mlxp.kr.ncr.ntruss.com/rlwrld-gpu-base:latest",
        "image_pull_secret": "mlxp-registry",
        "zone": "private-h200-rlwrld-0",
        "wandb_secret": f"{user}-wandb",
    }


class MlxpSettings(BaseModel):
    user: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    # Cluster constant (GPUs per H200 node), not user config. Kept because it is
    # non-discoverable under this project's RBAC (`kubectl get nodes` is
    # forbidden); node GPU type and the GPU-node set are derived, not configured.
    gpus_per_node: int = Field(default=8, ge=1)
    ddn_mount: str = Field(min_length=1)
    ddn_user_home: str = Field(min_length=1)
    datasets_dir: str = Field(min_length=1)
    experiments_dir: str = Field(min_length=1)
    hf_home: str = Field(min_length=1)
    workspace_dir: str = Field(min_length=1)
    isaac_dir: str = Field(min_length=1)
    dexjoco_dir: str = Field(min_length=1)
    micromamba_bin: str = Field(min_length=1)
    mamba_root_prefix: str = Field(min_length=1)
    dexjoco_eval_env: str = Field(min_length=1)
    dexjoco_openpi_env: str = Field(min_length=1)
    data_pod_name: str = Field(min_length=1)
    ddn_pvc: str = Field(min_length=1)
    image: str = Field(min_length=1)
    image_pull_secret: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    wandb_secret: str = Field(min_length=1)


class MlxpSettingsUpdate(BaseModel):
    user: str = Field(min_length=1)


# Each settings field resolves from an env var. The primary name follows the
# slurm cluster-config convention (plain DATA_DIR, ISAAC_DIR, DEXJOCO_DIR, … for
# concepts shared with kakao/skt; an MLXP_ prefix for MLXP-only concepts). The
# legacy TRAIN_EVAL_MLXP_* name is still accepted so effective envs saved before
# this rename keep resolving identically. Priority: primary, then legacy.
_FIELD_ENV_NAMES: dict[str, tuple[str, str]] = {
    "user": ("USER", "TRAIN_EVAL_MLXP_USER"),
    "namespace": ("MLXP_NAMESPACE", "TRAIN_EVAL_MLXP_NAMESPACE"),
    "gpus_per_node": ("MLXP_GPUS_PER_NODE", "TRAIN_EVAL_MLXP_GPUS_PER_NODE"),
    "ddn_mount": ("MLXP_DDN_MOUNT", "TRAIN_EVAL_MLXP_DDN_MOUNT"),
    "ddn_user_home": ("MLXP_HOME", "TRAIN_EVAL_MLXP_HOME"),
    "datasets_dir": ("DATA_DIR", "TRAIN_EVAL_MLXP_DATASETS_DIR"),
    "experiments_dir": ("UNIFIED_EXPERIMENTS_DIR", "TRAIN_EVAL_MLXP_EXPERIMENTS_DIR"),
    "hf_home": ("HF_HOME", "TRAIN_EVAL_MLXP_HF_HOME"),
    "workspace_dir": ("WORKSPACE_DIR", "TRAIN_EVAL_MLXP_WORKSPACE_DIR"),
    "isaac_dir": ("ISAAC_DIR", "TRAIN_EVAL_MLXP_ISAAC_DIR"),
    "dexjoco_dir": ("DEXJOCO_DIR", "TRAIN_EVAL_MLXP_DEXJOCO_DIR"),
    "micromamba_bin": ("MICROMAMBA_BIN", "TRAIN_EVAL_MLXP_MICROMAMBA_BIN"),
    "mamba_root_prefix": ("MAMBA_ROOT_PREFIX", "TRAIN_EVAL_MLXP_MAMBA_ROOT_PREFIX"),
    "dexjoco_eval_env": ("DEXJOCO_EVAL_ENV", "TRAIN_EVAL_MLXP_DEXJOCO_EVAL_ENV"),
    "dexjoco_openpi_env": ("DEXJOCO_OPENPI_ENV", "TRAIN_EVAL_MLXP_DEXJOCO_OPENPI_ENV"),
    "data_pod_name": ("MLXP_DATA_POD", "TRAIN_EVAL_MLXP_DATA_POD"),
    "ddn_pvc": ("MLXP_DDN_PVC", "TRAIN_EVAL_MLXP_DDN_PVC"),
    "image": ("MLXP_IMAGE", "TRAIN_EVAL_MLXP_IMAGE"),
    "image_pull_secret": ("MLXP_IMAGE_PULL_SECRET", "TRAIN_EVAL_MLXP_IMAGE_PULL_SECRET"),
    "zone": ("MLXP_ZONE", "TRAIN_EVAL_MLXP_ZONE"),
    "wandb_secret": ("MLXP_WANDB_SECRET", "TRAIN_EVAL_MLXP_WANDB_SECRET"),
}

# Prefixes considered namespaced-and-safe to read from the process environment.
# Bare slurm-style names (USER, HF_HOME, DATA_DIR, …) are only honored in the
# cluster env FILE, never from os.environ, so a stray shell export can't shadow
# the saved config.
_PREFIXED = ("MLXP_", "TRAIN_EVAL_MLXP_")


def _load_saved() -> dict[str, Any]:
    if not _SETTINGS_FILE.is_file():
        return {}
    try:
        data = json.loads(_SETTINGS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _cluster_env_values() -> dict[str, str]:
    try:
        return cluster_settings.parse_env_text(cluster_settings.load_env_text("mlxp"))
    except FileNotFoundError:
        return {}


def _overrides_from(
    values: dict[str, str] | os._Environ[str], *, allow_plain: bool
) -> dict[str, Any]:
    """Resolve field overrides from ``values``. Each field takes the first of
    its candidate env names present and non-empty. With ``allow_plain`` false,
    only prefixed names (MLXP_*, TRAIN_EVAL_MLXP_*) are considered."""
    out: dict[str, Any] = {}
    for field_name, names in _FIELD_ENV_NAMES.items():
        for name in names:
            if not allow_plain and not name.startswith(_PREFIXED):
                continue
            raw = values.get(name)
            if raw is None or raw == "":
                continue
            if field_name == "gpus_per_node":
                try:
                    out[field_name] = int(raw)
                except ValueError:
                    continue
                break
            out[field_name] = raw
            break
    return out


def get_settings() -> MlxpSettings:
    saved = _load_saved()
    # Cluster env FILE accepts plain slurm-style names; the process environment
    # only namespaced ones. Process env wins over file, file over derived
    # defaults — matching the historical defaults < file < env order.
    overrides = {
        **_overrides_from(_cluster_env_values(), allow_plain=True),
        **_overrides_from(os.environ, allow_plain=False),
    }
    user = str(overrides.get("user") or saved.get("user") or _default_user())
    ddn_mount = str(overrides.get("ddn_mount") or _DEFAULT_DDN_MOUNT)
    ddn_home = str(overrides.get("ddn_user_home") or f"{ddn_mount}/{user}")
    data = _defaults_for(user, ddn_mount, ddn_home)
    data.update(overrides)
    return MlxpSettings.model_validate(data)


def save_user(user: str) -> MlxpSettings:
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps({"user": user}, indent=2) + "\n")
    return get_settings()


def labels(settings: MlxpSettings | None = None) -> dict[str, str]:
    s = settings or get_settings()
    return {"owner": s.user, "tool": TOOL_LABEL}


def owner_selector(settings: MlxpSettings | None = None) -> str:
    s = settings or get_settings()
    return f"owner={s.user},tool={TOOL_LABEL}"
