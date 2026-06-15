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


def _default_user() -> str:
    return os.environ.get("TRAIN_EVAL_MLXP_USER") or os.environ.get("USER") or "youngwoong"


def _defaults_for(user: str | None = None) -> dict[str, Any]:
    u = user or _default_user()
    ddn_mount = "/data"
    ddn_home = f"{ddn_mount}/{u}"
    return {
        "user": u,
        "namespace": "p-rlwrld",
        "owner_label": u,
        "tool_label": "train-eval-web",
        "default_node": "",
        "gpu_node_prefix": "h200-",
        "gpu_type": "H200",
        "gpus_per_node": 8,
        "ddn_mount": ddn_mount,
        "ddn_user_home": ddn_home,
        "datasets_dir": f"{ddn_home}/datasets",
        "experiments_dir": f"{ddn_home}/experiments",
        "hf_home": f"{ddn_home}/.cache/huggingface",
        "workspace_dir": f"{ddn_home}/workspace",
        "isaac_dir": f"{ddn_home}/workspace/rlwrld_isaac",
        "data_pod_name": f"{u}-data-pod",
        "ddn_pvc": "ddn-rlwrld-shared",
        "image": "mlxp.kr.ncr.ntruss.com/rlwrld-gpu-base:latest",
        "image_pull_secret": "mlxp-registry",
        "zone": "private-h200-rlwrld-0",
        "wandb_secret": f"{u}-wandb",
    }


class MlxpSettings(BaseModel):
    user: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    owner_label: str = Field(min_length=1)
    tool_label: str = Field(min_length=1)
    # Optional legacy pin; queue-class scheduling made a per-user default
    # node obsolete. Kept for back-compat with saved TRAIN_EVAL_MLXP_NODE.
    default_node: str = ""
    gpu_node_prefix: str = ""
    gpu_type: str = "GPU"
    gpus_per_node: int = Field(default=8, ge=1)
    ddn_mount: str = Field(min_length=1)
    ddn_user_home: str = Field(min_length=1)
    datasets_dir: str = Field(min_length=1)
    experiments_dir: str = Field(min_length=1)
    hf_home: str = Field(min_length=1)
    workspace_dir: str = Field(min_length=1)
    isaac_dir: str = Field(min_length=1)
    data_pod_name: str = Field(min_length=1)
    ddn_pvc: str = Field(min_length=1)
    image: str = Field(min_length=1)
    image_pull_secret: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    wandb_secret: str = Field(min_length=1)


class MlxpSettingsUpdate(BaseModel):
    user: str = Field(min_length=1)


_ENV_FIELDS = {
    "TRAIN_EVAL_MLXP_USER": "user",
    "TRAIN_EVAL_MLXP_NAMESPACE": "namespace",
    "TRAIN_EVAL_MLXP_OWNER": "owner_label",
    "TRAIN_EVAL_MLXP_TOOL_LABEL": "tool_label",
    "TRAIN_EVAL_MLXP_NODE": "default_node",
    "TRAIN_EVAL_MLXP_GPU_NODE_PREFIX": "gpu_node_prefix",
    "TRAIN_EVAL_MLXP_GPU_TYPE": "gpu_type",
    "TRAIN_EVAL_MLXP_GPUS_PER_NODE": "gpus_per_node",
    "TRAIN_EVAL_MLXP_DDN_MOUNT": "ddn_mount",
    "TRAIN_EVAL_MLXP_HOME": "ddn_user_home",
    "TRAIN_EVAL_MLXP_DATASETS_DIR": "datasets_dir",
    "TRAIN_EVAL_MLXP_EXPERIMENTS_DIR": "experiments_dir",
    "TRAIN_EVAL_MLXP_HF_HOME": "hf_home",
    "TRAIN_EVAL_MLXP_WORKSPACE_DIR": "workspace_dir",
    "TRAIN_EVAL_MLXP_ISAAC_DIR": "isaac_dir",
    "TRAIN_EVAL_MLXP_DATA_POD": "data_pod_name",
    "TRAIN_EVAL_MLXP_DDN_PVC": "ddn_pvc",
    "TRAIN_EVAL_MLXP_IMAGE": "image",
    "TRAIN_EVAL_MLXP_IMAGE_PULL_SECRET": "image_pull_secret",
    "TRAIN_EVAL_MLXP_ZONE": "zone",
    "TRAIN_EVAL_MLXP_WANDB_SECRET": "wandb_secret",
}


def _load_saved() -> dict[str, Any]:
    if not _SETTINGS_FILE.is_file():
        return {}
    try:
        data = json.loads(_SETTINGS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _env_overrides() -> dict[str, Any]:
    return _coerce_overrides(os.environ)


def _cluster_env_overrides() -> dict[str, Any]:
    try:
        values = cluster_settings.parse_env_text(cluster_settings.load_env_text("mlxp"))
    except FileNotFoundError:
        return {}
    return _coerce_overrides(values)


def _coerce_overrides(values: dict[str, str] | os._Environ[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_name, field_name in _ENV_FIELDS.items():
        raw = values.get(env_name)
        if raw is None or raw == "":
            continue
        if field_name == "gpus_per_node":
            try:
                out[field_name] = int(raw)
            except ValueError:
                continue
        else:
            out[field_name] = raw
    return out


def get_settings() -> MlxpSettings:
    saved = _load_saved()
    cluster_env = _cluster_env_overrides()
    user = str(cluster_env.get("user") or saved.get("user") or _default_user())
    data = _defaults_for(user)
    data.update(cluster_env)
    data.update(_env_overrides())
    return MlxpSettings.model_validate(data)


def save_user(user: str) -> MlxpSettings:
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps({"user": user}, indent=2) + "\n")
    return get_settings()


def labels(settings: MlxpSettings | None = None) -> dict[str, str]:
    s = settings or get_settings()
    return {"owner": s.owner_label, "tool": s.tool_label}


def owner_selector(settings: MlxpSettings | None = None) -> str:
    s = settings or get_settings()
    return f"owner={s.owner_label},tool={s.tool_label}"
