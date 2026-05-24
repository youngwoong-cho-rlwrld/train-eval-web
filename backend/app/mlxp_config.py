"""Central MLXP/Kubernetes defaults used by backend modules.

Environment variables can override every user/site-specific value so the
application logic does not depend on one person's paths or labels.
"""

from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


MLXP_USER = _env("TRAIN_EVAL_MLXP_USER", os.environ.get("USER", "youngwoong"))
NAMESPACE = _env("TRAIN_EVAL_MLXP_NAMESPACE", "p-rlwrld")
OWNER_LABEL = _env("TRAIN_EVAL_MLXP_OWNER", MLXP_USER)
TOOL_LABEL = _env("TRAIN_EVAL_MLXP_TOOL_LABEL", "train-eval-web")

DEFAULT_NODE = _env("TRAIN_EVAL_MLXP_NODE", "h200-03-w-3c55")
GPU_NODE_PREFIX = _env("TRAIN_EVAL_MLXP_GPU_NODE_PREFIX", "h200-")
GPUS_PER_NODE = _env_int("TRAIN_EVAL_MLXP_GPUS_PER_NODE", 8)

DDN_MOUNT = _env("TRAIN_EVAL_MLXP_DDN_MOUNT", "/data")
DDN_USER_HOME = _env("TRAIN_EVAL_MLXP_HOME", f"{DDN_MOUNT}/{MLXP_USER}")
DATASETS_DIR = _env("TRAIN_EVAL_MLXP_DATASETS_DIR", f"{DDN_USER_HOME}/datasets")
EXPERIMENTS_DIR = _env("TRAIN_EVAL_MLXP_EXPERIMENTS_DIR", f"{DDN_USER_HOME}/experiments")
HF_HOME = _env("TRAIN_EVAL_MLXP_HF_HOME", f"{DDN_USER_HOME}/.cache/huggingface")

DATA_POD_NAME = _env("TRAIN_EVAL_MLXP_DATA_POD", f"{MLXP_USER}-data-pod")
DDN_PVC = _env("TRAIN_EVAL_MLXP_DDN_PVC", "ddn-rlwrld-shared")
IMAGE = _env("TRAIN_EVAL_MLXP_IMAGE", "mlxp.kr.ncr.ntruss.com/rlwrld-gpu-base:latest")
IMAGE_PULL_SECRET = _env("TRAIN_EVAL_MLXP_IMAGE_PULL_SECRET", "mlxp-registry")
ZONE = _env("TRAIN_EVAL_MLXP_ZONE", "private-h200-rlwrld-0")
WANDB_SECRET = _env("TRAIN_EVAL_MLXP_WANDB_SECRET", f"{MLXP_USER}-wandb")


def labels() -> dict[str, str]:
    return {"owner": OWNER_LABEL, "tool": TOOL_LABEL}


def owner_selector() -> str:
    return f"owner={OWNER_LABEL}"
