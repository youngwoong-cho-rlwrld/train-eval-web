"""Shared train-time override resolution."""

from __future__ import annotations

from typing import Any

from .data_interface import load_data_interface_for_variant
from .training_models import (
    TrainingModel,
    rewrites_modality_action_horizon,
)
from .variant_values import variant_int_opt


def resolve_train_action_horizon(
    *,
    variant: Any,
    model: TrainingModel,
    action_horizon_mode: str | None = None,
    requested: int | None = None,
) -> int | None:
    """Resolve and validate the action horizon for train submissions.

    Model registry entries decide how a resolved horizon is applied. Plain
    GR00T N1.6 only stages a matching modality config; PhysiXel can also pass
    the value to model code through --action-horizon.
    """
    mode = action_horizon_mode or model.action_horizon_mode
    if mode == "none":
        return None

    modality_horizon = load_data_interface_for_variant(variant).action_horizon
    config_horizon = variant_int_opt(variant, "TRAIN_ACTION_HORIZON")
    action_horizon = requested if requested is not None else config_horizon or modality_horizon

    if action_horizon is None:
        return None
    if action_horizon <= 0:
        raise ValueError(f"action horizon must be positive, got {action_horizon}")
    if (
        not rewrites_modality_action_horizon(mode)
        and modality_horizon is not None
        and action_horizon != modality_horizon
    ):
        source = "requested" if requested is not None else "configured"
        raise ValueError(
            f"{source} action horizon {action_horizon} does not match "
            f"modality action horizon {modality_horizon}; use a matching "
            "TRAIN_MODALITY_CONFIG for clean ablations"
        )
    return action_horizon


def validate_global_batch_divisible(model_family: str, global_batch_size: int | None, num_gpus: int) -> None:
    """n1.5 training requires the global batch size to divide evenly across GPUs."""
    if model_family == "n1.5" and global_batch_size is not None and num_gpus and global_batch_size % num_gpus != 0:
        raise ValueError("global_batch_size must be divisible by num_gpus for n1.5 training")
