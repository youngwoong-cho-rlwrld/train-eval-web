"""Shared train-time override resolution."""

from __future__ import annotations

from typing import Any

from .data_interface import load_data_interface_for_variant
from .training_models import (
    TrainingModel,
    rewrites_modality_action_horizon,
)


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
    if model.family != "n1.6" or mode == "none":
        return None

    modality_horizon = load_data_interface_for_variant(variant).action_horizon
    config_horizon = _variant_action_horizon(variant)
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


def _variant_action_horizon(variant: Any) -> int | None:
    raw = (getattr(variant, "vars", {}) or {}).get("TRAIN_ACTION_HORIZON", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"variant {variant.name}: TRAIN_ACTION_HORIZON must be an integer")
