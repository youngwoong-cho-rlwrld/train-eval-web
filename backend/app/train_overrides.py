"""Shared train-time override resolution."""

from __future__ import annotations

from typing import Any

from .data_interface import load_data_interface_for_variant


def resolve_train_action_horizon(
    *,
    variant: Any,
    model_family: str,
    requested: int | None = None,
) -> int | None:
    """Resolve and validate the action horizon for N1.6 training.

    The modality config determines the data action delta length. The value
    passed to the training entrypoint must match that length so model,
    action-head, processor, and data horizon stay aligned.
    """
    if model_family != "n1.6":
        return None

    modality_horizon = load_data_interface_for_variant(variant).action_horizon
    config_horizon = _variant_action_horizon(variant)
    action_horizon = requested if requested is not None else config_horizon or modality_horizon

    if action_horizon is None:
        return None
    if action_horizon <= 0:
        raise ValueError(f"action horizon must be positive, got {action_horizon}")
    if modality_horizon is not None and action_horizon != modality_horizon:
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
