"""Effective DexJoCo client rollout settings shared by preview and submit.

The policy checkpoint's action chunk and the evaluator's execution policy are
separate concerns.  These helpers validate the latter without interpreting it
as a train-time action horizon.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# Canonical client semantics:
#   sync             execute a complete returned chunk before requesting again
#   blocking_overlap preserve the historical thresholded blocking replan loop
#   async            request replacement chunks without blocking environment steps
INFERENCE_MODES = frozenset({"async", "sync", "blocking_overlap"})
DEFAULT_INFERENCE_MODE = "sync"
DEFAULT_ACTION_HORIZON = "auto"
DEFAULT_REPLAN_RATIO = "0.8"


@dataclass(frozen=True)
class DexjocoRollout:
    inference_mode: str
    action_horizon: str
    replan_ratio: str

    def metadata(self) -> dict[str, str]:
        return asdict(self)


def rollout_for_variant(variant: Any) -> DexjocoRollout:
    vars_ = getattr(variant, "vars", {}) or {}
    mode = (vars_.get("DEXJOCO_INFERENCE_MODE") or DEFAULT_INFERENCE_MODE).strip()
    horizon = (vars_.get("DEXJOCO_ACTION_HORIZON") or DEFAULT_ACTION_HORIZON).strip()
    ratio = (vars_.get("DEXJOCO_REPLAN_RATIO") or DEFAULT_REPLAN_RATIO).strip()

    if mode not in INFERENCE_MODES:
        raise ValueError(
            "DEXJOCO_INFERENCE_MODE must be one of "
            f"{', '.join(sorted(INFERENCE_MODES))}, got {mode!r}"
        )
    if horizon != "auto":
        try:
            horizon_int = int(horizon)
        except ValueError as exc:
            raise ValueError(
                "DEXJOCO_ACTION_HORIZON must be a positive integer or 'auto', "
                f"got {horizon!r}"
            ) from exc
        if horizon_int <= 0:
            raise ValueError(
                "DEXJOCO_ACTION_HORIZON must be a positive integer or 'auto', "
                f"got {horizon!r}"
            )
    try:
        ratio_float = float(ratio)
    except ValueError as exc:
        raise ValueError(
            f"DEXJOCO_REPLAN_RATIO must be in [0, 1], got {ratio!r}"
        ) from exc
    if not 0.0 <= ratio_float <= 1.0:
        raise ValueError(f"DEXJOCO_REPLAN_RATIO must be in [0, 1], got {ratio!r}")

    return DexjocoRollout(mode, horizon, ratio)
