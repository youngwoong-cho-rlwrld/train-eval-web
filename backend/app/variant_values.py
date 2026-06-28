"""Helpers for reading typed values from loaded variant configs."""

from __future__ import annotations

from typing import Any


def _coerce_int(variant: Any, key: str, raw: Any) -> int | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"variant {variant.name}: {key} must be an integer")


def variant_int(variant: Any, key: str, default: int) -> int:
    raw = (getattr(variant, "vars", None) or {}).get(key)
    value = _coerce_int(variant, key, raw)
    return default if value is None else value


def variant_int_opt(variant: Any, key: str) -> int | None:
    raw = (getattr(variant, "vars", None) or {}).get(key)
    return _coerce_int(variant, key, raw)
