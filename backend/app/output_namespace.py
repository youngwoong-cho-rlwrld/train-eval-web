"""Per-submission output namespace helpers."""

from __future__ import annotations

import re
import uuid

from .submission_snapshot import snapshot_suffix


_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_UNSAFE_SEGMENT_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_segment(value: str | None, fill: str = "_") -> str:
    if not value:
        return ""
    return _UNSAFE_SEGMENT_CHARS_RE.sub(fill, value.strip()).strip("._-")


def make_output_namespace(job_name: str, experiment: str | None = None) -> str:
    """Return a unique, path-safe namespace for one concrete submission."""
    suffix = f"{snapshot_suffix(job_name)}_{uuid.uuid4().hex[:6]}"
    prefix = _safe_segment(experiment)
    return f"{prefix}_{suffix}" if prefix else suffix


def validate_output_namespace(value: str) -> str:
    namespace = value.strip()
    if not namespace:
        raise ValueError("output namespace cannot be empty")
    if not _SAFE_SEGMENT_RE.fullmatch(namespace):
        raise ValueError(
            "output namespace may only contain letters, numbers, dot, underscore, or hyphen"
        )
    return namespace
