"""Per-submission output namespace helpers."""

from __future__ import annotations

import re
import uuid
from datetime import datetime


_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def make_output_namespace(job_name: str) -> str:
    """Return a unique, path-safe namespace for one concrete submission."""
    matches = re.findall(r"\d{8}_\d{6}", job_name)
    stamp = matches[-1] if matches else datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


def validate_output_namespace(value: str) -> str:
    namespace = value.strip()
    if not namespace:
        raise ValueError("output namespace cannot be empty")
    if not _SAFE_SEGMENT_RE.fullmatch(namespace):
        raise ValueError(
            "output namespace may only contain letters, numbers, dot, underscore, or hyphen"
        )
    return namespace
