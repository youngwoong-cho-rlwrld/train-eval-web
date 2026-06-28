"""Shared helpers for recovering a job's phase and experiment variant."""

from __future__ import annotations

import re

from .variants import list_variants


def parse_comment_fields(comment: str | None) -> dict[str, str]:
    """Parse the semicolon-delimited scheduler Comment metadata."""
    fields: dict[str, str] = {}
    if not comment:
        return fields
    for chunk in comment.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def comment_field_fragment(fields: dict[str, str | None]) -> str:
    """Render scheduler Comment metadata without a leading semicolon."""
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        rendered = str(value)
        if not rendered:
            continue
        parts.append(f"{key}={rendered}")
    return ";".join(parts)


def _match_known_variant(candidate: str) -> str:
    """Normalize variant slugs from all historical job-name conventions."""
    try:
        variants = list_variants()
    except Exception:
        variants = []
    variants_by_len = sorted(variants, key=len, reverse=True)

    candidates = [candidate]
    parts = candidate.split("_")
    # Legacy Slurm names appended `<cluster>_<partition>` after the variant:
    # train_n15_cube_stack_3cm_right_kakao_background_20260514_...
    if len(parts) > 2:
        candidates.append("_".join(parts[:-2]))

    for c in candidates:
        if not c:
            continue
        for variant in variants_by_len:
            if c == variant or c.startswith(f"{variant}_"):
                return variant
        # Older job names predate the explicit 480 suffix, but the current
        # repo/cluster variant directories carry it.
        if f"{c}_480" in variants:
            return f"{c}_480"

    return candidate


def parse_phase_and_variant(job_name: str) -> tuple[str, str | None]:
    """Pull (phase, variant) out of a display name.

    Supported shapes:
      - `{phase}_{variant}_{YYYYMMDD}_{HHMMSS}`
      - `{prefix}_{phase}_{variant}_{YYYYMMDD}_{HHMMSS}`
      - legacy `{phase}_{variant}_{cluster}_{partition}_{YYYYMMDD}_{HHMMSS}`

    The variant itself contains underscores, so we first strip the trailing
    timestamp, then use the known local variant names to remove legacy suffixes.
    """
    m = re.match(r"^(.+)_(\d{8}_\d{6})$", job_name)
    if not m:
        return "unknown", None

    body = m.group(1)
    parts = body.split("_")
    for idx, part in enumerate(parts):
        if part in ("train", "resume", "eval") and idx + 1 < len(parts):
            candidate = "_".join(parts[idx + 1:])
            return part, _match_known_variant(candidate)
    return "unknown", None


def parse_comment_metadata(comment: str) -> tuple[str | None, str | None]:
    """Recover (phase, variant) from persisted submit metadata.

    Shape: `phase=<p>;variant=<v>`.
    """
    fields = parse_comment_fields(comment)
    phase = fields.get("phase")
    variant = fields.get("variant")
    if phase not in ("train", "resume", "eval"):
        phase = None
    return phase, variant


def resolve_phase_and_variant(job_name: str, record: dict | None = None) -> tuple[str, str | None]:
    """Prefer explicit metadata, fall back to parsing the job name."""
    if record:
        comment = record.get("Comment") or record.get("JobComment") or ""
        phase, variant = parse_comment_metadata(comment)
        if phase and variant:
            return phase, variant
    return parse_phase_and_variant(job_name)


def phase_variant_from_meta(fields: dict[str, str]) -> tuple[str | None, str | None]:
    phase = fields.get("phase")
    if phase not in ("train", "resume", "eval"):
        phase = None
    return phase, fields.get("variant")


def resolve_identity(
    meta: dict[str, str] | None,
    job_name: str,
    record: dict | None = None,
) -> tuple[str, str | None]:
    """Recover (phase, variant) using the canonical precedence:
    sidecar metadata -> scheduler Comment -> job name.

    Returns ("unknown", None) when nothing resolves the phase.
    """
    if meta:
        phase, variant = phase_variant_from_meta(meta)
        if phase and variant:
            return phase, variant
    return resolve_phase_and_variant(job_name, record)
