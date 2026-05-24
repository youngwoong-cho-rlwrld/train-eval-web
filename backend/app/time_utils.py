"""Scheduler timestamp normalization."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .ssh import ssh_run


KST = timezone(timedelta(hours=9), name="KST")
_TZ_CACHE: dict[str, timezone] = {}


def _parse_offset(value: str) -> timezone | None:
    raw = value.strip()
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", raw)
    if not m:
        return None
    sign = 1 if m.group(1) == "+" else -1
    hours = int(m.group(2))
    minutes = int(m.group(3))
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


async def scheduler_timezone(host: str) -> timezone:
    if host in _TZ_CACHE:
        return _TZ_CACHE[host]
    try:
        r = await ssh_run(host, "date +%z", timeout=8.0)
    except Exception:
        r = None
    tz = _parse_offset(r.stdout.strip() if r and r.returncode == 0 else "") or timezone.utc
    _TZ_CACHE[host] = tz
    return tz


def to_kst_iso(value: str | None, source_tz: timezone | None = None) -> str | None:
    raw = (value or "").strip()
    if not raw or raw in {"Unknown", "None", "N/A"}:
        return None
    normalized = raw.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if re.search(r"[+-]\d{2}:?\d{2}$", normalized):
        # datetime.fromisoformat accepts +0900 in recent Python, but normalize
        # to +09:00 for older edge cases and consistent output.
        normalized = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", normalized)
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return raw
    else:
        parsed = None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                parsed = datetime.strptime(normalized, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return raw
        dt = parsed.replace(tzinfo=source_tz or timezone.utc)
    return dt.astimezone(KST).isoformat(timespec="seconds")
