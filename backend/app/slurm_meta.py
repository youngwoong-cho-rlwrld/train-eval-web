"""Helpers for Slurm submit sidecar metadata."""

import asyncio
import shlex

from .ssh import ssh_run


def parse_meta_lines(text: str | None) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not text:
        return fields
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


async def read_slurm_meta(host: str, job_id: str) -> dict[str, str]:
    """Read `~/.train-eval-web/jobs/<job_id>.meta` written by submit.py."""
    try:
        r = await ssh_run(
            host,
            f"cat $HOME/.train-eval-web/jobs/{shlex.quote(job_id)}.meta 2>/dev/null",
            timeout=10.0,
        )
    except (asyncio.TimeoutError, OSError):
        return {}
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    return parse_meta_lines(r.stdout)


async def read_slurm_meta_many(host: str, job_ids: list[str]) -> dict[str, dict[str, str]]:
    """Bulk-read Slurm sidecar metadata for job list pages.

    This avoids spawning one SSH process per row in `/api/jobs`; slow clusters
    should degrade to missing metadata rather than failing the entire request.
    """
    unique_ids = sorted({str(job_id) for job_id in job_ids if str(job_id)})
    if not unique_ids:
        return {}
    marker = "__TRAIN_EVAL_WEB_META__"
    quoted_ids = " ".join(shlex.quote(job_id) for job_id in unique_ids)
    cmd = (
        f"for job_id in {quoted_ids}; do "
        'path="$HOME/.train-eval-web/jobs/${job_id}.meta"; '
        'if [ -s "$path" ]; then '
        f'printf "{marker}%s\\n" "$job_id"; '
        'cat "$path"; '
        "fi; "
        "done"
    )
    try:
        r = await ssh_run(host, cmd, timeout=15.0)
    except (asyncio.TimeoutError, OSError):
        return {}
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    return parse_meta_blocks(r.stdout, marker)


def parse_meta_blocks(text: str, marker: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    current_job_id: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(marker):
            if current_job_id is not None:
                out[current_job_id] = parse_meta_lines("\n".join(current_lines))
            current_job_id = line.removeprefix(marker).strip()
            current_lines = []
            continue
        if current_job_id is not None:
            current_lines.append(line)
    if current_job_id is not None:
        out[current_job_id] = parse_meta_lines("\n".join(current_lines))
    return out
