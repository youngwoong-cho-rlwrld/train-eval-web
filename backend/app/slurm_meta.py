"""Helpers for Slurm submit sidecar metadata."""

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
    r = await ssh_run(
        host,
        f"cat $HOME/.train-eval-web/jobs/{job_id}.meta 2>/dev/null",
        timeout=10.0,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    return parse_meta_lines(r.stdout)
