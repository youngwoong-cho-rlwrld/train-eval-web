"""Cluster discovery + cluster.env parsing.

A cluster.env is a bash file with `export FOO=bar` lines. We `source` it in a
subprocess and capture the exported variables via `declare -p`. This handles
quoting / variable expansion correctly without us re-implementing bash.
"""
from __future__ import annotations


import asyncio
import re
import shutil
import subprocess

from pydantic import BaseModel

from . import cluster_settings


def _find_bash() -> str:
    """Find bash ≥ 4. macOS ships 3.2 which emits a non-structured `declare -p`."""
    for cand in ["/opt/homebrew/bin/bash", "/usr/local/bin/bash", shutil.which("bash")]:
        if cand and _bash_version(cand) >= 4:
            return cand
    raise RuntimeError(
        "bash ≥ 4 required. Install via `brew install bash`. macOS's /bin/bash is 3.2."
    )


def _bash_version(path: str) -> int:
    try:
        out = subprocess.check_output([path, "-c", "echo $BASH_VERSINFO"], text=True)
        return int(out.split()[0])
    except Exception:
        return 0


_BASH = _find_bash()


class ClusterEnv(BaseModel):
    name: str
    vars: dict[str, str]

    @property
    def ssh_alias(self) -> str:
        """SSH alias to use. Can be overridden with SSH_ALIAS in the cluster env."""
        alias = (self.vars.get("SSH_ALIAS") or "").strip()
        if alias:
            return alias
        match self.name:
            case "kakao":
                return "kakao-login-1"
            case other:
                return other


def list_clusters() -> list[str]:
    return cluster_settings.list_cluster_names()


async def load_cluster(name: str) -> ClusterEnv:
    text = cluster_settings.load_env_text(name)
    if not text.strip():
        raise FileNotFoundError(f"Cluster env for {name} is not configured")
    vars = await _source_and_dump(text)
    if name != "mlxp":
        missing = [k for k in ("PARTITION", "LOG_DIR", "DATA_DIR") if not vars.get(k)]
        if missing:
            raise FileNotFoundError(
                f"Cluster env for {name} is missing required values: {', '.join(missing)}"
            )
    return ClusterEnv(name=name, vars=vars)


# `declare -p` emits lines like:
#   declare -x FOO="bar"
#   declare -ax DATASETS=([0]="a|b|1.0" [1]="c|d|1.0")
# We only capture string scalars for cluster.env (no arrays expected there).
_SCALAR_RE = re.compile(r'^declare -[a-zA-Z\-]+ ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')


async def _source_and_dump(script_text: str) -> dict[str, str]:
    """Source a bash snippet and return the resulting `declare -p` exports."""
    cmd = f"set -a\n{script_text}\nset +a\ndeclare -p"
    proc = await asyncio.create_subprocess_exec(
        _BASH, "-c", cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"bash failed: {stderr.decode()}")
    out: dict[str, str] = {}
    for line in stdout.decode().splitlines():
        m = _SCALAR_RE.match(line)
        if m:
            out[m.group(1)] = _bash_unescape(m.group(2))
    return out


def _bash_unescape(s: str) -> str:
    """Reverse bash's `declare -p` quoting: \" → ", \\ → \\, \\$ → $.

    Decode in a single left-to-right scan so overlapping escapes (e.g. an
    escaped backslash immediately followed by an escaped dollar) decode
    unambiguously: a backslash introduces an escape and the next char is
    emitted literally.
    """
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            out.append(s[i + 1])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)
