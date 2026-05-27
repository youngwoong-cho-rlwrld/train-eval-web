"""Remote path probes shared by API routes and checkpoint copy bookkeeping."""

from __future__ import annotations

import asyncio
import shlex
from typing import Literal

from .clusters import load_cluster
from .mlxp_config import get_settings as get_mlxp_settings
from .mlxp_data_pod import ensure_listing_pod
from .ssh import ssh_run


PathKind = Literal["dir", "file"]


async def remote_path_kind(cluster: str, path: str, timeout: float = 15.0) -> PathKind | None:
    """Return dir/file for a remote path, or None when it is absent or unsupported."""
    target = path.strip()
    if not target:
        return None
    kind = await _remote_path_probe(cluster, target, _kind_script(target), timeout)
    return kind if kind in ("dir", "file") else None


async def remote_path_exists(cluster: str, path: str, timeout: float = 15.0) -> bool | None:
    """Return whether a remote path exists; None means the probe itself failed."""
    target = path.strip()
    if not target:
        return False
    try:
        out = await _remote_path_probe(cluster, target, _exists_script(target), timeout)
    except Exception:
        return None
    return out == "1"


async def _remote_path_probe(
    cluster: str,
    path: str,
    script: str,
    timeout: float,
) -> str:
    if cluster == "mlxp":
        settings = get_mlxp_settings()
        pod = await ensure_listing_pod()
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            "exec",
            "-n",
            settings.namespace,
            pod,
            "--",
            "bash",
            "-lc",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace").strip() or f"path probe failed: {path}")
        return stdout.decode(errors="replace").strip()

    env = await load_cluster(cluster)
    r = await ssh_run(env.ssh_alias, script, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip() or f"path probe failed: {path}")
    return r.stdout.strip()


def _exists_script(path: str) -> str:
    p = shlex.quote(path)
    return f"if [ -e {p} ]; then echo 1; else echo 0; fi"


def _kind_script(path: str) -> str:
    p = shlex.quote(path)
    return f"if [ -d {p} ]; then echo dir; elif [ -f {p} ]; then echo file; else echo none; fi"
