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
_REMOTE_HOME_CACHE: dict[str, str] = {}

# Leading shell tokens that denote the remote $HOME, and the same tokens as
# path prefixes. Shared by the home-token helpers below.
_HOME_TOKENS = ("$HOME", "${HOME}", "~")
_HOME_PREFIXES = ("$HOME/", "${HOME}/", "~/")


async def _kubectl_bash_lc(script: str, timeout: float) -> tuple[int, str, str]:
    """Run `bash -lc <script>` in the MLXP listing pod via kubectl exec.

    Returns (returncode, decoded stdout, decoded stderr). Shared by remote_home
    and _remote_path_probe so the kubectl-exec plumbing lives in one place.
    """
    settings = get_mlxp_settings()
    pod = await ensure_listing_pod()
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-n", settings.namespace, pod, "--", "bash", "-lc", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def remote_home(cluster: str) -> str | None:
    """Return the remote shell's HOME for a cluster."""
    cached = _REMOTE_HOME_CACHE.get(cluster)
    if cached:
        return cached

    try:
        if cluster == "mlxp":
            rc, out, _err = await _kubectl_bash_lc('printf "%s" "$HOME"', 10.0)
            home = out.strip()
            if rc == 0 and home.startswith("/"):
                _REMOTE_HOME_CACHE[cluster] = home.rstrip("/")
                return _REMOTE_HOME_CACHE[cluster]
            return None

        env = await load_cluster(cluster)
        r = await ssh_run(env.ssh_alias, 'printf "%s" "$HOME"', timeout=10.0)
    except Exception:
        return None

    home = r.stdout.strip()
    if r.returncode == 0 and home.startswith("/"):
        _REMOTE_HOME_CACHE[cluster] = home.rstrip("/")
        return _REMOTE_HOME_CACHE[cluster]
    return None


def expand_home_path(path: str | None, home: str | None) -> str | None:
    """Expand a leading remote-home token for display/copy responses."""
    if not path or not home:
        return path
    home = home.rstrip("/")
    if path in _HOME_TOKENS:
        return home
    for prefix in _HOME_PREFIXES:
        if path.startswith(prefix):
            return f"{home}/{path[len(prefix):]}"
    return path


async def expand_cluster_home(cluster: str, path: str | None) -> str | None:
    if not _has_home_token(path):
        return path
    return expand_home_path(path, await remote_home(cluster))


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
        rc, out, err = await _kubectl_bash_lc(script, timeout)
        if rc != 0:
            raise RuntimeError(err.strip() or f"path probe failed: {path}")
        return out.strip()

    env = await load_cluster(cluster)
    r = await ssh_run(env.ssh_alias, script, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip() or f"path probe failed: {path}")
    return r.stdout.strip()


def _exists_script(path: str) -> str:
    p = remote_shell_path(path)
    return f"if [ -e {p} ]; then echo 1; else echo 0; fi"


def _kind_script(path: str) -> str:
    p = remote_shell_path(path)
    return f"if [ -d {p} ]; then echo dir; elif [ -f {p} ]; then echo file; else echo none; fi"


def remote_shell_path(path: str) -> str:
    """Quote a remote path while preserving a leading shell HOME token."""
    if path in _HOME_TOKENS:
        return '"$HOME"'
    for prefix in _HOME_PREFIXES:
        if path.startswith(prefix):
            return '"$HOME"/' + shlex.quote(path[len(prefix):])
    return shlex.quote(path)


def remote_path_expr(path: str) -> str:
    """Render a path for a remote shell, passing a `$HOME/` prefix through.

    Distinct from ``remote_shell_path``: this only special-cases a literal
    ``$HOME/`` prefix (left unquoted so the shell expands it) and otherwise
    ``shlex.quote``s the whole path.
    """
    return path if path.startswith("$HOME/") else shlex.quote(path)


def _has_home_token(path: str | None) -> bool:
    if not path:
        return False
    return path in _HOME_TOKENS or path.startswith(_HOME_PREFIXES)
