"""Cluster dataset enumeration.

Each call lists everything under a configurable directory that contains
a `meta/info.json` (LeRobot v2.1 shape) and pulls resolution + episode
count from it.

- For slurm clusters, we hop through SSH and run the listing on the
  cluster host directly.
- For MLXP we `kubectl exec` into any running owned pod
  with the DDN PVC mounted. The data-pod or a training pod both work
  as long as something is alive on the cluster.

The directory to scan is supplied per request by the caller (the
frontend persists it in localStorage). When omitted we fall back to
`~/datasets/` for slurm and the configured MLXP datasets dir for MLXP.
"""
from __future__ import annotations


import asyncio
import shlex

from pydantic import BaseModel

from .clusters import load_cluster
from .mlxp_config import get_settings
from .mlxp_data_pod import ensure_listing_pod
from .ssh import ssh_run


class DatasetInfo(BaseModel):
    name: str
    path: str            # absolute path on cluster
    height: int | None
    width: int | None
    episodes: int | None
    codec: str | None


SLURM_DEFAULT_DIR = "~/datasets"


# A single python -c is more robust than a bash loop for parsing JSON
# and avoiding quoting hell over ssh / kubectl exec.
def _list_py(dataset_dir: str) -> str:
    # `dataset_dir` is interpolated as a quoted string literal inside python.
    # Escape backslashes and single quotes so the embedded string is safe.
    safe = dataset_dir.replace("\\", "\\\\").replace("'", "\\'")
    return rf"""
import os, json, glob
base = os.path.expanduser('{safe}')
for p in sorted(glob.glob(os.path.join(base, '*/meta/info.json'))):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    parts = p.split('/')
    name = parts[-3]
    path = '/'.join(parts[:-2])
    v = next((f for f in d.get('features', {{}}).values() if f.get('dtype')=='video'), None)
    if v:
        shape = v.get('shape') or [None, None, None]
        h = shape[1] if len(shape) >= 2 else None
        w = shape[2] if len(shape) >= 3 else None
        codec = (v.get('info') or {{}}).get('video.codec')
    else:
        h = w = codec = None
    eps = d.get('total_episodes')
    print('|'.join([
        name, path,
        str(h) if h is not None else '',
        str(w) if w is not None else '',
        str(eps) if eps is not None else '',
        codec or '',
    ]))
"""


def _parse_lines(text: str) -> list[DatasetInfo]:
    out: list[DatasetInfo] = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 6:
            continue
        name, path, h, w, eps, codec = parts
        out.append(DatasetInfo(
            name=name,
            path=path,
            height=int(h) if h else None,
            width=int(w) if w else None,
            episodes=int(eps) if eps else None,
            codec=codec or None,
        ))
    return out


async def list_datasets(cluster: str, path: str | None = None) -> list[DatasetInfo]:
    if cluster == "mlxp":
        settings = get_settings()
        return await _list_datasets_mlxp(path or settings.datasets_dir)
    return await _list_datasets_slurm(cluster, path or SLURM_DEFAULT_DIR)


async def _list_datasets_slurm(cluster: str, dir_path: str) -> list[DatasetInfo]:
    env = await load_cluster(cluster)
    script = _list_py(dir_path)
    r = await ssh_run(env.ssh_alias, f"python3 -c {shlex.quote(script)}", timeout=30.0)
    if r.returncode != 0:
        raise RuntimeError(f"list_datasets({cluster}, {dir_path}) failed: {r.stderr}")
    return _parse_lines(r.stdout)


async def _list_datasets_mlxp(dir_path: str) -> list[DatasetInfo]:
    """Enumerate datasets on MLXP DDN via kubectl exec.

    `ensure_listing_pod()` reuses any running pod with the DDN PVC
    mounted (training pod or data pod) and provisions a fresh data
    pod if nothing is up — the first call after a quiet period can
    take ~30-90s while the pod schedules.
    """
    pod = await ensure_listing_pod()
    settings = get_settings()

    script = _list_py(dir_path)
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "exec", "-n", settings.namespace, pod, "--",
        "python3", "-c", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"list_datasets(mlxp, {dir_path}) timed out")
    if proc.returncode != 0:
        raise RuntimeError(
            f"list_datasets(mlxp, {dir_path}) failed in pod {pod}: "
            f"{stderr.decode(errors='replace').strip()}"
        )
    return _parse_lines(stdout.decode())
