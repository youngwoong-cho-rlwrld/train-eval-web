"""Cluster dataset enumeration.

Lists everything under $HOME/datasets/ that has a meta/info.json
(LeRobot v2.1 shape) and pulls resolution + episode count from it. We
hop through SSH so the listing reflects the actual cluster filesystem,
not the local Mac repo.
"""

from typing import Any

from pydantic import BaseModel

from .clusters import load_cluster
from .ssh import ssh_run


class DatasetInfo(BaseModel):
    name: str
    path: str            # absolute path on cluster
    height: int | None
    width: int | None
    episodes: int | None
    codec: str | None


# A single python -c is more robust than a bash loop for parsing JSON
# and avoiding quoting hell over ssh.
_LIST_PY = r"""
import os, json, glob, sys
out = []
for p in sorted(glob.glob(os.path.expanduser("~/datasets/*/meta/info.json"))):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    name = p.split("/")[-3]
    path = "/".join(p.split("/")[:-2])
    v = next((f for f in d.get("features", {}).values() if f.get("dtype")=="video"), None)
    if v:
        shape = v.get("shape") or [None, None, None]
        h = shape[1] if len(shape) >= 2 else None
        w = shape[2] if len(shape) >= 3 else None
        codec = (v.get("info") or {}).get("video.codec")
    else:
        h = w = codec = None
    eps = d.get("total_episodes")
    parts = [name, path, str(h) if h is not None else "",
             str(w) if w is not None else "",
             str(eps) if eps is not None else "",
             codec or ""]
    print("|".join(parts))
"""


async def list_datasets(cluster: str) -> list[DatasetInfo]:
    env = await load_cluster(cluster)
    r = await ssh_run(env.ssh_alias, f"python3 -c {_quote(_LIST_PY)}", timeout=30.0)
    if r.returncode != 0:
        raise RuntimeError(f"list_datasets failed: {r.stderr}")
    out: list[DatasetInfo] = []
    for line in r.stdout.splitlines():
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


def _quote(s: str) -> str:
    """Shell-quote a string for inline use in `python3 -c '...'`."""
    # We use single quotes around the python and replace any embedded ' with '"'"'.
    return "'" + s.replace("'", "'\"'\"'") + "'"
