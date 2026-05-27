"""Partition listing + node/GPU availability via `sinfo`.

Format used: `sinfo -h -o '%P|%t|%D|%G'` emits one row per (partition, state)
group, with node count and GRES like `gpu:h200:8` or `gpu:8(S:0-1)`.
"""

import re
from collections import defaultdict

from pydantic import BaseModel

from .clusters import load_cluster
from .ssh import ssh_run


_GPU_PER_NODE_RE = re.compile(r"gpu(?::[A-Za-z0-9_-]+)?:(\d+)")
_GPU_TYPE_RE = re.compile(r"gpu:([A-Za-z0-9_-]+):\d+")


class PartitionInfo(BaseModel):
    name: str
    is_default: bool
    is_background: bool          # treat as preemptible → submit auto-adds --requeue
    total_nodes: int
    idle_nodes: int
    gpu_total: int
    gpu_idle: int
    gpu_type: str | None = None
    states: dict[str, int]       # e.g. {"idle": 1, "mix": 5}


def _is_idle(state: str) -> bool:
    # 'idle' or 'idle~' (cloud-suspended idle) both count as idle.
    return state.startswith("idle")


def _gpus_per_node(gres: str) -> int:
    m = _GPU_PER_NODE_RE.search(gres)
    return int(m.group(1)) if m else 0


def _gpu_type(gres: str) -> str | None:
    m = _GPU_TYPE_RE.search(gres)
    return m.group(1).upper() if m else None


def _strip_state(s: str) -> str:
    """Trim trailing modifiers like '*', '~', '-'. Keep the base state."""
    return s.rstrip("*~-")


async def list_partitions(cluster: str) -> list[PartitionInfo]:
    env = await load_cluster(cluster)
    configured_default = (env.vars.get("PARTITION") or "").strip()
    r = await ssh_run(env.ssh_alias, "sinfo -h -o '%P|%t|%D|%G'", timeout=15.0)
    if r.returncode != 0:
        raise RuntimeError(f"sinfo failed: {r.stderr}")

    # Each row: PARTITION | STATE | NODE_COUNT | GRES
    rows: list[tuple[str, str, int, str]] = []
    defaults: set[str] = set()
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        name = parts[0]
        if name.endswith("*"):
            name = name[:-1]
            defaults.add(name)
        state = parts[1]
        try:
            count = int(parts[2])
        except ValueError:
            continue
        rows.append((name, state, count, parts[3]))

    per_part: dict[str, dict] = defaultdict(lambda: {
        "states": defaultdict(int),
        "total_nodes": 0, "idle_nodes": 0,
        "gpu_total": 0, "gpu_idle": 0, "gpu_type": None,
    })
    for name, state, count, gres in rows:
        gpn = _gpus_per_node(gres)
        gpu_type = _gpu_type(gres)
        d = per_part[name]
        if gpu_type and not d["gpu_type"]:
            d["gpu_type"] = gpu_type
        st_key = _strip_state(state)
        d["states"][st_key] += count
        d["total_nodes"] += count
        d["gpu_total"] += count * gpn
        if _is_idle(state):
            d["idle_nodes"] += count
            d["gpu_idle"] += count * gpn

    def is_bg(name: str) -> bool:
        return name == "background" or name.endswith("_background")

    out = []
    for name in sorted(per_part):
        d = per_part[name]
        is_default = name == configured_default if configured_default else name in defaults
        out.append(PartitionInfo(
            name=name,
            is_default=is_default,
            is_background=is_bg(name),
            total_nodes=d["total_nodes"],
            idle_nodes=d["idle_nodes"],
            gpu_total=d["gpu_total"],
            gpu_idle=d["gpu_idle"],
            gpu_type=d["gpu_type"],
            states=dict(d["states"]),
        ))
    return out
