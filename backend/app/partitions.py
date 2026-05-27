"""Partition listing + schedulable node/GPU availability via `sinfo`.

The monitor should only label GPUs as available when Slurm reports the node as
plain `idle`. Cloud/power/health-check states such as `idle~`, `idle#`, or
`down#` are useful context, but they are not capacity a newly submitted job can
reliably use right now.
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
    idle_nodes: int              # schedulable plain-idle nodes
    gpu_total: int
    gpu_idle: int                # schedulable plain-idle GPUs
    gpu_type: str | None = None
    states: dict[str, int]       # e.g. {"idle": 1, "idle~": 2, "mix": 5}


def _is_schedulable_idle(state: str, reason: str) -> bool:
    return state == "idle" and reason.strip() in {"", "none"}


def _gpus_per_node(gres: str) -> int:
    m = _GPU_PER_NODE_RE.search(gres)
    return int(m.group(1)) if m else 0


def _gpu_type(gres: str) -> str | None:
    m = _GPU_TYPE_RE.search(gres)
    return m.group(1).upper() if m else None


def _clean_partition_name(name: str, defaults: set[str]) -> str:
    clean = name.strip()
    if clean.endswith("*"):
        clean = clean[:-1]
        defaults.add(clean)
    return clean


async def list_partitions(cluster: str) -> list[PartitionInfo]:
    env = await load_cluster(cluster)
    configured_default = (env.vars.get("PARTITION") or "").strip()
    r = await ssh_run(env.ssh_alias, "sinfo -N -h -o '%P|%N|%t|%G|%E'", timeout=15.0)
    if r.returncode != 0:
        raise RuntimeError(f"sinfo failed: {r.stderr}")

    # Each row: PARTITION | NODE | STATE | GRES | REASON.
    rows: list[tuple[str, str, str, str]] = []
    defaults: set[str] = set()
    for line in r.stdout.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        name = _clean_partition_name(parts[0], defaults)
        if not name:
            continue
        rows.append((name, parts[2].strip(), parts[3].strip(), parts[4].strip()))

    per_part: dict[str, dict] = defaultdict(lambda: {
        "states": defaultdict(int),
        "total_nodes": 0, "idle_nodes": 0,
        "gpu_total": 0, "gpu_idle": 0, "gpu_type": None,
    })
    for name, state, gres, reason in rows:
        gpn = _gpus_per_node(gres)
        gpu_type = _gpu_type(gres)
        d = per_part[name]
        if gpu_type and not d["gpu_type"]:
            d["gpu_type"] = gpu_type
        d["states"][state] += 1
        d["total_nodes"] += 1
        d["gpu_total"] += gpn
        if _is_schedulable_idle(state, reason):
            d["idle_nodes"] += 1
            d["gpu_idle"] += gpn

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
