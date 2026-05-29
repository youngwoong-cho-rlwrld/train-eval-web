"""Partition listing + schedulable node/GPU availability via `sinfo`.

The monitor should only label GPUs as available when Slurm reports the node as
plain `idle`. Cloud/power/health-check states such as `idle~`, `idle#`, or
`down#` are useful context, but they are not capacity a newly submitted job can
reliably use right now.
"""

import asyncio
import re
import shlex
from collections import defaultdict

from pydantic import BaseModel

from .clusters import load_cluster
from .ssh import ssh_run


_GPU_PER_NODE_RE = re.compile(r"gpu(?::[A-Za-z0-9_-]+)?:(\d+)")
_GPU_TYPE_RE = re.compile(r"gpu:([A-Za-z0-9_-]+):\d+")
_GPU_TRES_RE = re.compile(r"(?:^|,)gres/gpu=(\d+)(?:,|$)")
_SCONTROL_FIELD_RE = re.compile(r"(\S+?)=(.*?)(?=\s+\S+=|$)")


class PartitionInfo(BaseModel):
    name: str
    is_default: bool
    is_background: bool          # treat as preemptible → submit auto-adds --requeue
    total_nodes: int
    idle_nodes: int              # schedulable plain-idle nodes
    gpu_total: int
    gpu_idle: int                # schedulable plain-idle GPUs
    queued_jobs: int = 0         # pending jobs requesting GPUs
    queued_gpus: int = 0         # pending GPU requests
    gpu_type: str | None = None
    states: dict[str, int]       # e.g. {"idle": 1, "idle~": 2, "mix": 5}


class GpuQueueNode(BaseModel):
    name: str
    gpu_type: str | None = None
    gpu_total: int
    gpu_used: int
    state: str | None = None
    reason: str | None = None


class GpuQueueJob(BaseModel):
    job_id: str
    requested_gpus: int
    reason: str | None = None
    name: str | None = None


class GpuQueueSnapshot(BaseModel):
    cluster: str
    partition: str
    nodes: list[GpuQueueNode]
    queue: list[GpuQueueJob]


def _is_schedulable_idle(state: str, reason: str) -> bool:
    return state == "idle" and reason.strip() in {"", "none"}


def _gpus_per_node(gres: str) -> int:
    m = _GPU_PER_NODE_RE.search(gres)
    return int(m.group(1)) if m else 0


def _gpu_type(gres: str) -> str | None:
    m = _GPU_TYPE_RE.search(gres)
    return m.group(1).upper() if m else None


def _gpu_tres_value(tres: str) -> int:
    m = _GPU_TRES_RE.search(tres)
    return int(m.group(1)) if m else 0


def _clean_partition_name(name: str, defaults: set[str]) -> str:
    clean = name.strip()
    if clean.endswith("*"):
        clean = clean[:-1]
        defaults.add(clean)
    return clean


def _scontrol_fields(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _SCONTROL_FIELD_RE.finditer(line)}


def _job_index_keys(fields: dict[str, str]) -> list[str]:
    keys = [fields.get("JobId") or ""]
    array_id = fields.get("ArrayJobId")
    task_id = fields.get("ArrayTaskId")
    if array_id and task_id and task_id not in {"N/A", "4294967294"}:
        keys.append(f"{array_id}_{task_id}")
    return [k for k in dict.fromkeys(k.strip() for k in keys if k.strip())]


def _node_sort_key(name: str) -> tuple[str, int, str]:
    match = re.search(r"(.*?)(\d+)$", name)
    if not match:
        return name, -1, name
    return match.group(1), int(match.group(2)), name


async def list_partitions(cluster: str) -> list[PartitionInfo]:
    env = await load_cluster(cluster)
    configured_default = (env.vars.get("PARTITION") or "").strip()
    sinfo_task = asyncio.create_task(
        ssh_run(env.ssh_alias, "sinfo -N -h -o '%P|%N|%t|%G|%E'", timeout=15.0)
    )
    squeue_task = asyncio.create_task(
        ssh_run(env.ssh_alias, "squeue -h -t PD -o '%P|%D|%b'", timeout=15.0)
    )
    r, queue_r = await asyncio.gather(sinfo_task, squeue_task)
    if r.returncode != 0:
        raise RuntimeError(f"sinfo failed: {r.stderr}")
    if queue_r.returncode != 0:
        raise RuntimeError(f"squeue failed: {queue_r.stderr}")

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
        "gpu_total": 0, "gpu_idle": 0,
        "queued_jobs": 0, "queued_gpus": 0,
        "gpu_type": None,
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

    # Each pending squeue row: PARTITION | NODE_COUNT | TRES_PER_NODE.
    # Count only jobs that request GPUs; CPU-only pending jobs are not useful
    # signal in the GPU monitor.
    for line in queue_r.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        name = _clean_partition_name(parts[0], defaults)
        if not name:
            continue
        try:
            node_count = max(1, int(parts[1]))
        except ValueError:
            node_count = 1
        requested_gpus = _gpus_per_node(parts[2]) * node_count
        if requested_gpus <= 0:
            continue
        d = per_part[name]
        d["queued_jobs"] += 1
        d["queued_gpus"] += requested_gpus

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
            queued_jobs=d["queued_jobs"],
            queued_gpus=d["queued_gpus"],
            gpu_type=d["gpu_type"],
            states=dict(d["states"]),
        ))
    return out


async def gpu_queue_snapshot(cluster: str, partition: str) -> GpuQueueSnapshot:
    env = await load_cluster(cluster)
    partition = partition.strip()
    if not partition:
        raise RuntimeError("partition is required")

    node_task = asyncio.create_task(
        ssh_run(env.ssh_alias, "scontrol show node -o", timeout=15.0)
    )
    queue_task = asyncio.create_task(
        ssh_run(
            env.ssh_alias,
            f"squeue -h -t PD -p {shlex.quote(partition)} -o '%i|%R|%j'",
            timeout=15.0,
        )
    )
    job_task = asyncio.create_task(
        ssh_run(env.ssh_alias, "scontrol show job -o", timeout=20.0)
    )
    node_r, queue_r, job_r = await asyncio.gather(node_task, queue_task, job_task)
    if node_r.returncode != 0:
        raise RuntimeError(f"scontrol show node failed: {node_r.stderr or node_r.stdout}")
    if queue_r.returncode != 0:
        raise RuntimeError(f"squeue failed: {queue_r.stderr or queue_r.stdout}")
    if job_r.returncode != 0:
        raise RuntimeError(f"scontrol show job failed: {job_r.stderr or job_r.stdout}")

    nodes: list[GpuQueueNode] = []
    for line in node_r.stdout.splitlines():
        fields = _scontrol_fields(line)
        node_partitions = {
            p.strip()
            for p in (fields.get("Partitions") or "").split(",")
            if p.strip()
        }
        if partition not in node_partitions:
            continue
        total = _gpu_tres_value(fields.get("CfgTRES") or "") or _gpus_per_node(fields.get("Gres") or "")
        if total <= 0:
            continue
        used = _gpu_tres_value(fields.get("AllocTRES") or "")
        nodes.append(
            GpuQueueNode(
                name=fields.get("NodeName") or "",
                gpu_type=_gpu_type(fields.get("Gres") or ""),
                gpu_total=total,
                gpu_used=max(0, min(used, total)),
                state=fields.get("State") or None,
                reason=fields.get("Reason") or None,
            )
        )

    job_fields_by_id: dict[str, dict[str, str]] = {}
    for line in job_r.stdout.splitlines():
        fields = _scontrol_fields(line)
        if fields.get("Partition") != partition or fields.get("JobState") != "PENDING":
            continue
        for key in _job_index_keys(fields):
            job_fields_by_id[key] = fields

    queue: list[GpuQueueJob] = []
    for line in queue_r.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        job_id = parts[0].strip()
        fields = job_fields_by_id.get(job_id, {})
        requested_gpus = _gpu_tres_value(fields.get("ReqTRES") or "")
        if requested_gpus <= 0:
            continue
        queue.append(
            GpuQueueJob(
                job_id=job_id,
                requested_gpus=requested_gpus,
                reason=(fields.get("Reason") or parts[1]).strip("() ") or None,
                name=fields.get("JobName") or parts[2].strip() or None,
            )
        )

    return GpuQueueSnapshot(
        cluster=cluster,
        partition=partition,
        nodes=sorted(nodes, key=lambda n: _node_sort_key(n.name)),
        queue=queue,
    )
