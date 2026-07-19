"""Naver MLXP (Kubernetes) GPU availability.

The user runs the web app on their Mac, where `kubectl` is already
configured against the MLXP control plane. We shell out to kubectl with
the user's existing kubeconfig — no separate auth.

Our service-account token is namespaced to MLXP's project namespace, so cluster-scope
`kubectl describe node` is forbidden. We derive per-node GPU usage by
listing pods in our namespace and summing their `nvidia.com/gpu`
requests. Nodes that only host other tenants' pods won't show up;
the configured default node is always emitted explicitly even if it has
no owned pods.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from .k8s_resources import (
    affinity_node,
    gpu_type_for_node,
    kubectl_json,
    parse_k8s_time,
    pending_pod_reason,
    pod_job_id,
    requested_gpus,
)
from .mlxp_config import get_settings
from .partitions import GpuQueueJob, GpuQueueNode, GpuQueueSnapshot


class MlxpNode(BaseModel):
    name: str
    gpu_used: int
    gpu_total: int
    gpu_free: int
    queued_jobs: int = 0
    queued_gpus: int = 0
    gpu_type: str | None = None


async def list_nodes() -> list[MlxpNode]:
    settings = get_settings()
    data = await kubectl_json("get", "pod", "-n", settings.namespace)

    used: dict[str, int] = defaultdict(int)
    queued_gpus: dict[str, int] = defaultdict(int)
    queued_jobs: dict[str, int] = defaultdict(int)
    for _, phase, node, gpu_count in _gpu_pod_rows(data):
        if not node:
            continue
        if phase == "Running":
            used[node] += gpu_count
        elif phase == "Pending":
            queued_gpus[node] += gpu_count
            queued_jobs[node] += 1

    out: list[MlxpNode] = []
    # The node set is exactly the nodes hosting GPU-requesting pods (see
    # _gpu_pod_rows); no node-name prefix filter is needed to exclude
    # CPU/control-plane nodes.
    for name in sorted(set(used) | set(queued_gpus)):
        u = used[name]
        out.append(MlxpNode(
            name=name,
            gpu_used=u,
            gpu_total=settings.gpus_per_node,
            gpu_free=max(0, settings.gpus_per_node - u),
            queued_jobs=queued_jobs[name],
            queued_gpus=queued_gpus[name],
            gpu_type=gpu_type_for_node(name),
        ))
    return out


async def gpu_queue_snapshot(
    job_id: str | None = None,
    node: str | None = None,
) -> GpuQueueSnapshot:
    settings = get_settings()
    pods_task = asyncio.create_task(kubectl_json("get", "pod", "-n", settings.namespace))
    jobs_task = asyncio.create_task(kubectl_json("get", "job", "-n", settings.namespace))
    pods_data, jobs_data = await asyncio.gather(pods_task, jobs_task)

    job_names = _job_display_names(jobs_data)
    used: dict[str, int] = defaultdict(int)
    pending: list[tuple[str, str | None, datetime, str, int, str | None, str | None]] = []
    current_node: str | None = None
    max_time = datetime.max.replace(tzinfo=timezone.utc)

    for pod, phase, pod_node, gpu_count in _gpu_pod_rows(pods_data):
        metadata = pod.get("metadata") or {}
        pod_job = pod_job_id(pod)
        if not pod_job:
            continue

        if phase == "Running" and pod_node:
            used[pod_node] += gpu_count
        elif phase == "Pending":
            created = parse_k8s_time(metadata.get("creationTimestamp")) or max_time
            pending.append((
                pod_job,
                pod_node,
                created,
                metadata.get("name") or pod_job,
                gpu_count,
                pending_pod_reason(pod),
                job_names.get(pod_job),
            ))
        if job_id and pod_job == job_id:
            current_node = pod_node

    scoped_node = current_node or (node.strip() if node else None)
    if scoped_node:
        pending = [item for item in pending if item[1] == scoped_node]
        node_names = {scoped_node}
    else:
        node_names = set(used)
        node_names.update(pod_node for _, pod_node, *_ in pending if pod_node)

    queue = [
        GpuQueueJob(
            job_id=queued_job_id,
            requested_gpus=gpu_count,
            reason=reason,
            name=display_name or pod_name,
        )
        for queued_job_id, _, _, pod_name, gpu_count, reason, display_name in sorted(
            pending,
            key=lambda item: (item[2], item[3]),
        )
    ]
    nodes = [
        GpuQueueNode(
            name=name,
            gpu_type=gpu_type_for_node(name),
            gpu_total=settings.gpus_per_node,
            gpu_used=max(0, min(used[name], settings.gpus_per_node)),
            state="k8s",
            reason="MLXP usage is derived from visible GPU pods in this namespace.",
        )
        for name in sorted(node_names)
        if name
    ]
    return GpuQueueSnapshot(
        cluster="mlxp",
        partition="mlxp",
        nodes=nodes,
        queue=queue,
    )


def _gpu_pod_rows(
    data: dict[str, Any],
) -> list[tuple[dict[str, Any], str, str | None, int]]:
    """GPU pods in the namespace: a pod is GPU-relevant iff it requests
    ``nvidia.com/gpu`` > 0. This (not a node-name prefix) is what identifies
    GPU nodes — the node set is exactly the nodes hosting these pods."""
    rows: list[tuple[dict[str, Any], str, str | None, int]] = []
    for pod in data.get("items", []):
        spec = pod.get("spec") or {}
        gpu_count = requested_gpus(spec)
        if gpu_count <= 0:
            continue
        node = spec.get("nodeName") or affinity_node(spec)
        phase = ((pod.get("status") or {}).get("phase") or "").strip()
        rows.append((pod, phase, node, gpu_count))
    return rows


def _job_display_names(data: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in data.get("items", []) or []:
        metadata = item.get("metadata") or {}
        name = metadata.get("name")
        display_name = (metadata.get("annotations") or {}).get("train-eval-web/display-name")
        if name and display_name:
            out[name] = display_name
    return out
