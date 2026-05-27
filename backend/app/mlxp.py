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

import asyncio
import json
import shutil
from collections import defaultdict

from pydantic import BaseModel

from .k8s_resources import affinity_node, requested_gpus
from .mlxp_config import get_settings


class MlxpNode(BaseModel):
    name: str
    gpu_used: int
    gpu_total: int
    gpu_free: int
    queued_jobs: int = 0
    queued_gpus: int = 0
    gpu_type: str | None = None


async def list_nodes() -> list[MlxpNode]:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    settings = get_settings()

    proc = await asyncio.create_subprocess_exec(
        "kubectl", "get", "pod", "-n", settings.namespace,
        "-o", "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20.0)
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl failed: {stderr.decode(errors='replace').strip()}")

    data = json.loads(stdout.decode())
    used: dict[str, int] = defaultdict(int)
    queued_gpus: dict[str, int] = defaultdict(int)
    queued_jobs: dict[str, int] = defaultdict(int)
    for p in data.get("items", []):
        phase = (p.get("status") or {}).get("phase")
        spec = p.get("spec") or {}
        node = spec.get("nodeName") or affinity_node(spec)
        if not node:
            continue
        gpu_count = requested_gpus(spec)
        if gpu_count <= 0:
            continue
        if phase == "Running":
            used[node] += gpu_count
        elif phase == "Pending":
            queued_gpus[node] += gpu_count
            queued_jobs[node] += 1

    out: list[MlxpNode] = []
    for name in sorted(set(used) | set(queued_gpus)):
        # GPU nodes only. CPU/control-plane nodes show up if we have a data
        # pod or pipeline pod there; those aren't GPU-relevant.
        if settings.gpu_node_prefix and not name.startswith(settings.gpu_node_prefix):
            continue
        u = used[name]
        out.append(MlxpNode(
            name=name,
            gpu_used=u,
            gpu_total=settings.gpus_per_node,
            gpu_free=max(0, settings.gpus_per_node - u),
            queued_jobs=queued_jobs[name],
            queued_gpus=queued_gpus[name],
            gpu_type=_gpu_type_for_node(name, settings.gpu_type),
        ))
    if settings.default_node and all(n.name != settings.default_node for n in out):
        out.append(MlxpNode(
            name=settings.default_node,
            gpu_used=0,
            gpu_total=settings.gpus_per_node,
            gpu_free=settings.gpus_per_node,
            queued_jobs=queued_jobs[settings.default_node],
            queued_gpus=queued_gpus[settings.default_node],
            gpu_type=_gpu_type_for_node(settings.default_node, settings.gpu_type),
        ))
        out.sort(key=lambda n: n.name)
    return out


def _gpu_type_for_node(node: str, fallback: str | None) -> str | None:
    prefix = node.split("-", 1)[0].strip()
    if prefix and any(ch.isdigit() for ch in prefix):
        return prefix.upper()
    return fallback or None
