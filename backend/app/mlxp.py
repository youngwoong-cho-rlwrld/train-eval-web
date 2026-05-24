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

from .mlxp_config import get_settings


class MlxpNode(BaseModel):
    name: str
    gpu_used: int
    gpu_total: int
    gpu_free: int
    gpu_type: str | None = None


async def list_nodes() -> list[MlxpNode]:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")
    settings = get_settings()

    proc = await asyncio.create_subprocess_exec(
        "kubectl", "get", "pod", "-n", settings.namespace,
        "--field-selector", "status.phase=Running",
        "-o", "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20.0)
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl failed: {stderr.decode(errors='replace').strip()}")

    data = json.loads(stdout.decode())
    used: dict[str, int] = defaultdict(int)
    for p in data.get("items", []):
        node = p.get("spec", {}).get("nodeName")
        if not node:
            continue
        for c in p.get("spec", {}).get("containers", []):
            req = (c.get("resources") or {}).get("requests") or {}
            try:
                used[node] += int(req.get("nvidia.com/gpu", 0))
            except (TypeError, ValueError):
                pass

    out: list[MlxpNode] = []
    for name in sorted(used):
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
            gpu_type=_gpu_type_for_node(name, settings.gpu_type),
        ))
    if settings.default_node and all(n.name != settings.default_node for n in out):
        out.append(MlxpNode(
            name=settings.default_node,
            gpu_used=0,
            gpu_total=settings.gpus_per_node,
            gpu_free=settings.gpus_per_node,
            gpu_type=_gpu_type_for_node(settings.default_node, settings.gpu_type),
        ))
        out.sort(key=lambda n: n.name)
    return out


def _gpu_type_for_node(node: str, fallback: str | None) -> str | None:
    prefix = node.split("-", 1)[0].strip()
    if prefix and any(ch.isdigit() for ch in prefix):
        return prefix.upper()
    return fallback or None
