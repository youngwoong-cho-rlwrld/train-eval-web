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

from .mlxp_config import DEFAULT_NODE, GPUS_PER_NODE, GPU_NODE_PREFIX, NAMESPACE


class MlxpNode(BaseModel):
    name: str
    gpu_used: int
    gpu_total: int
    gpu_free: int


async def list_nodes() -> list[MlxpNode]:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")

    proc = await asyncio.create_subprocess_exec(
        "kubectl", "get", "pod", "-n", NAMESPACE,
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
        if not name.startswith(GPU_NODE_PREFIX):
            continue
        u = used[name]
        out.append(MlxpNode(
            name=name,
            gpu_used=u,
            gpu_total=GPUS_PER_NODE,
            gpu_free=max(0, GPUS_PER_NODE - u),
        ))
    if DEFAULT_NODE and all(n.name != DEFAULT_NODE for n in out):
        out.append(MlxpNode(
            name=DEFAULT_NODE,
            gpu_used=0,
            gpu_total=GPUS_PER_NODE,
            gpu_free=GPUS_PER_NODE,
        ))
        out.sort(key=lambda n: n.name)
    return out
