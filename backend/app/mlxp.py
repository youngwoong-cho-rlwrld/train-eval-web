"""Naver MLXP (Kubernetes) GPU availability.

The user runs the web app on their Mac, where `kubectl` is already
configured against the MLXP control plane. We shell out to kubectl with
the user's existing kubeconfig — no separate auth.

Our service-account token is namespaced to `p-rlwrld`, so cluster-scope
`kubectl describe node` is forbidden. We derive per-node GPU usage by
listing pods in our namespace and summing their `nvidia.com/gpu`
requests. Nodes that only host other tenants' pods won't show up;
the sanctioned-for-us node is always emitted explicitly even if it has
no rlwrld pods.
"""

import asyncio
import json
import shutil
from collections import defaultdict

from pydantic import BaseModel


SANCTIONED_NODE = "h200-03-w-3a18"
GPUS_PER_H200_NODE = 8


class MlxpNode(BaseModel):
    name: str
    gpu_used: int
    gpu_total: int
    gpu_free: int
    sanctioned: bool


async def list_nodes() -> list[MlxpNode]:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")

    proc = await asyncio.create_subprocess_exec(
        "kubectl", "get", "pod", "-n", "p-rlwrld",
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

    used.setdefault(SANCTIONED_NODE, 0)

    out: list[MlxpNode] = []
    for name in sorted(used):
        # H200 GPU nodes only. CPU/control-plane nodes show up if we have a
        # data pod or pipeline pod there; those aren't GPU-relevant.
        if not name.startswith("h200-"):
            continue
        u = used[name]
        out.append(MlxpNode(
            name=name,
            gpu_used=u,
            gpu_total=GPUS_PER_H200_NODE,
            gpu_free=max(0, GPUS_PER_H200_NODE - u),
            sanctioned=(name == SANCTIONED_NODE),
        ))
    return out
