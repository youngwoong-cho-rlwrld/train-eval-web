"""Small Kubernetes resource helpers shared by MLXP views."""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime
from typing import Any


def ensure_kubectl() -> None:
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")


async def kubectl_json(*args: str, timeout: float = 20.0) -> dict[str, Any]:
    ensure_kubectl()
    proc = await asyncio.create_subprocess_exec(
        "kubectl",
        *args,
        "-o",
        "json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"kubectl {' '.join(args)} timed out after {timeout:g}s")
    if proc.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {message}")
    try:
        return json.loads(stdout.decode())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"kubectl {' '.join(args)} returned invalid JSON: {exc}") from exc


def requested_gpus(spec: dict[str, Any]) -> int:
    total = 0
    for container in spec.get("containers", []):
        req = (container.get("resources") or {}).get("requests") or {}
        try:
            total += int(req.get("nvidia.com/gpu", 0))
        except (TypeError, ValueError):
            pass
    return total


def pod_job_id(pod: dict[str, Any]) -> str | None:
    metadata = pod.get("metadata") or {}
    labels = metadata.get("labels") or {}
    return (
        labels.get("job-name")
        or labels.get("batch.kubernetes.io/job-name")
        or metadata.get("name")
    )


def affinity_node(spec: dict[str, Any]) -> str | None:
    affinity = spec.get("affinity") or {}
    node_affinity = affinity.get("nodeAffinity") or {}
    required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution") or {}
    for term in required.get("nodeSelectorTerms", []):
        for expr in term.get("matchExpressions", []):
            if expr.get("key") != "kubernetes.io/hostname":
                continue
            values = expr.get("values") or []
            if values:
                return str(values[0])
    return None


def parse_k8s_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return None


def pending_pod_reason(pod: dict[str, Any]) -> str | None:
    status = pod.get("status") or {}
    for condition in status.get("conditions", []) or []:
        if condition.get("type") != "PodScheduled" or condition.get("status") != "False":
            continue
        return condition.get("message") or condition.get("reason")
    return status.get("message") or status.get("reason") or None


def gpu_type_for_node(node: str, fallback: str | None) -> str | None:
    prefix = node.split("-", 1)[0].strip()
    if prefix and any(ch.isdigit() for ch in prefix):
        return prefix.upper()
    return fallback or None
