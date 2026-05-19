"""Auto-provision a data pod on MLXP for DDN listing operations."""

import asyncio
import json
import shutil

from .mlxp_config import (
    DATA_POD_NAME,
    DDN_MOUNT,
    DDN_PVC,
    DEFAULT_NODE,
    IMAGE,
    IMAGE_PULL_SECRET,
    NAMESPACE,
    OWNER_LABEL,
    TOOL_LABEL,
    ZONE,
    owner_selector,
)

DATA_POD_YAML = f"""apiVersion: v1
kind: Pod
metadata:
  name: {DATA_POD_NAME}
  namespace: {NAMESPACE}
  annotations:
    mlx.navercorp.com/zone: {ZONE}
    sidecar.istio.io/inject: "false"
  labels:
    owner: {OWNER_LABEL}
    tool: {TOOL_LABEL}
spec:
  restartPolicy: Never
  imagePullSecrets:
  - name: {IMAGE_PULL_SECRET}
  volumes:
  - name: ddn
    persistentVolumeClaim:
      claimName: {DDN_PVC}
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: kubernetes.io/hostname
            operator: In
            values:
            - {DEFAULT_NODE}
  containers:
  - name: main
    image: {IMAGE}
    command: ["sleep", "14400"]
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "none"
    resources:
      requests:
        cpu: "4"
        memory: "16Gi"
      limits:
        cpu: "4"
        memory: "16Gi"
    volumeMounts:
    - name: ddn
      mountPath: {DDN_MOUNT}
"""


_CACHE_TTL = 5.0  # seconds
_pods_cache: tuple[float, dict] | None = None
_pods_lock = asyncio.Lock()


def _label_matches(item: dict, label: str | None) -> bool:
    if not label or "=" not in label:
        return True
    k, v = label.split("=", 1)
    return (item.get("metadata", {}).get("labels") or {}).get(k) == v


async def _kubectl_get_pods_json(label: str | None = None) -> dict:
    """Fetch all pods in the namespace, with a small shared TTL cache.

    Many endpoints (each ProgressCell, ensure_listing_pod, mlxp_jobs.list_jobs)
    call this in rapid bursts. One kubectl per 5s window is plenty; the lock
    keeps concurrent callers from firing duplicates while the first is still
    in flight against MLXP's sometimes-slow API.
    """
    global _pods_cache
    async with _pods_lock:
        now = asyncio.get_event_loop().time()
        if _pods_cache and now - _pods_cache[0] < _CACHE_TTL:
            data = _pods_cache[1]
        else:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "get", "pods", "-n", NAMESPACE, "-o", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                # MLXP API server is occasionally slow on TLS handshake.
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {"items": []}
            if proc.returncode != 0:
                return {"items": []}
            try:
                data = json.loads(stdout.decode())
            except json.JSONDecodeError:
                return {"items": []}
            _pods_cache = (now, data)

    if not label:
        return data
    items = [it for it in data.get("items", []) if _label_matches(it, label)]
    return {**data, "items": items}


async def _find_running_with_ddn() -> str | None:
    """First Running owned pod that has the configured DDN PVC mounted."""
    data = await _kubectl_get_pods_json(owner_selector())
    for item in data.get("items", []):
        if (item.get("status") or {}).get("phase") != "Running":
            continue
        vols = ((item.get("spec") or {}).get("volumes") or [])
        if any(
            ((v.get("persistentVolumeClaim") or {}).get("claimName") == DDN_PVC)
            for v in vols
        ):
            return item["metadata"]["name"]
    return None


async def _apply_yaml(yaml_text: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "create", "-f", "-", "--validate=false", "-n", NAMESPACE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=yaml_text.encode())
    if proc.returncode != 0:
        raise RuntimeError(
            f"kubectl apply (data-pod) failed: "
            f"{stderr.decode(errors='replace').strip()}"
        )


async def _wait_until_running(name: str, timeout: float = 90.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        data = await _kubectl_get_pods_json()
        for item in data.get("items", []):
            if item.get("metadata", {}).get("name") != name:
                continue
            phase = (item.get("status") or {}).get("phase")
            if phase == "Running":
                return
            if phase in ("Failed", "Succeeded"):
                raise RuntimeError(f"data pod entered terminal phase {phase}")
            break
        await asyncio.sleep(2.0)
    raise RuntimeError(f"data pod {name} did not reach Running within {timeout:.0f}s")


async def _get_pod_by_name(name: str) -> dict | None:
    data = await _kubectl_get_pods_json()
    for item in data.get("items", []):
        if item.get("metadata", {}).get("name") == name:
            return item
    return None


async def ensure_listing_pod() -> str:
    """Return a Running pod usable for DDN listing, creating one if needed."""
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")

    existing = await _find_running_with_ddn()
    if existing:
        return existing

    # Pod may already exist without our label (e.g. left over from the
    # manual YAML pre-this-session). Don't double-apply; just wait.
    stale = await _get_pod_by_name(DATA_POD_NAME)
    if not stale:
        await _apply_yaml(DATA_POD_YAML)
    elif (stale.get("status") or {}).get("phase") in ("Failed", "Succeeded"):
        # Pod is dead but lingering — recreate it.
        await _delete_pod(DATA_POD_NAME)
        await _apply_yaml(DATA_POD_YAML)

    await _wait_until_running(DATA_POD_NAME)
    return DATA_POD_NAME


async def _delete_pod(name: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "delete", "pod", name, "-n", NAMESPACE, "--wait=false",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(proc.communicate(), timeout=15.0)
