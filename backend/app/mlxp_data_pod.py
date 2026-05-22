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
    command: ["sleep", "infinity"]
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


def _transient_kubectl_error(message: str) -> bool:
    lower = message.lower()
    return any(
        token in lower
        for token in (
            "failed calling webhook",
            "failed to call webhook",
            "admission-webhook",
            "error sending request",
            "cannot assign requested address",
            "connection refused",
            "i/o timeout",
            "tls handshake timeout",
            "context deadline exceeded",
            "internal error occurred",
            "no such host",
        )
    )


def _label_matches(item: dict, label: str | None) -> bool:
    if not label or "=" not in label:
        return True
    k, v = label.split("=", 1)
    return (item.get("metadata", {}).get("labels") or {}).get(k) == v


def invalidate_pods_cache() -> None:
    global _pods_cache
    _pods_cache = None


async def _kubectl_get_pods_json(
    label: str | None = None,
    *,
    refresh: bool = False,
    strict: bool = False,
) -> dict:
    """Fetch all pods in the namespace, with a small shared TTL cache.

    Many endpoints (each ProgressCell, ensure_listing_pod, mlxp_jobs.list_jobs)
    call this in rapid bursts. One kubectl per 5s window is plenty; the lock
    keeps concurrent callers from firing duplicates while the first is still
    in flight against MLXP's sometimes-slow API.
    """
    global _pods_cache
    async with _pods_lock:
        now = asyncio.get_event_loop().time()
        if not refresh and _pods_cache and now - _pods_cache[0] < _CACHE_TTL:
            data = _pods_cache[1]
        else:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "get", "pods", "-n", NAMESPACE, "-o", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                # MLXP API server is occasionally slow on TLS handshake.
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                if strict:
                    raise RuntimeError("kubectl get pods timed out")
                return {"items": []}
            if proc.returncode != 0:
                if strict:
                    raise RuntimeError(
                        "kubectl get pods failed: "
                        f"{stderr.decode(errors='replace').strip()}"
                    )
                return {"items": []}
            try:
                data = json.loads(stdout.decode())
            except json.JSONDecodeError:
                if strict:
                    raise RuntimeError("kubectl get pods returned invalid JSON")
                return {"items": []}
            _pods_cache = (now, data)

    if not label:
        return data
    items = [it for it in data.get("items", []) if _label_matches(it, label)]
    return {**data, "items": items}


async def _find_running_with_ddn(*, refresh: bool = False, strict: bool = False) -> str | None:
    """First Running owned pod that has the configured DDN PVC mounted."""
    data = await _kubectl_get_pods_json(owner_selector(), refresh=refresh, strict=strict)
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
    global _pods_cache
    last_error = ""
    for attempt in range(1, 6):
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "create", "-f", "-", "--validate=false", "-n", NAMESPACE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=yaml_text.encode())
        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        if proc.returncode == 0:
            _pods_cache = None
            return
        last_error = err or out or "kubectl create failed"
        if "already exists" in last_error.lower():
            _pods_cache = None
            return
        if not _transient_kubectl_error(last_error) or attempt == 5:
            break
        await asyncio.sleep(1.5 * attempt)
    raise RuntimeError(f"kubectl apply (data-pod) failed: {last_error}")


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


async def _get_pod_by_name(
    name: str,
    *,
    refresh: bool = False,
    strict: bool = False,
) -> dict | None:
    data = await _kubectl_get_pods_json(refresh=refresh, strict=strict)
    for item in data.get("items", []):
        if item.get("metadata", {}).get("name") == name:
            return item
    return None


async def ensure_listing_pod() -> str:
    """Return a Running pod usable for DDN listing, creating one if needed."""
    if shutil.which("kubectl") is None:
        raise RuntimeError("kubectl not found on PATH")

    # Use the dedicated CPU-only data pod. Reusing arbitrary running training
    # pods makes data movement depend on unrelated job lifecycles.
    for attempt in range(1, 6):
        try:
            pod = await _get_pod_by_name(DATA_POD_NAME, refresh=True, strict=True)
        except RuntimeError as e:
            if not _transient_kubectl_error(str(e)) or attempt == 5:
                raise
            await asyncio.sleep(1.5 * attempt)
            continue

        if pod is None:
            await _apply_yaml(DATA_POD_YAML)
            await _wait_until_running(DATA_POD_NAME)
            return DATA_POD_NAME

        phase = (pod.get("status") or {}).get("phase")
        if phase == "Running":
            vols = ((pod.get("spec") or {}).get("volumes") or [])
            if any(
                ((v.get("persistentVolumeClaim") or {}).get("claimName") == DDN_PVC)
                for v in vols
            ):
                return DATA_POD_NAME
            raise RuntimeError(f"data pod {DATA_POD_NAME} is running without DDN PVC {DDN_PVC}")

        if phase in ("Failed", "Succeeded"):
            await _delete_pod(DATA_POD_NAME)
            await _apply_yaml(DATA_POD_YAML)
            await _wait_until_running(DATA_POD_NAME)
            return DATA_POD_NAME

        await _wait_until_running(DATA_POD_NAME)
        return DATA_POD_NAME

    return DATA_POD_NAME


async def _delete_pod(name: str) -> None:
    global _pods_cache
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "delete", "pod", name, "-n", NAMESPACE,
        "--ignore-not-found=true", "--wait=true", "--timeout=60s",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=75.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"kubectl delete pod failed: "
            f"{stderr.decode(errors='replace').strip() or stdout.decode(errors='replace').strip()}"
        )
    _pods_cache = None
