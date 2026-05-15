"""Auto-provision a data pod on MLXP for DDN listing operations.

Listing datasets on /data/youngwoong/ requires `kubectl exec` into a pod
that mounts the DDN PVC. A long-running data pod is the cheapest way to
have that available on demand.

`ensure_listing_pod()` returns the name of any running pod that satisfies
this — any existing `owner=youngwoong` Running pod will do, including
training pods. If nothing is running, it kubectl-applies our standard
data-pod spec and waits for it to schedule.
"""

import asyncio
import json
import shutil

DATA_POD_NAME = "youngwoong-data-pod"
NAMESPACE = "p-rlwrld"
SANCTIONED_NODE = "h200-03-w-3a18"

# Inline so the backend doesn't depend on a YAML file shipped alongside it.
# Matches /Users/youngwoong/workspace/mlxp/youngwoong-data-pod.yaml; updates
# there should be mirrored here.
DATA_POD_YAML = f"""apiVersion: v1
kind: Pod
metadata:
  name: {DATA_POD_NAME}
  namespace: {NAMESPACE}
  annotations:
    mlx.navercorp.com/zone: private-h200-rlwrld-0
    sidecar.istio.io/inject: "false"
  labels:
    owner: youngwoong
    tool: train-eval-web
spec:
  restartPolicy: Never
  imagePullSecrets:
  - name: mlxp-registry
  volumes:
  - name: ddn
    persistentVolumeClaim:
      claimName: ddn-rlwrld-shared
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: kubernetes.io/hostname
            operator: In
            values:
            - {SANCTIONED_NODE}
  containers:
  - name: main
    image: mlxp.kr.ncr.ntruss.com/rlwrld-gpu-base:latest
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
      mountPath: /data
"""


async def _kubectl_get_pods_json(label: str | None = None) -> dict:
    args = ["kubectl", "get", "pods", "-n", NAMESPACE]
    if label:
        args += ["-l", label]
    args += ["-o", "json"]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(stdout.decode())
    except json.JSONDecodeError:
        return {}


async def _find_running_with_ddn() -> str | None:
    """First Running owner=youngwoong pod that has the ddn PVC mounted."""
    data = await _kubectl_get_pods_json("owner=youngwoong")
    for item in data.get("items", []):
        if (item.get("status") or {}).get("phase") != "Running":
            continue
        vols = ((item.get("spec") or {}).get("volumes") or [])
        if any(
            ((v.get("persistentVolumeClaim") or {}).get("claimName") == "ddn-rlwrld-shared")
            for v in vols
        ):
            return item["metadata"]["name"]
    return None


async def _apply_yaml(yaml_text: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "kubectl", "apply", "-f", "-", "-n", NAMESPACE,
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
