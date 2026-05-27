"""Small Kubernetes resource helpers shared by MLXP views."""


def requested_gpus(spec: dict) -> int:
    total = 0
    for container in spec.get("containers", []):
        req = (container.get("resources") or {}).get("requests") or {}
        try:
            total += int(req.get("nvidia.com/gpu", 0))
        except (TypeError, ValueError):
            pass
    return total


def affinity_node(spec: dict) -> str | None:
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
