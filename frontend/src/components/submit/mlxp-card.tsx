"use client";

import type { MlxpNode } from "@/lib/api";
import { sumGpu } from "@/lib/gpu";
import { CollapsibleGpuCard } from "@/components/submit/collapsible-gpu-card";

export function MlxpCard({ nodes }: { nodes: MlxpNode[] }) {
  return (
    <CollapsibleGpuCard
      title="MLXP (Naver, k8s)"
      description={`${nodes[0]?.gpu_type ?? "GPU"} nodes`}
      available={sumGpu(nodes, "gpu_free")}
      total={sumGpu(nodes, "gpu_total")}
      items={nodes.map((n) => ({
        key: n.name,
        label: <div className="min-w-0 flex-1 truncate font-mono">{n.name}</div>,
        free: n.gpu_free,
        total: n.gpu_total,
        unit: `${n.gpu_type ?? "GPU"} available`,
      }))}
    />
  );
}
