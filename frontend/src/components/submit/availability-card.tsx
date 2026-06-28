"use client";

import type { Partition } from "@/lib/api";
import { sumGpu } from "@/lib/gpu";
import { Badge } from "@/components/ui/badge";
import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { CollapsibleGpuCard } from "@/components/submit/collapsible-gpu-card";

export function AvailabilityCard({
  cluster,
  partitions,
}: {
  cluster: string;
  partitions: Partition[];
}) {
  return (
    <CollapsibleGpuCard
      title="GPU availability"
      description={`${cluster} · refreshes every 30s`}
      available={sumGpu(partitions, "gpu_free")}
      total={sumGpu(partitions, "gpu_total")}
      items={partitions.map((p) => ({
        key: p.name,
        label: (
          <ImmediateTooltip content={p.name} className="min-w-0 flex-1">
            <div className="truncate font-mono">
              {p.name}
              {p.is_default && (
                <Badge variant="secondary" className="ml-1 text-[10px]">
                  def
                </Badge>
              )}
            </div>
          </ImmediateTooltip>
        ),
        free: p.gpu_free,
        total: p.gpu_total,
        unit: p.gpu_type ?? "GPU",
      }))}
    />
  );
}
