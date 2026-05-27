"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { Partition } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ImmediateTooltip } from "@/components/immediate-tooltip";

export function AvailabilityCard({
  cluster,
  partitions,
}: {
  cluster: string;
  partitions: Partition[];
}) {
  const [open, setOpen] = useState(true);
  const totalAvailableGpu = partitions.reduce((s, p) => s + p.gpu_idle, 0);
  const totalGpu = partitions.reduce((s, p) => s + p.gpu_total, 0);
  return (
    <Card>
      <CardHeader
        className="cursor-pointer select-none"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            {open ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            GPU availability
          </CardTitle>
          <span className="font-mono text-xs text-slate-500">
            <span
              className={
                totalAvailableGpu > 0 ? "text-green-600 dark:text-green-400" : ""
              }
            >
              {totalAvailableGpu}
            </span>
            <span className="text-slate-400"> / {totalGpu} available</span>
          </span>
        </div>
        <CardDescription>
          {cluster} · refreshes every 30s
        </CardDescription>
      </CardHeader>
      {open && (
        <CardContent>
          <div className="space-y-2">
            {partitions.map((p) => (
              <div
                key={p.name}
                className="flex items-center justify-between gap-3 text-xs"
              >
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
                <div className="shrink-0 font-mono">
                  <span
                    className={
                      p.gpu_idle > 0
                        ? "text-green-600 dark:text-green-400"
                        : "text-slate-500"
                    }
                  >
                    {p.gpu_idle}
                  </span>
                  <span className="text-slate-400"> / {p.gpu_total} {p.gpu_type ?? "GPU"}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}
