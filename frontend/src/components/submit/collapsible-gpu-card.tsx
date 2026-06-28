"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export type CollapsibleGpuItem = {
  key: string;
  // Left-hand label (e.g. node/partition name, possibly with a tooltip/badge).
  label: React.ReactNode;
  free: number;
  total: number;
  // Trailing unit text after "free / total " (e.g. "H100" or "GPU available").
  unit: React.ReactNode;
};

// Generic collapsible availability card shared by the slurm AvailabilityCard
// and the MLXP card: a clickable header that toggles a per-item list, with a
// right-aligned "available / total" summary (N179).
export function CollapsibleGpuCard({
  title,
  description,
  available,
  total,
  items,
}: {
  title: React.ReactNode;
  description: React.ReactNode;
  available: number;
  total: number;
  items: CollapsibleGpuItem[];
}) {
  const [open, setOpen] = useState(true);
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
            {title}
          </CardTitle>
          <span className="font-mono text-xs text-slate-500">
            <span
              className={available > 0 ? "text-green-600 dark:text-green-400" : ""}
            >
              {available}
            </span>
            <span className="text-slate-400"> / {total} available</span>
          </span>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      {open && (
        <CardContent>
          <div className="space-y-2">
            {items.map((item) => (
              <div
                key={item.key}
                className="flex items-center justify-between gap-3 text-xs"
              >
                {item.label}
                <div className="shrink-0 font-mono">
                  <span
                    className={
                      item.free > 0
                        ? "text-green-600 dark:text-green-400"
                        : "text-slate-500"
                    }
                  >
                    {item.free}
                  </span>
                  <span className="text-slate-400">
                    {" "}
                    / {item.total} {item.unit}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}
