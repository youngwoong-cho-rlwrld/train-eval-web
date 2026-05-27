"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { MlxpNode } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function MlxpCard({
  nodes,
  yoursNode,
}: {
  nodes: MlxpNode[];
  yoursNode: string;
}) {
  const [open, setOpen] = useState(true);
  const available = nodes.reduce((s, n) => s + n.gpu_free, 0);
  const total = nodes.reduce((s, n) => s + n.gpu_total, 0);
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
            MLXP (Naver, k8s)
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
        <CardDescription>
          {nodes[0]?.gpu_type ?? "GPU"} nodes · your node: <code>{yoursNode || "—"}</code>
        </CardDescription>
      </CardHeader>
      {open && (
        <CardContent>
          <div className="space-y-2">
            {nodes.map((n) => (
              <div
                key={n.name}
                className="flex items-center justify-between gap-3 text-xs"
              >
                <div className="min-w-0 flex-1 truncate font-mono">
                  {n.name}
                  {n.name === yoursNode && (
                    <Badge variant="default" className="ml-1 text-[10px]">
                      yours
                    </Badge>
                  )}
                </div>
                <div className="shrink-0 font-mono">
                  <span
                    className={
                      n.gpu_free > 0
                        ? "text-green-600 dark:text-green-400"
                        : "text-slate-500"
                    }
                  >
                    {n.gpu_free}
                  </span>
                  <span className="text-slate-400">
                    {" "}
                    / {n.gpu_total} GPU available
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
