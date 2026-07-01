"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useIsFetching, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type GpuQueueSnapshot, type MlxpNode, type Partition } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { RefreshButton } from "@/components/refresh-button";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";
import { GpuQueueTooltipContent } from "@/components/gpu-queue-visualization";
import { sumGpu } from "@/lib/gpu";
import { cn } from "@/lib/utils";

const REFRESH_MS = 60_000;
const TOOLTIP_VIEWPORT_GAP = 12;
const TOOLTIP_OFFSET = 8;

export default function MonitorPage() {
  const qc = useQueryClient();

  const clusters = useQuery({
    queryKey: ["clusters"],
    queryFn: () => api<{ clusters: string[] }>("/api/clusters").then((d) => d.clusters),
  });
  // mlxp is k8s, not slurm — it has its own panel below.
  const slurm = (clusters.data ?? []).filter((c) => c !== "mlxp");

  const refreshAll = () => {
    qc.invalidateQueries({ queryKey: ["clusters"] });
    for (const c of slurm) qc.invalidateQueries({ queryKey: ["partitions", c] });
    qc.invalidateQueries({ queryKey: ["mlxp-gpus"] });
    qc.invalidateQueries({ queryKey: ["gpu-queue"] });
  };

  const isFetching =
    useIsFetching({
      predicate: (q) => {
        const k = q.queryKey[0];
        return k === "clusters" || k === "partitions" || k === "mlxp-gpus" || k === "gpu-queue";
      },
    }) > 0;

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">GPU monitor</h1>
        <RefreshButton
          isFetching={isFetching}
          onRefresh={refreshAll}
          intervalMs={REFRESH_MS}
        />
      </div>

      <div className="mt-8 space-y-6">
        {(clusters.isLoading || clusters.error || slurm.length === 0) && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Slurm clusters</CardTitle>
            </CardHeader>
            <CardContent>
              {clusters.isLoading ? (
                <LoadingState label="Loading clusters..." rows={3} />
              ) : clusters.error ? (
                <ErrorState message={(clusters.error as Error).message} />
              ) : (
                <EmptyState message="No Slurm clusters configured." />
              )}
            </CardContent>
          </Card>
        )}
        {slurm.map((c) => (
          <SlurmClusterPanel key={c} cluster={c} />
        ))}
        <MlxpPanel />
      </div>
    </div>
  );
}

function SlurmClusterPanel({ cluster }: { cluster: string }) {
  const q = useQuery({
    queryKey: ["partitions", cluster],
    queryFn: () => api<Partition[]>(`/api/clusters/${cluster}/partitions`),
    refetchInterval: REFRESH_MS,
  });
  const ps = q.data ?? [];
  const available = sumGpu(ps, "gpu_free");
  const total = sumGpu(ps, "gpu_total");
  const queued = sumGpu(ps, "queued_gpus");

  return (
    <Card>
      <CardHeader>
        <div className="flex items-baseline justify-between">
          <CardTitle className="text-base">{cluster} <span className="ml-1 text-xs font-normal text-slate-500">slurm</span></CardTitle>
          <GpuAvailabilitySummary available={available} total={total} queued={queued} />
        </div>
        <CardDescription>{ps.length} partitions</CardDescription>
      </CardHeader>
      <CardContent>
        {q.isLoading && <LoadingState label="Loading partitions..." rows={3} />}
        {q.error && <ErrorState message={(q.error as Error).message} />}
        {!q.isLoading && !q.error && ps.length === 0 && (
          <EmptyState message="No partitions reported." />
        )}
        {ps.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
                <tr>
                  <th className="py-2 pr-4 font-medium">Partition</th>
                  <th className="py-2 pr-4 font-medium">GPUs available / total</th>
                  <th className="py-2 pr-4 font-medium">Queue</th>
                  <th className="py-2 pr-4 font-medium">Nodes available / total</th>
                  <th className="py-2 pr-4 font-medium">States</th>
                </tr>
              </thead>
              <tbody>
                {ps.map((p) => (
                  <QueueTooltipRow
                    key={p.name}
                    cluster={cluster}
                    partition={p.name}
                  >
                    <td className="py-2 pr-4 font-mono text-xs">
                      {p.name}
                      {p.is_default && <Badge variant="secondary" className="ml-1 text-[10px]">default</Badge>}
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">
                      <span className={p.gpu_free > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>
                        {p.gpu_free}
                      </span>
                      <span className="text-slate-400"> / {p.gpu_total}</span>
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">
                      <QueueLabel queuedGpus={p.queued_gpus} queuedJobs={p.queued_jobs} />
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">
                      {p.idle_nodes} / {p.total_nodes}
                    </td>
                    <td className="py-2 pr-4 text-xs">
                      {Object.entries(p.states).map(([st, n]) => (
                        <span key={st} className="mr-2 font-mono">
                          <span className={st === "idle" ? "text-green-600 dark:text-green-400" : "text-slate-500"}>{st}</span>:{n}
                        </span>
                      ))}
                    </td>
                  </QueueTooltipRow>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function MlxpPanel() {
  const q = useQuery({
    queryKey: ["mlxp-gpus"],
    queryFn: () => api<MlxpNode[]>("/api/mlxp/gpus"),
    refetchInterval: REFRESH_MS,
    retry: false,
  });
  const nodes = q.data ?? [];
  const available = sumGpu(nodes, "gpu_free");
  const total = sumGpu(nodes, "gpu_total");
  const queued = sumGpu(nodes, "queued_gpus");

  return (
    <Card>
      <CardHeader>
        <div className="flex items-baseline justify-between">
          <CardTitle className="text-base">MLXP <span className="ml-1 text-xs font-normal text-slate-500">naver, k8s</span></CardTitle>
          <GpuAvailabilitySummary available={available} total={total} queued={queued} />
        </div>
      </CardHeader>
      <CardContent>
        {q.isLoading && <LoadingState label="Loading MLXP GPUs..." rows={3} />}
        {q.error && <ErrorState message={(q.error as Error).message} />}
        {!q.isLoading && !q.error && nodes.length === 0 && (
          <EmptyState message="No MLXP GPU nodes reported." />
        )}
        {nodes.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
                <tr>
                  <th className="py-2 pr-4 font-medium">Node</th>
                  <th className="py-2 pr-4 font-medium">GPUs available / total</th>
                  <th className="py-2 pr-4 font-medium">Queue</th>
                </tr>
              </thead>
              <tbody>
                {nodes.map((n) => (
                  <QueueTooltipRow
                    key={n.name}
                    cluster="mlxp"
                    partition="mlxp"
                    node={n.name}
                  >
                    <td className="py-2 pr-4 font-mono text-xs">
                      {n.name}
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">
                      <span className={n.gpu_free > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>
                        {n.gpu_free}
                      </span>
                      <span className="text-slate-400"> / {n.gpu_total}</span>
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">
                      <QueueLabel queuedGpus={n.queued_gpus} queuedJobs={n.queued_jobs} />
                    </td>
                  </QueueTooltipRow>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function GpuAvailabilitySummary({
  available,
  total,
  queued,
}: {
  available: number;
  total: number;
  queued: number;
}) {
  return (
    <span className="font-mono text-sm">
      <span className={available > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>{available}</span>
      <span className="text-slate-400"> / {total} GPU available</span>
      <span className="text-slate-400"> · </span>
      <span className={queued > 0 ? "text-amber-600 dark:text-amber-400" : "text-slate-500"}>{queued}</span>
      <span className="text-slate-400"> GPU queued</span>
    </span>
  );
}

function QueueLabel({
  queuedGpus,
  queuedJobs,
}: {
  queuedGpus: number;
  queuedJobs: number;
}) {
  if (queuedGpus <= 0 && queuedJobs <= 0) {
    return <span className="text-slate-400">—</span>;
  }
  return (
    <span>
      <span className="text-amber-600 dark:text-amber-400">{queuedGpus}</span>
      <span className="text-slate-400"> GPU / {queuedJobs} {queuedJobs === 1 ? "job" : "jobs"}</span>
    </span>
  );
}

function QueueTooltipRow({
  cluster,
  partition,
  node,
  children,
}: {
  cluster: string;
  partition: string;
  node?: string;
  children: React.ReactNode;
}) {
  const triggerRef = useRef<HTMLTableRowElement>(null);
  const tooltipRef = useRef<HTMLSpanElement>(null);
  const [visible, setVisible] = useState(false);

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    const tooltip = tooltipRef.current;
    if (!trigger || !tooltip) return;

    const triggerRect = trigger.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const tooltipWidth = tooltipRect.width || 0;
    const tooltipHeight = tooltipRect.height || 0;
    const centeredLeft = triggerRect.left + triggerRect.width / 2 - tooltipWidth / 2;
    const left = Math.min(
      Math.max(TOOLTIP_VIEWPORT_GAP, centeredLeft),
      Math.max(TOOLTIP_VIEWPORT_GAP, window.innerWidth - tooltipWidth - TOOLTIP_VIEWPORT_GAP),
    );
    const top = Math.min(
      triggerRect.bottom + TOOLTIP_OFFSET,
      window.innerHeight - tooltipHeight - TOOLTIP_VIEWPORT_GAP,
    );

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }, []);

  useLayoutEffect(() => {
    if (visible) updatePosition();
  }, [visible, updatePosition]);

  useEffect(() => {
    if (!visible) return;
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [visible, updatePosition]);

  return (
    <tr
      ref={triggerRef}
      tabIndex={0}
      className={cn(
        "border-b border-slate-100 last:border-0 dark:border-slate-900",
        "cursor-help hover:bg-slate-50 focus-visible:bg-slate-50 focus-visible:outline-none dark:hover:bg-slate-900/40 dark:focus-visible:bg-slate-900/40",
      )}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {children}
      {visible && typeof document !== "undefined"
        ? createPortal(
            <span
              ref={tooltipRef}
              role="tooltip"
              className="pointer-events-none fixed z-[1000] w-max max-w-[min(32rem,calc(100vw-2rem))] whitespace-normal rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-normal leading-snug text-slate-700 shadow-lg [overflow-wrap:anywhere] dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200"
              style={{ left: TOOLTIP_VIEWPORT_GAP, top: TOOLTIP_VIEWPORT_GAP }}
            >
              <MonitorQueueTooltip cluster={cluster} partition={partition} node={node} />
            </span>,
            document.body,
          )
        : null}
    </tr>
  );
}

function MonitorQueueTooltip({
  cluster,
  partition,
  node,
}: {
  cluster: string;
  partition: string;
  node?: string;
}) {
  const qs = new URLSearchParams({ partition });
  if (node) qs.set("node", node);
  const queue = useQuery({
    queryKey: ["gpu-queue", cluster, partition, node ?? null],
    queryFn: () =>
      api<GpuQueueSnapshot>(
        `/api/clusters/${encodeURIComponent(cluster)}/gpu-queue?${qs}`,
      ),
    staleTime: 15_000,
    refetchInterval: REFRESH_MS,
    retry: false,
  });

  return (
    <GpuQueueTooltipContent
      snapshot={queue.data}
      loading={queue.isLoading}
      fetching={queue.isFetching}
      error={queue.error as Error | null}
    />
  );
}
