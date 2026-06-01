"use client";

import { useQuery } from "@tanstack/react-query";

import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { api, type GpuQueueSnapshot, type Job } from "@/lib/api";

const REFRESH_MS = 60_000;

export function PendingQueueLabel({ job, fallbackLabel }: { job: Job; fallbackLabel: string }) {
  const jobIdParam = job.cluster === "mlxp" ? `&job_id=${encodeURIComponent(job.job_id)}` : "";
  const queue = useQuery({
    queryKey: ["gpu-queue", job.cluster, job.partition, job.cluster === "mlxp" ? job.job_id : null],
    queryFn: () =>
      api<GpuQueueSnapshot>(
        `/api/clusters/${encodeURIComponent(job.cluster)}/gpu-queue?partition=${encodeURIComponent(job.partition)}${jobIdParam}`,
      ),
    staleTime: 15_000,
    refetchInterval: REFRESH_MS,
    retry: false,
  });
  const queuePosition = queue.data ? queuePositionInSnapshot(queue.data, job.job_id) : null;
  const label = queue.data && queuePosition
    ? pendingQueuePositionLabel(job.state.toLowerCase(), queuePosition, queue.data.queue.length)
    : fallbackLabel;

  return (
    <ImmediateTooltip
      side="bottom"
      content={
        <PendingQueueTooltip
          currentJobId={job.job_id}
          snapshot={queue.data}
          loading={queue.isLoading}
          error={queue.error as Error | null}
        />
      }
      contentClassName="px-3 py-2"
    >
      <span className="text-xs text-slate-500">{label}</span>
    </ImmediateTooltip>
  );
}

export function pendingQueuePositionLabel(state: string, position: number | null | undefined, total?: number) {
  if (!position) return state;
  if (total !== undefined) return `${state} (queue pos ${position}/${total})`;
  return `${state} (queue pos ${position})`;
}

function queuePositionInSnapshot(snapshot: GpuQueueSnapshot, jobId: string) {
  const index = snapshot.queue.findIndex((job) => job.job_id === jobId);
  return index >= 0 ? index + 1 : null;
}

function PendingQueueTooltip({
  currentJobId,
  snapshot,
  loading,
  error,
}: {
  currentJobId: string;
  snapshot?: GpuQueueSnapshot;
  loading: boolean;
  error: Error | null;
}) {
  if (loading) {
    return <span className="font-mono text-xs text-slate-500">loading queue...</span>;
  }
  if (error) {
    return <span className="text-xs text-red-600 dark:text-red-400">{error.message}</span>;
  }
  if (!snapshot) {
    return <span className="font-mono text-xs text-slate-500">queue unavailable</span>;
  }

  return (
    <div className="space-y-1 font-mono text-[11px] leading-snug">
      {snapshot.nodes.map((node) => (
        <div key={node.name} className="grid grid-cols-[4.5rem_3rem_auto_2.5rem_4rem] items-center gap-2">
          <span className="text-slate-500" title={node.name}>{nodeLabel(node.name)}</span>
          <span>{node.gpu_type ?? "GPU"}</span>
          <span className="tracking-wider">{gpuSquares(node.gpu_used, node.gpu_total)}</span>
          <span className="text-right text-slate-500">
            {node.gpu_used}/{node.gpu_total}
          </span>
          <span className={nodeStateClassName(node.state)} title={node.reason ?? undefined}>
            {node.state ?? "-"}
          </span>
        </div>
      ))}
      <div className="border-t border-slate-200 pt-1 dark:border-slate-800">
        <span className="text-slate-500">gpu queue ({snapshot.queue.length})</span>{" "}
        {snapshot.queue.length > 0 ? (
          <span className="inline-flex flex-wrap gap-x-2 gap-y-1">
            {snapshot.queue.map((job) => (
              <span
                key={job.job_id}
                className={
                  job.job_id === currentJobId
                    ? "font-semibold text-green-600 dark:text-green-400"
                    : undefined
                }
                title={`${job.job_id}${job.reason ? ` · ${job.reason}` : ""}${job.name ? ` · ${job.name}` : ""}`}
              >
                {job.requested_gpus}
                {job.job_id === currentJobId ? "*" : ""}
              </span>
            ))}
          </span>
        ) : (
          <span>-</span>
        )}
      </div>
    </div>
  );
}

function nodeLabel(name: string) {
  const l40s = name.match(/^l40s-gpu-(dy|st)-g6e-12xl(?:-debug)?-(\d+)$/);
  if (l40s) {
    return name.includes("-debug-") ? `debug-${l40s[2]}` : `${l40s[1]}-${l40s[2]}`;
  }
  const h200 = name.match(/^rlwrld-gpu-.*-p5en-48xl-(\d+)$/);
  if (h200) return `node ${h200[1]}`;
  const match = name.match(/-(\d+)$/);
  return match ? `node ${match[1]}` : name;
}

function nodeStateClassName(state: string | null) {
  const normalized = state?.toLowerCase() ?? "";
  if (normalized === "idle") return "text-green-600 dark:text-green-400";
  if (normalized.startsWith("idle")) return "text-amber-600 dark:text-amber-400";
  return "text-slate-500";
}

function gpuSquares(used: number, total: number) {
  const safeTotal = Math.max(0, total);
  const safeUsed = Math.max(0, Math.min(used, safeTotal));
  return "■".repeat(safeUsed) + "□".repeat(safeTotal - safeUsed);
}
