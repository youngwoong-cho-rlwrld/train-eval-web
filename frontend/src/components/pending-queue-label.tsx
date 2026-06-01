"use client";

import { useQuery } from "@tanstack/react-query";

import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { GpuQueueTooltipContent } from "@/components/gpu-queue-visualization";
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
        <GpuQueueTooltipContent
          currentJobId={job.job_id}
          snapshot={queue.data}
          loading={queue.isLoading}
          fetching={queue.isFetching}
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
