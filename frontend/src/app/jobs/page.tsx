"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useIsFetching, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Job, type JobDetails } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CopyButton } from "@/components/copy-button";
import { CopyCheckpointDialog } from "@/components/copy-checkpoint-dialog";
import { ResumeJobButton } from "@/components/resume-job-button";
import { RefreshButton } from "@/components/refresh-button";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";
import { JobStateBadge } from "@/components/job-state-badge";
import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { formatJobTimestamp, parseJobTimestampMs } from "@/lib/job-time";

const REFRESH_MS = 60_000;

const ACTIVE_STATES = new Set(["RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"]);

export default function JobsPage() {
  const qc = useQueryClient();
  const [hours, setHours] = useState<string>("24");

  const { data, isLoading, error } = useQuery({
    queryKey: ["jobs", hours],
    queryFn: () =>
      api<{ jobs: Job[] }>(`/api/jobs?hours=${hours}`).then((d) => d.jobs),
    refetchInterval: REFRESH_MS,
  });

  const refreshAll = () => {
    qc.invalidateQueries({ queryKey: ["jobs"] });
    qc.invalidateQueries({ queryKey: ["job-details"] });
  };

  const isFetching =
    useIsFetching({
      predicate: (q) => {
        const k = q.queryKey[0];
        return k === "jobs" || k === "job-details";
      },
    }) > 0;

  const { active, finished } = useMemo(() => {
    const all = data ?? [];
    const active = all
      .filter((j) => ACTIVE_STATES.has(j.state))
      .sort((a, b) => Number(b.job_id) - Number(a.job_id));
    const finished = all
      .filter((j) => !ACTIVE_STATES.has(j.state))
      .sort((a, b) => compareEndedDesc(a, b));
    return { active, finished };
  }, [data]);

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Jobs</h1>
        <RefreshButton
          isFetching={isFetching}
          onRefresh={refreshAll}
          intervalMs={REFRESH_MS}
        />
      </div>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Active</CardTitle>
          <CardDescription>{active.length} {active.length === 1 ? "job" : "jobs"} in queue or running.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading && <LoadingState label="Loading active jobs..." rows={4} />}
          {error && <ErrorState message={(error as Error).message} />}
          {!isLoading && !error && active.length === 0 && (
            <EmptyState message="No active jobs." />
          )}
          {active.length > 0 && <JobTable rows={active} showProgress showActions={false} />}
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader className="flex flex-row items-start justify-between gap-4">
          <div>
            <CardTitle>Recent</CardTitle>
            <CardDescription>{finished.length} finished {finished.length === 1 ? "job" : "jobs"} in the last {hours}h.</CardDescription>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-500">history window:</span>
            <Select value={hours} onValueChange={setHours}>
              <SelectTrigger className="h-8 w-[120px]"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="6">6 hours</SelectItem>
                <SelectItem value="24">24 hours</SelectItem>
                <SelectItem value="72">3 days</SelectItem>
                <SelectItem value="168">1 week</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading && <LoadingState label="Loading recent jobs..." rows={4} />}
          {error && <ErrorState message={(error as Error).message} />}
          {!isLoading && !error && finished.length === 0 && (
            <EmptyState message="Nothing in this window." />
          )}
          {finished.length > 0 && <JobTable rows={finished} />}
        </CardContent>
      </Card>
    </div>
  );
}

function JobTable({
  rows,
  showProgress = false,
  showActions = true,
}: {
  rows: Job[];
  showProgress?: boolean;
  showActions?: boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
          <tr>
            <Th>Job ID</Th>
            <Th>Phase</Th>
            <Th>State</Th>
            {showProgress && <Th>Progress</Th>}
            <Th>Name</Th>
            {showActions && <Th>Actions</Th>}
            <Th>Cluster</Th>
            <Th>Partition</Th>
            <Th>Node</Th>
            <Th>Started</Th>
            <Th>Ended</Th>
            <Th>Elapsed</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((j) => {
            const phase = phaseOf(j.job_name);
            return (
              <tr key={`${j.cluster}-${j.job_id}`} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 dark:border-slate-900 dark:hover:bg-slate-900/40">
                <td className="py-2 pr-4 font-mono">
                  <Link href={`/jobs/${j.cluster}/${j.job_id}`} className="text-blue-600 hover:underline">
                    {j.job_id}
                  </Link>
                </td>
                <Td><PhaseBadge phase={phase} /></Td>
                <Td>
                  <JobStateBadge state={j.state} />
                </Td>
                {showProgress && (
                  <Td className="min-w-[220px]">
                    <ActiveProgressCell job={j} />
                  </Td>
                )}
                <td className="py-2 pr-4 font-mono text-xs">
                  <div className="flex items-center gap-1">
                    <ImmediateTooltip content={j.job_name} className="max-w-[240px]">
                      <span className="truncate">{j.job_name}</span>
                    </ImmediateTooltip>
                    <CopyButton value={j.job_name} title="Copy job name" />
                  </div>
                </td>
                {showActions && (
                  <Td>
                    <div className="flex items-center gap-2">
                      {canResumeJob(j) && (
                        <ResumeJobButton
                          cluster={j.cluster}
                          jobId={j.job_id}
                          phase={phase}
                          jobName={j.job_name}
                          className="h-7 px-2 text-xs"
                        />
                      )}
                      {canCopyCheckpoint(j, phase) && (
                        <CopyCheckpointShortcut job={j} />
                      )}
                      {!canResumeJob(j) && !canCopyCheckpoint(j, phase) && (
                        <span className="text-slate-400">—</span>
                      )}
                    </div>
                  </Td>
                )}
                <Td>{j.cluster}</Td>
                <Td className="font-mono text-xs">{j.partition}</Td>
                <Td className="font-mono text-xs text-slate-500 min-w-[180px]">{j.nodelist}</Td>
                <Td className="font-mono text-xs"><Timestamp iso={j.start} cluster={j.cluster} /></Td>
                <Td className="font-mono text-xs"><Timestamp iso={j.end} cluster={j.cluster} /></Td>
                <Td className="font-mono text-xs">{j.elapsed}</Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ActiveProgressCell({ job }: { job: Job }) {
  const shouldFetch = /^(RUNNING|COMPLETING)$/i.test(job.state);
  const details = useQuery({
    queryKey: ["job-details", job.cluster, job.job_id, "progress"],
    queryFn: () =>
      api<JobDetails>(`/api/jobs/${job.cluster}/${job.job_id}/details`),
    enabled: shouldFetch,
    refetchInterval: REFRESH_MS,
    staleTime: 10_000,
    retry: false,
  });

  if (!shouldFetch) {
    return <span className="text-xs text-slate-500">{job.state.toLowerCase()}</span>;
  }

  if (details.isLoading) {
    return (
      <div className="space-y-1.5" aria-busy="true">
        <div className="h-3 w-24 animate-pulse rounded bg-slate-100 dark:bg-slate-800" />
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800" />
      </div>
    );
  }

  if (details.error) {
    return (
      <ImmediateTooltip content={(details.error as Error).message}>
        <span className="text-xs text-slate-500">unavailable</span>
      </ImmediateTooltip>
    );
  }

  const p = details.data?.progress;
  const percent = p?.percent ?? null;
  const hasStepProgress =
    p?.current_step !== null &&
    p?.current_step !== undefined &&
    p.max_steps !== null &&
    p.max_steps !== undefined;
  const hasRunProgress =
    p?.completed_runs !== null &&
    p?.completed_runs !== undefined &&
    p.total_runs !== null &&
    p.total_runs !== undefined;
  const showBar = percent !== null || hasStepProgress || hasRunProgress;
  if (!showBar) {
    return <span className="text-xs text-slate-500">{p?.current_label ?? "waiting for progress"}</span>;
  }

  const effectivePercent = Math.max(0, Math.min(100, percent ?? 0));
  const label = p?.current_label ?? `${effectivePercent.toFixed(1)}%`;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="min-w-0 truncate font-mono" title={label}>
          {label}
        </span>
        <span className="shrink-0 font-mono text-slate-500">
          {effectivePercent.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
        <div
          className="h-full rounded-full bg-slate-900 transition-all dark:bg-slate-50"
          style={{ width: `${effectivePercent}%` }}
        />
      </div>
    </div>
  );
}

function CopyCheckpointShortcut({ job }: { job: Job }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className="h-7 px-2 text-xs"
        onClick={() => setOpen(true)}
      >
        Copy checkpoint
      </Button>
      <CopyCheckpointDialog
        open={open}
        onOpenChange={setOpen}
        cluster={job.cluster}
        jobId={job.job_id}
      />
    </>
  );
}

function Timestamp({ iso, cluster }: { iso?: string | null; cluster: string }) {
  const formatted = formatJobTimestamp(iso, cluster);
  if (!formatted) return <span className="text-slate-400">—</span>;
  return (
    <ImmediateTooltip content={formatted.full}>
      <span>{formatted.short}</span>
    </ImmediateTooltip>
  );
}

function phaseOf(jobName: string): "train" | "resume" | "eval" | "other" {
  // Slurm names:  train_<variant>_<cluster>_<partition>_<ts>
  // MLXP names:   train_<variant>_<ts> or <user>-train-<variant>-<ts>
  // Anchor on (start | hyphen | underscore) + phase + (hyphen | underscore).
  const m = jobName.match(/(?:^|[-_])(train|resume|eval)[-_]/);
  return (m?.[1] as "train" | "resume" | "eval") ?? "other";
}

function PhaseBadge({ phase }: { phase: ReturnType<typeof phaseOf> }) {
  if (phase === "train" || phase === "resume") return <Badge variant="default">{phase}</Badge>;
  if (phase === "eval") return <Badge variant="outline">eval</Badge>;
  return <Badge variant="secondary">other</Badge>;
}

function isTimeout(state: string): boolean {
  return state.toUpperCase().startsWith("TIMEOUT");
}

function canResumeJob(job: Job): boolean {
  return job.cluster !== "mlxp" && isTimeout(job.state);
}

function canCopyCheckpoint(job: Job, phase: ReturnType<typeof phaseOf>): boolean {
  return (
    (phase === "train" || phase === "resume") &&
    job.state.toUpperCase().startsWith("COMPLET")
  );
}

function compareEndedDesc(a: Job, b: Job): number {
  const aEnd = parseJobTimestampMs(a.end, a.cluster);
  const bEnd = parseJobTimestampMs(b.end, b.cluster);
  if (aEnd !== bEnd) return bEnd - aEnd;

  const aStart = parseJobTimestampMs(a.start, a.cluster);
  const bStart = parseJobTimestampMs(b.start, b.cluster);
  if (aStart !== bStart) return bStart - aStart;
  return Number(b.job_id) - Number(a.job_id);
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="py-2 pr-4 font-medium whitespace-nowrap">{children}</th>;
}
function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={`py-2 pr-4 ${className ?? ""}`}>{children}</td>;
}
