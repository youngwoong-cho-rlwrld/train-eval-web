"use client";

import { Loader2 } from "lucide-react";
import { useMemo, useState } from "react";
import { keepPreviousData, useIsFetching, useQueries, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { api, type Job, type JobProgress } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CopyButton } from "@/components/copy-button";
import { CopyCheckpointDialog } from "@/components/copy-checkpoint-dialog";
import { ResumeJobButton } from "@/components/resume-job-button";
import { RefreshButton } from "@/components/refresh-button";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";
import { JobStateBadge } from "@/components/job-state-badge";
import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { PendingQueueLabel, pendingQueuePositionLabel } from "@/components/pending-queue-label";
import { parseJobTimestampMs } from "@/lib/job-time";
import { activeProgressLabel, hasRenderableProgress, stepEta } from "@/lib/job-progress";
import { Th } from "@/components/table";
import { JobTimestamp } from "@/components/job-timestamp";
import { ProgressBar } from "@/components/progress-bar";
import { JobLink } from "@/components/job-link";
import {
  canCopyCheckpoint,
  canResumeJob,
  canRetryJob,
  isActiveJobState,
  isRunningOrCompletingJobState,
  isTrainJobPhase,
  jobPhase,
  normalizeJobPhase,
  resubmitSourceLabel,
  type JobPhase,
} from "@/lib/job-status";

const REFRESH_MS = 60_000;

// Query-key families that make up the /jobs view. The refresh button
// invalidates them and the in-flight indicator watches them, so keep the two
// derived from one list (N163).
const JOB_QUERY_FAMILIES = ["jobs", "job-progress", "gpu-queue"] as const;

type ClusterJobError = { cluster: string; error: string };

// Merge per-cluster job queries: jobs as they arrive, initial-loading only
// until at least one cluster responds, per-cluster errors, and which clusters
// are still in flight (the MLXP archived scan is slow, so it streams in last).
function mergeJobQueries(
  queries: UseQueryResult<Job[]>[],
  clusterNames: string[],
  clustersLoading: boolean,
): { jobs: Job[]; initialLoading: boolean; errors: ClusterJobError[]; probing: string[] } {
  const jobs = queries.flatMap((q) => q.data ?? []);
  const anyLoaded = queries.some((q) => q.data !== undefined);
  const initialLoading = clustersLoading || (clusterNames.length > 0 && !anyLoaded);
  const errors = clusterNames.flatMap((c, i) =>
    queries[i]?.error ? [{ cluster: c, error: (queries[i].error as Error).message }] : [],
  );
  const probing = clusterNames.filter((_, i) => queries[i]?.isLoading);
  return { jobs, initialLoading, errors, probing };
}

export default function JobsPage() {
  const qc = useQueryClient();
  const [draftHistoryRange, setDraftHistoryRange] = useState(() => defaultHistoryRange());
  const [appliedHistoryRange, setAppliedHistoryRange] = useState(() => defaultHistoryRange());
  const [recentNameFilter, setRecentNameFilter] = useState("");
  const [recentPhaseFilter, setRecentPhaseFilter] = useState<RecentPhaseFilter>("all");
  const [recentStateFilter, setRecentStateFilter] = useState("all");
  const historyQuery = useMemo(
    () => historyRangeQuery(appliedHistoryRange),
    [appliedHistoryRange],
  );
  const hasHistoryRangeChanges =
    draftHistoryRange.startDate !== appliedHistoryRange.startDate ||
    draftHistoryRange.endDate !== appliedHistoryRange.endDate;

  const clustersQuery = useQuery({
    queryKey: ["clusters"],
    queryFn: () => api<{ clusters: string[] }>("/api/clusters").then((d) => d.clusters),
    staleTime: 5 * 60_000,
  });
  const clusterNames = clustersQuery.data ?? [];

  // One query per cluster so the slow MLXP archived-runs scan can't block the
  // fast slurm clusters; each section renders as clusters respond.
  const activeQueries = useQueries({
    queries: clusterNames.map((c) => ({
      queryKey: ["jobs", "active", c],
      queryFn: () =>
        api<{ jobs: Job[] }>(`/api/jobs?cluster=${encodeURIComponent(c)}&hours=0`).then((d) => d.jobs),
      refetchInterval: REFRESH_MS,
      placeholderData: keepPreviousData,
    })),
  });
  const recentQueries = useQueries({
    queries: clusterNames.map((c) => ({
      queryKey: ["jobs", "recent", c, historyQuery],
      queryFn: () =>
        api<{ jobs: Job[] }>(`/api/jobs?cluster=${encodeURIComponent(c)}&${historyQuery}`).then((d) => d.jobs),
      refetchInterval: REFRESH_MS,
      placeholderData: keepPreviousData,
    })),
  });
  const activeMerged = mergeJobQueries(activeQueries, clusterNames, clustersQuery.isLoading);
  const recentMerged = mergeJobQueries(recentQueries, clusterNames, clustersQuery.isLoading);

  const refreshAll = () => {
    for (const family of JOB_QUERY_FAMILIES) {
      qc.invalidateQueries({ queryKey: [family] });
    }
  };

  const isFetching =
    useIsFetching({
      predicate: (q) =>
        JOB_QUERY_FAMILIES.includes(q.queryKey[0] as (typeof JOB_QUERY_FAMILIES)[number]),
    }) > 0;

  const active = useMemo(() => {
    return activeMerged.jobs
      .filter((j) => isActiveJobState(j.state))
      .sort((a, b) => compareActiveDesc(a, b));
  }, [activeMerged.jobs]);
  const finished = useMemo(() => {
    return recentMerged.jobs
      .filter((j) => !isActiveJobState(j.state))
      .sort((a, b) => compareEndedDesc(a, b));
  }, [recentMerged.jobs]);
  const recentStateOptions = useMemo(() => {
    const values = new Set(finished.map((j) => normalizeStateFilterValue(j.state)));
    return Array.from(values).sort();
  }, [finished]);
  const filteredFinished = useMemo(
    () =>
      finished.filter((job) =>
        recentJobMatchesFilters(job, {
          name: recentNameFilter,
          phase: recentPhaseFilter,
          state: recentStateFilter,
        }),
      ),
    [finished, recentNameFilter, recentPhaseFilter, recentStateFilter],
  );

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
          {activeMerged.initialLoading && <LoadingState label="Loading active jobs..." rows={4} />}
          {activeMerged.errors.map((e) => (
            <ErrorState key={`active-${e.cluster}`} message={`${e.cluster}: ${e.error}`} />
          ))}
          {!activeMerged.initialLoading && activeMerged.probing.length === 0 && active.length === 0 && (
            <EmptyState message="No active jobs." />
          )}
          {active.length > 0 && <JobTable rows={active} showProgress showActions={false} />}
          {!activeMerged.initialLoading && activeMerged.probing.length > 0 && (
            <p className="mt-2 text-xs text-slate-400">loading {activeMerged.probing.join(", ")}…</p>
          )}
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Recent</CardTitle>
          <CardDescription>
            {filteredFinished.length} of {finished.length} finished {finished.length === 1 ? "job" : "jobs"} from{" "}
            {formatDateRange(appliedHistoryRange)}.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-5 flex flex-wrap items-end gap-3">
            <div className="space-y-1">
              <Label htmlFor="recent-name" className="text-xs text-slate-500">Name</Label>
              <Input
                id="recent-name"
                value={recentNameFilter}
                onChange={(e) => setRecentNameFilter(e.target.value)}
                placeholder="job or variant"
                className="h-8 w-[180px] font-mono text-xs"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="recent-phase" className="text-xs text-slate-500">Phase</Label>
              <Select value={recentPhaseFilter} onValueChange={(v) => setRecentPhaseFilter(v as RecentPhaseFilter)}>
                <SelectTrigger id="recent-phase" className="h-8 w-[110px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="train">train</SelectItem>
                  <SelectItem value="resume">resume</SelectItem>
                  <SelectItem value="eval">eval</SelectItem>
                  <SelectItem value="other">other</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label htmlFor="recent-state" className="text-xs text-slate-500">State</Label>
              <Select value={recentStateFilter} onValueChange={setRecentStateFilter}>
                <SelectTrigger id="recent-state" className="h-8 w-[150px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  {recentStateOptions.map((state) => (
                    <SelectItem key={state} value={state}>
                      {state.toLowerCase()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label htmlFor="recent-start" className="text-xs text-slate-500">From</Label>
              <Input
                id="recent-start"
                type="date"
                value={draftHistoryRange.startDate}
                onChange={(e) =>
                  setDraftHistoryRange((current) => ({
                    ...current,
                    startDate: e.target.value,
                  }))
                }
                className="h-8 w-[150px]"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="recent-end" className="text-xs text-slate-500">To</Label>
              <Input
                id="recent-end"
                type="date"
                value={draftHistoryRange.endDate}
                onChange={(e) =>
                  setDraftHistoryRange((current) => ({
                    ...current,
                    endDate: e.target.value,
                  }))
                }
                className="h-8 w-[150px]"
              />
            </div>
            <Button
              type="button"
              size="sm"
              onClick={() => setAppliedHistoryRange(draftHistoryRange)}
              disabled={!hasHistoryRangeChanges}
              className="h-8"
            >
              Apply
            </Button>
            {recentMerged.probing.length > 0 && (
              <span className="flex h-8 items-center gap-1.5 text-xs text-slate-400">
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-slate-400" />
                loading {recentMerged.probing.join(", ")}…
              </span>
            )}
          </div>
          {recentMerged.initialLoading && <LoadingState label="Loading recent jobs..." rows={4} />}
          {recentMerged.errors.map((e) => (
            <ErrorState key={`recent-${e.cluster}`} message={`${e.cluster}: ${e.error}`} />
          ))}
          {!recentMerged.initialLoading && recentMerged.probing.length === 0 && finished.length === 0 && (
            <EmptyState message="Nothing in this window." />
          )}
          {!recentMerged.initialLoading && finished.length > 0 && filteredFinished.length === 0 && (
            <EmptyState message="No jobs match these filters." />
          )}
          {filteredFinished.length > 0 && <JobTable rows={filteredFinished} />}
        </CardContent>
      </Card>
    </div>
  );
}

type HistoryRange = {
  startDate: string;
  endDate: string;
};

type RecentPhaseFilter = JobPhase | "all";

type RecentJobFilters = {
  name: string;
  phase: RecentPhaseFilter;
  state: string;
};

function normalizeStateFilterValue(state: string) {
  const head = state.trim().split(/\s+/)[0] ?? "";
  return head.toUpperCase() || "UNKNOWN";
}

function recentJobMatchesFilters(job: Job, filters: RecentJobFilters) {
  const name = filters.name.trim().toLowerCase();
  if (name) {
    const haystack = [
      job.job_id,
      job.job_name,
      job.variant ?? "",
      job.cluster,
      job.partition,
      job.nodelist,
    ].join(" ").toLowerCase();
    if (!haystack.includes(name)) return false;
  }
  const phase = normalizeJobPhase(job.phase) ?? jobPhase(job.job_name);
  if (filters.phase !== "all" && phase !== filters.phase) return false;
  if (filters.state !== "all" && normalizeStateFilterValue(job.state) !== filters.state) return false;
  return true;
}

function defaultHistoryRange(): HistoryRange {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - 1);
  return {
    startDate: dateInputValue(start),
    endDate: dateInputValue(end),
  };
}

function dateInputValue(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function orderedDateRange(range: HistoryRange): HistoryRange {
  if (!range.startDate || !range.endDate || range.startDate <= range.endDate) {
    return range;
  }
  return {
    startDate: range.endDate,
    endDate: range.startDate,
  };
}

function historyRangeQuery(range: HistoryRange): string {
  const ordered = orderedDateRange(range);
  const params = new URLSearchParams();
  if (ordered.startDate) {
    params.set("start", `${ordered.startDate}T00:00:00`);
  }
  if (ordered.endDate) {
    params.set("end", `${ordered.endDate}T23:59:59`);
  }
  if (!params.has("start") && !params.has("end")) {
    params.set("hours", "24");
  }
  return params.toString();
}

function formatDateRange(range: HistoryRange): string {
  const ordered = orderedDateRange(range);
  if (ordered.startDate && ordered.endDate) {
    return `${ordered.startDate} to ${ordered.endDate}`;
  }
  if (ordered.startDate) {
    return `${ordered.startDate} onward`;
  }
  if (ordered.endDate) {
    return `before ${ordered.endDate}`;
  }
  return "the last 24h";
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
            {showProgress && <Th>Experiment</Th>}
            <Th>Job name</Th>
            {showActions && <Th>Actions</Th>}
            <Th>Cluster</Th>
            <Th>Partition</Th>
            <Th>Node</Th>
            {showProgress && <Th>Server remaining</Th>}
            <Th>Started</Th>
            {!showProgress && <Th>Ended</Th>}
            <Th>Elapsed</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((j) => {
            const phase = normalizeJobPhase(j.phase) ?? jobPhase(j.job_name);
            return (
              <tr key={`${j.cluster}-${j.job_id}`} className="border-b border-slate-100 last:border-0 hover:bg-slate-50 dark:border-slate-900 dark:hover:bg-slate-900/40">
                <td className="py-2 pr-4 font-mono">
                  <div className="flex items-center gap-1">
                    <JobIdLink cluster={j.cluster} jobId={j.job_id} />
                    <CopyButton value={j.job_id} title="Copy job ID" />
                  </div>
                  {j.resume_of && (
                    <span className="ml-1 block max-w-[26ch] text-xs text-slate-500">
                      ({resubmitSourceLabel(j.resubmit_action)}{" "}
                      <JobIdLink
                        cluster={j.cluster}
                        jobId={j.resume_of}
                        fixedWidth={false}
                        className="dark:text-blue-400"
                      />
                      )
                    </span>
                  )}
                </td>
                <Td><PhaseBadge phase={phase} /></Td>
                <Td>
                  <JobStateBadge state={j.state} />
                </Td>
                {showProgress && (
                  <Td className="min-w-[300px]">
                    <ActiveProgressCell job={j} />
                  </Td>
                )}
                {showProgress && (
                  <Td className="font-mono text-xs">
                    {j.variant ? (
                      <div className="flex items-center gap-1">
                        <ImmediateTooltip content={j.variant} className="max-w-[220px]">
                          <span className="truncate">{j.variant}</span>
                        </ImmediateTooltip>
                        <CopyButton value={j.variant} title="Copy experiment" />
                      </div>
                    ) : (
                      <span className="text-slate-400">—</span>
                    )}
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
                    {(() => {
                      const resumable = canResumeJob(j);
                      const retryable = canRetryJob(j);
                      const copyable = canCopyCheckpoint({ state: j.state, phase });
                      return (
                        <div className="flex items-center gap-2">
                          {resumable && (
                            <ResumeJobButton
                              cluster={j.cluster}
                              jobId={j.job_id}
                              phase={phase}
                              variant={j.variant}
                              jobName={j.job_name}
                              className="h-7 px-2 text-xs"
                            />
                          )}
                          {retryable && (
                            <ResumeJobButton
                              cluster={j.cluster}
                              jobId={j.job_id}
                              phase={phase}
                              variant={j.variant}
                              jobName={j.job_name}
                              action="retry"
                              className="h-7 px-2 text-xs"
                            />
                          )}
                          {copyable && <CopyCheckpointShortcut job={j} />}
                          {!resumable && !retryable && !copyable && (
                            <span className="text-slate-400">—</span>
                          )}
                        </div>
                      );
                    })()}
                  </Td>
                )}
                <Td>{j.cluster}</Td>
                <Td className="font-mono text-xs">{j.partition}</Td>
                <Td className="font-mono text-xs text-slate-500 min-w-[180px]">{j.nodelist}</Td>
                {showProgress && (
                  <Td className="font-mono text-xs">
                    {j.time_left || <span className="text-slate-400">—</span>}
                  </Td>
                )}
                <Td className="font-mono text-xs"><JobTimestamp iso={j.start} /></Td>
                {!showProgress && (
                  <Td className="font-mono text-xs"><JobTimestamp iso={j.end} /></Td>
                )}
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
  const shouldFetch = isRunningOrCompletingJobState(job.state);
  const progressQuery = useQuery({
    queryKey: ["job-progress", job.cluster, job.job_id],
    queryFn: () =>
      api<JobProgress>(`/api/jobs/${job.cluster}/${job.job_id}/progress`),
    enabled: shouldFetch,
    refetchInterval: REFRESH_MS,
    staleTime: 10_000,
    retry: false,
  });

  if (!shouldFetch) {
    const state = job.state.toLowerCase();
    const label = /^PENDING$/i.test(job.state) && job.queue_position
      ? pendingQueuePositionLabel(state, job.queue_position)
      : state;
    if (/^PENDING$/i.test(job.state) && job.partition) {
      return <PendingQueueLabel job={job} fallbackLabel={label} />;
    }
    return <span className="text-xs text-slate-500">{label}</span>;
  }

  if (progressQuery.isLoading) {
    return (
      <div className="space-y-1.5" aria-busy="true">
        <div className="h-3 w-24 animate-pulse rounded bg-slate-100 dark:bg-slate-800" />
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800" />
      </div>
    );
  }

  if (progressQuery.error) {
    return (
      <ImmediateTooltip content={(progressQuery.error as Error).message}>
        <span className="text-xs text-slate-500">unavailable</span>
      </ImmediateTooltip>
    );
  }

  const d = progressQuery.data;
  const p = d?.progress;
  const percent = p?.percent ?? null;
  const showBar = hasRenderableProgress(p);
  if (!showBar) {
    return (
      <span className="text-xs text-slate-500">
        {activeProgressLabel(d?.phase, p) ?? "waiting for progress"}
      </span>
    );
  }

  const effectivePercent = Math.max(0, Math.min(100, percent ?? 0));
  const label = activeProgressLabel(d?.phase, p) ?? `${effectivePercent.toFixed(1)}%`;
  const eta = stepEta(d?.elapsed, p?.current_step, p?.max_steps, d?.phase);

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 text-xs">
        <span className="min-w-[8rem] flex-1 break-words font-mono leading-snug" title={label}>
          {label}
        </span>
        <span className="shrink-0 whitespace-nowrap font-mono text-slate-500">
          {effectivePercent.toFixed(1)}%
          {eta && (
            <ImmediateTooltip content={eta.etaTitle}>
              <span> · ~{eta.etaLabel}</span>
            </ImmediateTooltip>
          )}
        </span>
      </div>
      <ProgressBar percent={effectivePercent} />
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

function JobIdLink({
  cluster,
  jobId,
  className = "",
  fixedWidth = true,
}: {
  cluster: string;
  jobId: string;
  className?: string;
  fixedWidth?: boolean;
}) {
  const widthClass = fixedWidth ? "inline-block min-w-[23ch]" : "inline";
  return (
    <ImmediateTooltip content={jobId} className="min-w-0">
      <JobLink
        cluster={cluster}
        jobId={jobId}
        className={`${widthClass} whitespace-nowrap ${className}`}
      >
        {shortJobId(jobId)}
      </JobLink>
    </ImmediateTooltip>
  );
}

function shortJobId(jobId: string): string {
  // MLXP job IDs are `<user>-<job-name>`, so they share a long common prefix
  // (e.g. `youngwoong-train-…`). Truncating from the end would collapse every
  // row to the same prefix and drop the distinguishing tail, so keep both ends
  // and elide the middle instead.
  const maxDisplayLength = 22;
  if (jobId.length <= maxDisplayLength) return jobId;
  const ellipsis = "…";
  const keep = maxDisplayLength - ellipsis.length;
  const head = Math.ceil(keep / 2);
  const tail = keep - head;
  return `${jobId.slice(0, head)}${ellipsis}${jobId.slice(jobId.length - tail)}`;
}

function PhaseBadge({ phase }: { phase: JobPhase }) {
  if (isTrainJobPhase(phase)) return <Badge variant="default">{phase}</Badge>;
  if (phase === "eval") return <Badge variant="outline">eval</Badge>;
  return <Badge variant="secondary">other</Badge>;
}

function compareEndedDesc(a: Job, b: Job): number {
  const aEnd = parseJobTimestampMs(a.end);
  const bEnd = parseJobTimestampMs(b.end);
  if (aEnd !== bEnd) return bEnd - aEnd;

  const aStart = parseJobTimestampMs(a.start);
  const bStart = parseJobTimestampMs(b.start);
  if (aStart !== bStart) return bStart - aStart;
  return compareJobIdDesc(a.job_id, b.job_id);
}

function compareActiveDesc(a: Job, b: Job): number {
  const aStart = parseJobTimestampMs(a.start);
  const bStart = parseJobTimestampMs(b.start);
  if (aStart !== bStart) return bStart - aStart;

  const aEnd = parseJobTimestampMs(a.end);
  const bEnd = parseJobTimestampMs(b.end);
  if (aEnd !== bEnd) return bEnd - aEnd;
  return compareJobIdDesc(a.job_id, b.job_id);
}

function compareJobIdDesc(a: string, b: string): number {
  return b.localeCompare(a, undefined, { numeric: true, sensitivity: "base" });
}

function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={`py-2 pr-4 ${className ?? ""}`}>{children}</td>;
}
