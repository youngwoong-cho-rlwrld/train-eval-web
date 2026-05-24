"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useIsFetching, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Job, type JobDetails } from "@/lib/api";
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
import { formatDuration, parseSlurmDuration } from "@/lib/duration";
import { formatJobTimestamp, parseJobTimestampMs } from "@/lib/job-time";
import {
  isActiveJobState,
  isCompletedJobState,
  isTimeoutJobState,
  isTrainJobPhase,
  jobPhase,
  normalizeJobPhase,
  type JobPhase,
} from "@/lib/job-status";

const REFRESH_MS = 60_000;

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

  const { data, isLoading, error } = useQuery({
    queryKey: ["jobs", historyQuery],
    queryFn: () =>
      api<{ jobs: Job[] }>(`/api/jobs?${historyQuery}`).then((d) => d.jobs),
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
      .filter((j) => isActiveJobState(j.state))
      .sort((a, b) => compareActiveDesc(a, b));
    const finished = all
      .filter((j) => !isActiveJobState(j.state))
      .sort((a, b) => compareEndedDesc(a, b));
    return { active, finished };
  }, [data]);
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
            <CardDescription>
              {filteredFinished.length} of {finished.length} finished {finished.length === 1 ? "job" : "jobs"} from{" "}
              {formatDateRange(appliedHistoryRange)}.
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-end justify-end gap-3">
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
          </div>
        </CardHeader>
        <CardContent>
          {isLoading && <LoadingState label="Loading recent jobs..." rows={4} />}
          {error && <ErrorState message={(error as Error).message} />}
          {!isLoading && !error && finished.length === 0 && (
            <EmptyState message="Nothing in this window." />
          )}
          {!isLoading && !error && finished.length > 0 && filteredFinished.length === 0 && (
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
  return state.trim().split(/\s+/, 1)[0].toUpperCase() || "UNKNOWN";
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
            <Th>Name</Th>
            {showActions && <Th>Actions</Th>}
            <Th>Cluster</Th>
            <Th>Partition</Th>
            <Th>Node</Th>
            {showProgress && <Th>Server remaining</Th>}
            <Th>Started</Th>
            <Th>Ended</Th>
            <Th>Elapsed</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((j) => {
            const phase = normalizeJobPhase(j.phase) ?? jobPhase(j.job_name);
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
                  <Td className="min-w-[300px]">
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
                {showProgress && (
                  <Td className="font-mono text-xs">
                    {j.time_left || <span className="text-slate-400">—</span>}
                  </Td>
                )}
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
  let etaLabel: string | null = null;
  let etaTitle = "";
  const d = details.data;
  if (p?.current_step && p.max_steps && p.current_step < p.max_steps && d) {
    const elapsedSec = parseSlurmDuration(d.elapsed);
    if (elapsedSec > 0) {
      const etaSec = (elapsedSec * (p.max_steps - p.current_step)) / p.current_step;
      etaLabel = formatDuration(etaSec);
      const unit = d.phase === "eval" ? "episode" : "step";
      etaTitle = `Estimated from aggregate ${unit} throughput`;
    }
  }

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 text-xs">
        <span className="min-w-[8rem] flex-1 break-words font-mono leading-snug" title={label}>
          {label}
        </span>
        <span className="shrink-0 whitespace-nowrap font-mono text-slate-500">
          {effectivePercent.toFixed(1)}%
          {etaLabel && (
            <ImmediateTooltip content={etaTitle}>
              <span> · ~{etaLabel}</span>
            </ImmediateTooltip>
          )}
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

function PhaseBadge({ phase }: { phase: JobPhase }) {
  if (isTrainJobPhase(phase)) return <Badge variant="default">{phase}</Badge>;
  if (phase === "eval") return <Badge variant="outline">eval</Badge>;
  return <Badge variant="secondary">other</Badge>;
}

function canResumeJob(job: Job): boolean {
  return job.cluster !== "mlxp" && isTimeoutJobState(job.state);
}

function canCopyCheckpoint(job: Job, phase: JobPhase): boolean {
  return isTrainJobPhase(phase) && isCompletedJobState(job.state);
}

function compareEndedDesc(a: Job, b: Job): number {
  const aEnd = parseJobTimestampMs(a.end, a.cluster);
  const bEnd = parseJobTimestampMs(b.end, b.cluster);
  if (aEnd !== bEnd) return bEnd - aEnd;

  const aStart = parseJobTimestampMs(a.start, a.cluster);
  const bStart = parseJobTimestampMs(b.start, b.cluster);
  if (aStart !== bStart) return bStart - aStart;
  return compareJobIdDesc(a.job_id, b.job_id);
}

function compareActiveDesc(a: Job, b: Job): number {
  const aStart = parseJobTimestampMs(a.start, a.cluster);
  const bStart = parseJobTimestampMs(b.start, b.cluster);
  if (aStart !== bStart) return bStart - aStart;

  const aEnd = parseJobTimestampMs(a.end, a.cluster);
  const bEnd = parseJobTimestampMs(b.end, b.cluster);
  if (aEnd !== bEnd) return bEnd - aEnd;
  return compareJobIdDesc(a.job_id, b.job_id);
}

function compareJobIdDesc(a: string, b: string): number {
  return b.localeCompare(a, undefined, { numeric: true, sensitivity: "base" });
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="py-2 pr-4 font-medium whitespace-nowrap">{children}</th>;
}
function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={`py-2 pr-4 ${className ?? ""}`}>{children}</td>;
}
