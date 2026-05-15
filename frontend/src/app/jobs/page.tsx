"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
// (useQuery referenced again in ProgressCell below — same import)
import { api, type Job, type JobDetails } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CopyButton } from "@/components/copy-button";

const ACTIVE_STATES = new Set(["RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"]);

export default function JobsPage() {
  const [hours, setHours] = useState<string>("24");

  const { data, isLoading, error } = useQuery({
    queryKey: ["jobs", hours],
    queryFn: () =>
      api<{ jobs: Job[] }>(`/api/jobs?hours=${hours}`).then((d) => d.jobs),
    refetchInterval: 5000,
  });

  const { active, finished } = useMemo(() => {
    const all = data ?? [];
    const active = all
      .filter((j) => ACTIVE_STATES.has(j.state))
      .sort((a, b) => Number(b.job_id) - Number(a.job_id));
    const finished = all
      .filter((j) => !ACTIVE_STATES.has(j.state))
      .sort((a, b) => Number(b.job_id) - Number(a.job_id));
    return { active, finished };
  }, [data]);

  return (
    <div className="mx-auto max-w-6xl px-8 py-12">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Jobs</h1>
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
      </div>
      <p className="mt-2 text-slate-600 dark:text-slate-400">
        Auto-refreshing every 5s. Active jobs from <code className="text-xs">squeue</code>; finished from <code className="text-xs">sacct</code>.
      </p>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Active</CardTitle>
          <CardDescription>{active.length} {active.length === 1 ? "job" : "jobs"} in queue or running.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading && <p className="text-sm text-slate-500">Loading…</p>}
          {error && <p className="text-sm text-red-600">{(error as Error).message}</p>}
          {!isLoading && active.length === 0 && <p className="text-sm text-slate-500">No active jobs.</p>}
          {active.length > 0 && <JobTable rows={active} kind="active" />}
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Recent</CardTitle>
          <CardDescription>{finished.length} finished {finished.length === 1 ? "job" : "jobs"} in the last {hours}h.</CardDescription>
        </CardHeader>
        <CardContent>
          {finished.length === 0 ? (
            <p className="text-sm text-slate-500">Nothing in this window.</p>
          ) : (
            <JobTable rows={finished} kind="recent" />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function JobTable({ rows, kind }: { rows: Job[]; kind: "active" | "recent" }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
          <tr>
            <Th>Job ID</Th>
            <Th>Phase</Th>
            <Th>Cluster</Th>
            <Th>State</Th>
            <Th>Partition</Th>
            <Th>Elapsed</Th>
            {kind === "active" && <Th>Time Left</Th>}
            {kind === "active" && <Th>Progress</Th>}
            {kind === "recent" && <Th>Started</Th>}
            {kind === "recent" && <Th>Ended</Th>}
            <Th>Name</Th>
            <Th>Node</Th>
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
                <Td>{j.cluster}</Td>
                <Td><StateBadge state={j.state} /></Td>
                <Td className="font-mono text-xs">{j.partition}</Td>
                <Td className="font-mono text-xs">{j.elapsed}</Td>
                {kind === "active" && (
                  <Td className="font-mono text-xs">
                    {j.time_left ?? <span className="text-slate-400">—</span>}
                  </Td>
                )}
                {kind === "active" && (
                  <Td><ProgressCell cluster={j.cluster} jobId={j.job_id} state={j.state} /></Td>
                )}
                {kind === "recent" && (
                  <Td className="font-mono text-xs"><Timestamp iso={j.start} /></Td>
                )}
                {kind === "recent" && (
                  <Td className="font-mono text-xs"><Timestamp iso={j.end} /></Td>
                )}
                <td className="py-2 pr-4 font-mono text-xs" title={j.job_name}>
                  <div className="flex items-center gap-1">
                    <span className="max-w-[240px] truncate">{j.job_name}</span>
                    <CopyButton value={j.job_name} title="Copy job name" />
                  </div>
                </td>
                <Td className="font-mono text-xs text-slate-500">{j.nodelist}</Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ProgressCell({ cluster, jobId, state }: { cluster: string; jobId: string; state: string }) {
  // Fetch for all active states. PENDING jobs that were preempted may have
  // prior checkpoints / results.json files — the backend surfaces those so
  // the user can see how far the previous run got while it waits in queue.
  // Running jobs refetch on a shorter interval so live progress feels current;
  // pending jobs poll slower since nothing's actively changing.
  const isRunning = state === "RUNNING" || state === "COMPLETING";
  const q = useQuery({
    queryKey: ["job-details", cluster, jobId],
    queryFn: () => api<JobDetails>(`/api/jobs/${cluster}/${jobId}/details`),
    refetchInterval: isRunning ? 15_000 : 60_000,
  });
  if (q.isLoading || !q.data) return <span className="text-xs text-slate-400">…</span>;
  const p = q.data.progress;
  if (p.percent == null || p.current_label == null) {
    return <span className="text-xs text-slate-400">—</span>;
  }
  return (
    <div className="min-w-[160px] space-y-1">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="font-mono text-slate-600 dark:text-slate-300">{p.current_label}</span>
        <span className="text-slate-500">{p.percent.toFixed(0)}%</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
        <div
          className={`h-full rounded transition-all ${isRunning ? "bg-slate-900 dark:bg-slate-50" : "bg-slate-400 dark:bg-slate-500"}`}
          style={{ width: `${Math.max(0, Math.min(100, p.percent))}%` }}
        />
      </div>
    </div>
  );
}

function Timestamp({ iso }: { iso?: string | null }) {
  if (!iso) return <span className="text-slate-400">—</span>;
  // ISO like "2026-05-14T15:55:07" → "05-14 15:55"
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  const short = m ? `${m[2]}-${m[3]} ${m[4]}:${m[5]}` : iso;
  return <span title={iso}>{short}</span>;
}

function phaseOf(jobName: string): "train" | "resume" | "eval" | "other" {
  // Slurm names:  train_<variant>_<cluster>_<partition>_<ts>
  // MLXP names:   youngwoong-train-<variant>-<ts>
  // Anchor on (start | hyphen | underscore) + phase + (hyphen | underscore).
  const m = jobName.match(/(?:^|[-_])(train|resume|eval)[-_]/);
  return (m?.[1] as "train" | "resume" | "eval") ?? "other";
}

function PhaseBadge({ phase }: { phase: ReturnType<typeof phaseOf> }) {
  if (phase === "train" || phase === "resume") return <Badge variant="default">{phase}</Badge>;
  if (phase === "eval") return <Badge variant="outline">eval</Badge>;
  return <Badge variant="secondary">other</Badge>;
}

function StateBadge({ state }: { state: string }) {
  const v =
    state === "RUNNING" ? "success"
    : state === "PENDING" ? "warning"
    : state === "FAILED" || state === "TIMEOUT" || state.startsWith("CANCEL") ? "danger"
    : state === "COMPLETED" ? "secondary"
    : "outline";
  return <Badge variant={v}>{state}</Badge>;
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="py-2 pr-4 font-medium">{children}</th>;
}
function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={`py-2 pr-4 ${className ?? ""}`}>{children}</td>;
}
