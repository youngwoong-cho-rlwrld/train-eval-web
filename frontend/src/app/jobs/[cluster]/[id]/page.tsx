"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  useIsFetching,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { toast } from "sonner";
import { ArrowLeft, ExternalLink } from "lucide-react";
import { api, logStreamUrl, type EvalRun, type JobDetails } from "@/lib/api";
import { formatDuration, parseSlurmDuration } from "@/lib/duration";
import { formatJobTimestamp } from "@/lib/job-time";
import {
  isCompletedJobState,
  isTerminalJobState,
  isTimeoutJobState,
  isTrainJobPhase,
} from "@/lib/job-status";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ConfigCard, DataInterfaceCard } from "@/components/config-card";
import { CopyButton } from "@/components/copy-button";
import { CopyCheckpointDialog } from "@/components/copy-checkpoint-dialog";
import { ResumeJobButton } from "@/components/resume-job-button";
import { RefreshButton } from "@/components/refresh-button";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";
import { JobStateBadge } from "@/components/job-state-badge";
import { ImmediateTooltip } from "@/components/immediate-tooltip";

const REFRESH_MS = 60_000;
const LOG_PAGE_SIZE = 100;

export default function JobDetail({ params }: { params: Promise<{ cluster: string; id: string }> }) {
  const { cluster, id } = use(params);
  const qc = useQueryClient();

  const sacct = useQuery({
    queryKey: ["job", cluster, id],
    queryFn: () => api<Record<string, string>>(`/api/jobs/${cluster}/${id}`),
    refetchInterval: (q) =>
      isTerminalJobState((q.state.data as Record<string, string> | undefined)?.State)
        ? false
        : REFRESH_MS,
  });

  const details = useQuery({
    queryKey: ["job-details", cluster, id, "gpu"],
    queryFn: () => api<JobDetails>(`/api/jobs/${cluster}/${id}/details?include_gpu=true`),
    refetchInterval: (q) =>
      isTerminalJobState((q.state.data as JobDetails | undefined)?.state)
        ? false
        : REFRESH_MS,
  });

  const stopped = isTerminalJobState(sacct.data?.State) || isTerminalJobState(details.data?.state);

  const refreshAll = () => {
    qc.invalidateQueries({ queryKey: ["job", cluster, id] });
    qc.invalidateQueries({ queryKey: ["job-details", cluster, id] });
    qc.invalidateQueries({ queryKey: ["job-flags", cluster, id] });
    qc.invalidateQueries({ queryKey: ["variant"] });
  };

  const isFetching =
    useIsFetching({
      predicate: (q) => {
        const [k0, k1, k2] = q.queryKey as unknown[];
        return (
          (k0 === "job" || k0 === "job-details" || k0 === "job-flags") &&
          k1 === cluster &&
          k2 === id
        );
      },
    }) > 0;

  const cancel = useMutation({
    mutationFn: () => api(`/api/jobs/${cluster}/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      toast.success(`cancel sent to ${id}`);
      qc.invalidateQueries({ queryKey: ["job", cluster, id] });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const phase = details.data?.phase;
  const isEval = phase === "eval";
  const isTrainPhase = isTrainJobPhase(phase);
  const isComplete = isCompletedJobState(sacct.data?.State);
  const stateForActions = sacct.data?.State ?? details.data?.state ?? "";
  const canResume = cluster !== "mlxp" && isTimeoutJobState(stateForActions);
  const canCopy = isTrainPhase && isComplete;
  const detailsError = details.error as Error | null;
  const resumeOf = details.data?.resume_of ?? null;
  const [stream, setStream] = useState<"out" | "err" | "isaac">("out");
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [copyOpen, setCopyOpen] = useState(false);

  const cancelLabel = cluster === "mlxp" ? "kubectl delete job" : "scancel";

  useEffect(() => {
    document.title = details.data?.job_name ?? `${cluster}/${id}`;
    return () => {
      document.title = "train-eval-web";
    };
  }, [cluster, details.data?.job_name, id]);

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <Link
        href="/jobs"
        className="inline-flex items-center gap-1 text-sm text-slate-500 transition-colors hover:text-slate-900 dark:hover:text-slate-50"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to jobs
      </Link>
      <div className="mt-4 flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Job <span className="font-mono">{id}</span>
            {resumeOf && (
              <span className="text-base font-normal text-slate-500">
                {" "}(resumed from{" "}
                <Link
                  href={`/jobs/${encodeURIComponent(cluster)}/${encodeURIComponent(resumeOf)}`}
                  target="_blank"
                  rel="noreferrer"
                  className="font-mono text-blue-600 hover:underline dark:text-blue-400"
                >
                  {resumeOf}
                </Link>
                )
              </span>
            )}{" "}
            <span className="text-slate-400">·</span>{" "}
            <span className="text-slate-500">{cluster}</span>
          </h1>
          {details.isLoading && (
            <div className="mt-1 flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
              <Badge variant="secondary">loading</Badge>
              <span className="font-mono text-xs">resolving job details...</span>
            </div>
          )}
          {details.error && (
            <div className="mt-1 text-sm text-red-600 dark:text-red-400">
              {detailsError?.message}
            </div>
          )}
          {details.data && (
            <div className="mt-1 flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
              <Badge variant={phase === "eval" ? "outline" : "default"}>{phase ?? "unknown"}</Badge>
              {details.data.variant && (
                <span className="font-mono text-xs">{details.data.variant}</span>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!stopped && (
            <RefreshButton
              isFetching={isFetching}
              onRefresh={refreshAll}
              intervalMs={REFRESH_MS}
            />
          )}
          {canCopy && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCopyOpen(true)}
            >
              Copy checkpoint
            </Button>
          )}
          {canResume && (
            <ResumeJobButton
              cluster={cluster}
              jobId={id}
              phase={phase}
              variant={details.data?.variant}
              jobName={details.data?.job_name}
            />
          )}
          <Button
            variant="destructive"
            size="sm"
            onClick={() => setConfirmCancel(true)}
            disabled={cancel.isPending}
          >
            {cancel.isPending ? "Cancelling…" : cancelLabel}
          </Button>
        </div>
      </div>

      <CopyCheckpointDialog
        open={copyOpen}
        onOpenChange={setCopyOpen}
        cluster={cluster}
        jobId={id}
      />

      <Dialog open={confirmCancel} onOpenChange={setConfirmCancel}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Cancel this job?</DialogTitle>
            <DialogDescription>
              This will run <code className="font-mono">{cancelLabel}</code> on
              job <span className="font-mono">{id}</span>{" "}
              {details.data?.variant && (
                <>
                  (<span className="font-mono">{details.data.variant}</span>){" "}
                </>
              )}
              on <span className="font-mono">{cluster}</span>. In-progress steps
              will be lost; checkpoints already on disk are preserved.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmCancel(false)}>
              Keep running
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                setConfirmCancel(false);
                cancel.mutate();
              }}
            >
              Yes, {cancelLabel}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <ProgressCard
          d={details.data}
          isLoading={details.isLoading}
          error={detailsError}
        />
        <Card>
          <CardHeader>
            <CardTitle>sacct</CardTitle>
          </CardHeader>
          <CardContent>
            {sacct.isLoading && <LoadingState label="Loading sacct..." rows={4} />}
            {sacct.error && <ErrorState message={(sacct.error as Error).message} />}
            {sacct.data && (
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                {["State", "ExitCode", "Elapsed", "Start", "End", "Partition", "NodeList", "Reason"].map((k) =>
                  sacct.data[k] ? (
                    <div key={k} className="flex flex-col">
                      <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
                      <dd className="font-mono text-xs">
                        {k === "State" ? <JobStateBadge state={sacct.data[k]} /> : formatSacctValue(k, sacct.data[k], cluster)}
                      </dd>
                    </div>
                  ) : null,
                )}
              </dl>
            )}
          </CardContent>
        </Card>
      </div>

      <ConfigCard
        variantName={details.data?.variant ?? null}
        flagsUrl={`/api/jobs/${cluster}/${id}/flags`}
        queryKey={["job-flags", cluster, id]}
        cluster={cluster}
        phase={details.data?.phase}
        checkpointOverride={
          details.data?.phase === "eval"
            ? details.data.paths.eval_checkpoint
            : null
        }
        effectiveConfigText={details.data?.config_snapshot?.text ?? null}
        effectiveConfigPath={details.data?.config_snapshot?.path ?? null}
        modelLabel={details.data?.config_snapshot?.git_repo_label ?? null}
        modelRepoPath={details.data?.config_snapshot?.git_repo_path ?? null}
        loading={details.isLoading}
        error={detailsError}
        className="mt-6"
      />

      <DataInterfaceCard
        variantName={details.data?.variant ?? null}
        loading={details.isLoading}
        error={detailsError}
        className="mt-6"
      />

      <SubmissionSnapshotCard
        d={details.data}
        isLoading={details.isLoading}
        error={detailsError}
      />

      <PathsCard
        d={details.data}
        cluster={cluster}
        isLoading={details.isLoading}
        error={detailsError}
      />

      {isEval && (
        <EvalRunsCard
          d={details.data}
          isLoading={details.isLoading}
          error={detailsError}
        />
      )}

      <Card className="mt-6">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Logs</CardTitle>
          <div className="flex gap-2">
            <Button variant={stream === "out" ? "default" : "outline"} size="sm" onClick={() => setStream("out")}>stdout</Button>
            <Button variant={stream === "err" ? "default" : "outline"} size="sm" onClick={() => setStream("err")}>stderr</Button>
            {isEval && (
              <Button variant={stream === "isaac" ? "default" : "outline"} size="sm" onClick={() => setStream("isaac")}>
                Isaac Sim
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <LogStream
            key={`${cluster}:${id}:${stream}`}
            cluster={cluster}
            jobId={id}
            stream={stream}
          />
        </CardContent>
      </Card>
    </div>
  );
}

function SubmissionSnapshotCard({
  d,
  isLoading = false,
  error,
}: {
  d?: JobDetails;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const snapshot = d?.config_snapshot;
  const isTrain = isTrainJobPhase(d?.phase);
  const wandbProject = snapshot?.wandb_project ?? d?.wandb_project ?? null;
  const extraArgs = snapshot?.extra_args?.length
    ? snapshot.extra_args.join(" ")
    : snapshotExtraArgs(snapshot?.text);

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle>Submission Snapshot</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading && <LoadingState label="Loading submission snapshot..." />}
        {!isLoading && error && <ErrorState message={error.message} />}
        {!isLoading && !error && !d && (
          <EmptyState message="Submission snapshot unavailable." />
        )}
        {!isLoading && !error && d && !isTrain && (
          <EmptyState message="Submission snapshots are recorded for training jobs." />
        )}
        {!isLoading && !error && d && isTrain && !snapshot && !wandbProject && (
          <EmptyState message="No submission snapshot was recorded for this job." />
        )}
        {!isLoading && !error && d && isTrain && (snapshot || wandbProject) && (
          <>
            <div className="divide-y divide-slate-100 dark:divide-slate-900">
              {snapshot?.path && (
                <SnapshotRow label="config" value={snapshot.path} />
              )}
              {snapshot?.meta_path && (
                <SnapshotRow label="metadata" value={snapshot.meta_path} />
              )}
              {snapshot?.extra_args_path && (
                <SnapshotRow label="extra args file" value={snapshot.extra_args_path} />
              )}
              {wandbProject && (
                <SnapshotRow label="wandb project" value={wandbProject} />
              )}
              {snapshot?.git_repo_label && (
                <SnapshotRow label="code repo" value={snapshot.git_repo_label} />
              )}
              {snapshot?.git_repo_path && (
                <SnapshotRow label="repo path" value={snapshot.git_repo_path} />
              )}
              {snapshot?.git_branch && (
                <SnapshotRow label="training branch" value={snapshot.git_branch} />
              )}
              {snapshot?.git_commit && (
                <SnapshotRow label="training commit" value={snapshot.git_commit} />
              )}
              {extraArgs && (
                <SnapshotRow label="extra args" value={extraArgs} />
              )}
              {snapshot && (
                <SnapshotRow
                  label="dirty"
                  value={
                    snapshot.git_dirty_at_submit == null
                      ? "unknown"
                      : snapshot.git_dirty_at_submit
                        ? snapshot.git_committed_dirty
                          ? "dirty changes committed before submit"
                          : "dirty at submit"
                        : "clean"
                  }
                />
              )}
            </div>
            {snapshot?.error && <ErrorState message={snapshot.error} />}
            {snapshot?.text && (
              <pre className="max-h-80 overflow-auto rounded border border-slate-200 bg-slate-50 p-3 text-xs dark:border-slate-800 dark:bg-slate-950">
                {snapshot.text}
              </pre>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function snapshotExtraArgs(text?: string | null) {
  if (!text) return null;
  const match = text.match(/^SUBMIT_EXTRA_ARGS=\(\n([\s\S]*?)^\)$/m);
  if (!match) return null;

  const args = match[1]
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  return args.length ? args.join(" ") : null;
}

function SnapshotRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <div className="min-w-[110px] text-xs uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className="flex-1 truncate font-mono text-xs">{value}</div>
      <CopyButton value={value} />
    </div>
  );
}

function formatSacctValue(key: string, value: string, cluster: string) {
  if (key !== "Start" && key !== "End") return value;
  const formatted = formatJobTimestamp(value, cluster);
  if (!formatted) return <span className="text-slate-400">—</span>;
  return (
    <ImmediateTooltip content={formatted.full}>
      <span>{formatted.short}</span>
    </ImmediateTooltip>
  );
}

function GpuUsageSection({ d }: { d: JobDetails }) {
  const gpu = d.gpu;
  const hasUsage = !!gpu?.devices.length;

  return (
    <div className="mt-5 border-t border-slate-100 pt-4 dark:border-slate-900">
      {!gpu && (
        <p className="text-sm text-slate-500">GPU sample unavailable.</p>
      )}
      {gpu && !hasUsage && (
        <div className="space-y-1 text-sm text-slate-500">
          <p>{gpu.error ?? "GPU utilization is unavailable."}</p>
          {gpu.node && <p className="font-mono text-xs">node: {gpu.node}</p>}
        </div>
      )}
      {gpu && hasUsage && (
        <div className="space-y-3 text-xs text-slate-500">
          {gpu.node && <div className="font-mono">node: {gpu.node}</div>}
          {gpu.devices.map((dev) => {
            const util = dev.utilization_gpu_percent;
            const memoryPercent =
              dev.total_mib > 0
                ? Math.max(0, Math.min(100, (dev.used_mib / dev.total_mib) * 100))
                : 0;
            const label = dev.name ? `gpu ${dev.index} · ${dev.name}` : `gpu ${dev.index}`;
            return (
              <div key={dev.index} className="space-y-1.5">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="min-w-0 truncate font-mono">{label}</span>
                  <span className="shrink-0 font-mono">
                    {dev.used_gb.toFixed(1)} / {dev.total_gb.toFixed(1)} GB
                    {util != null && ` · compute ${util}%`}
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                  <div
                    className="h-full rounded-full bg-slate-900 transition-all dark:bg-slate-50"
                    style={{ width: `${memoryPercent}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ProgressCard({
  d,
  isLoading = false,
  error,
}: {
  d?: JobDetails;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const p = d?.progress;
  const isComplete = isCompletedJobState(d?.state);
  // Treat a completed job as 100% even when the backend signal didn't make
  // it (e.g. wandb run already archived, checkpoint path moved off DDN).
  const effectivePercent = isComplete ? 100 : (p?.percent ?? 0);
  const showBar = Boolean(
    isComplete ||
    (p?.current_step !== null && p?.max_steps) ||
    (p?.completed_runs !== null && p?.total_runs),
  );
  // Surface wandb-not-configured as the actionable cause of empty progress,
  // rather than the generic "No progress yet" which looks like a backend bug.
  const wandbStatus = useQuery({
    queryKey: ["wandb-status"],
    queryFn: () => api<{ logged_in: boolean; entity: string | null; project: string; error: string | null }>("/api/wandb/status"),
    enabled: !!d && isTrainJobPhase(d.phase) && !isComplete,
    staleTime: 60_000,
  });
  const isTrain = isTrainJobPhase(d?.phase);
  const wandbMissing =
    !isComplete && isTrain && wandbStatus.data && !wandbStatus.data.logged_in;
  const showGpu = /^(RUNNING|COMPLETING)/i.test(d?.state ?? "");

  // ETA from elapsed × steps-remaining / current-step. Same linear model the
  // /jobs table uses.
  let etaLabel: string | null = null;
  let etaTitle = "";
  if (!isComplete && p?.current_step && p.max_steps && p.current_step < p.max_steps && d) {
    const elapsedSec = parseSlurmDuration(d.elapsed);
    if (elapsedSec > 0) {
      const etaSec = (elapsedSec * (p.max_steps - p.current_step)) / p.current_step;
      etaLabel = formatDuration(etaSec);
      const unit = d.phase === "eval" ? "episode" : "step";
      etaTitle = `Estimated from aggregate ${unit} throughput`;
    }
  }

  // Training jobs deserve a tidy "step N/N" at completion even if the live
  // current_label decayed; eval/other phases already carry a meaningful
  // current_label (e.g. "15/15 runs · episode 70/70") so respect it.
  const isTrainPhase = isTrainJobPhase(d?.phase);
  const label = isComplete && isTrainPhase
    ? p?.max_steps
      ? `step ${p.max_steps.toLocaleString()}/${p.max_steps.toLocaleString()}`
      : "Complete"
    : (p?.current_label ?? (isComplete ? "Complete" : null));

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Progress</CardTitle>
        {d?.wandb_url && (
          <a
            href={d.wandb_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-sm text-blue-600 hover:underline"
          >
            wandb <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
      </CardHeader>
      <CardContent>
        {isLoading && <LoadingState label="Loading progress..." />}
        {!isLoading && error && <ErrorState message={error.message} />}
        {!isLoading && !error && !d && (
          <EmptyState message="Progress unavailable." />
        )}
        {!isLoading && !error && d && (
          <>
            {!showBar && wandbMissing && (
              <p className="text-sm text-slate-600 dark:text-slate-400">
                <Link href="/settings" className="text-blue-600 hover:underline">
                  Sign in
                </Link>{" "}
                to see live progress
              </p>
            )}
            {!showBar && !wandbMissing && (
              <p className="text-sm text-slate-500">{p?.current_label ?? "No progress yet."}</p>
            )}
            {showBar && (
              <>
                <div className="flex items-baseline justify-between text-sm">
                  <span className="font-mono">{label}</span>
                  <span className="text-slate-500">{effectivePercent.toFixed(1)}%</span>
                </div>
                <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                  <div
                    className="h-full rounded-full bg-slate-900 transition-all dark:bg-slate-50"
                    style={{ width: `${Math.max(0, Math.min(100, effectivePercent))}%` }}
                  />
                </div>
                {etaLabel && (
                  <ImmediateTooltip content={etaTitle}>
                    <div className="mt-2 text-xs text-slate-500">
                      ~{etaLabel} left
                    </div>
                  </ImmediateTooltip>
                )}
              </>
            )}
            {showGpu && <GpuUsageSection d={d} />}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function PathsCard({
  d,
  cluster,
  isLoading = false,
  error,
}: {
  d?: JobDetails;
  cluster: string;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const rows: { label: string; value: string }[] = [
    ...(d
      ? [
          { label: "stdout", value: d.paths.stdout },
          { label: "stderr", value: d.paths.stderr },
          { label: "exp dir", value: d.paths.exp_dir },
        ]
      : []),
  ];
  if (d?.paths.ckpt_dir) rows.push({ label: "checkpoints", value: d.paths.ckpt_dir });
  if (d?.paths.eval_checkpoint) rows.push({ label: "checkpoint", value: d.paths.eval_checkpoint });
  if (d?.paths.eval_dir) rows.push({ label: "eval results", value: d.paths.eval_dir });
  if (d?.paths.isaac_logs_glob) rows.push({ label: "isaac sim logs", value: d.paths.isaac_logs_glob });

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle>Paths <span className="text-xs font-normal text-slate-500">on {d?.cluster ?? cluster}</span></CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <LoadingState label="Loading paths..." />}
        {!isLoading && error && <ErrorState message={error.message} />}
        {!isLoading && !error && !d && <EmptyState message="Paths unavailable." />}
        {!isLoading && !error && d && (
          <div className="divide-y divide-slate-100 dark:divide-slate-900">
            {rows.map((r) => (
              <div key={r.label} className="flex items-center justify-between gap-4 py-2">
                <div className="min-w-[110px] text-xs uppercase tracking-wide text-slate-500">{r.label}</div>
                <div className="flex-1 truncate font-mono text-xs">{r.value}</div>
                <CopyButton value={r.value} />
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function EvalRunsCard({
  d,
  isLoading = false,
  error,
}: {
  d?: JobDetails;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const rows = d?.eval_runs ?? [];
  const hasTask = rows.some((row) => row.task);

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle>Eval Runs</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <LoadingState label="Loading eval runs..." />}
        {!isLoading && error && <ErrorState message={error.message} />}
        {!isLoading && !error && !d && <EmptyState message="Eval runs unavailable." />}
        {!isLoading && !error && d && rows.length === 0 && (
          <EmptyState message="No per-run result files found yet." />
        )}
        {!isLoading && !error && rows.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
                <tr>
                  {hasTask && <Th>Task</Th>}
                  <Th>Set</Th>
                  <Th>Run</Th>
                  <Th>Seed</Th>
                  <Th>Success</Th>
                  <Th>Result file</Th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <EvalRunRow key={row.path} row={row} showTask={hasTask} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function EvalRunRow({ row, showTask }: { row: EvalRun; showTask: boolean }) {
  return (
    <tr className="border-b border-slate-100 last:border-0 dark:border-slate-900">
      {showTask && (
        <td className="py-2 pr-4 font-mono text-xs">
          {row.task ?? <span className="text-slate-400">—</span>}
        </td>
      )}
      <td className="py-2 pr-4 font-mono text-xs">{row.eval_set}</td>
      <td className="py-2 pr-4 font-mono text-xs">{row.run}</td>
      <td className="py-2 pr-4 font-mono text-xs">
        {row.seed ?? <span className="text-slate-400">—</span>}
      </td>
      <td className="py-2 pr-4 font-mono text-xs">
        {formatEvalRunSuccess(row)}
      </td>
      <td className="py-2 pr-4">
        <div className="flex items-center gap-1">
          <ImmediateTooltip content={row.path} className="max-w-[420px]">
            <span className="truncate font-mono text-xs">{row.path}</span>
          </ImmediateTooltip>
          <CopyButton value={row.path} title="Copy result file" />
        </div>
      </td>
    </tr>
  );
}

function formatEvalRunSuccess(row: EvalRun) {
  const rate = row.success_rate == null ? null : `${(row.success_rate * 100).toFixed(2)}%`;
  if (row.success_count != null && row.total_episodes != null) {
    return `${row.success_count}/${row.total_episodes}${rate ? ` (${rate})` : ""}`;
  }
  return rate ?? <span className="text-slate-400">—</span>;
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="py-2 pr-4 font-medium whitespace-nowrap">{children}</th>;
}

function LogStream({
  cluster,
  jobId,
  stream,
  enabled = true,
}: {
  cluster: string;
  jobId: string;
  stream: "out" | "err" | "isaac";
  enabled?: boolean;
}) {
  const [visibleLines, setVisibleLines] = useState<string[]>([]);
  const [receivedCount, setReceivedCount] = useState(0);
  const [hiddenOlderCount, setHiddenOlderCount] = useState(0);
  const preRef = useRef<HTMLPreElement>(null);
  const allLinesRef = useRef<string[]>([]);
  const visibleStartRef = useRef(0);
  const visibleEndRef = useRef(0);
  const frameRef = useRef<number | null>(null);
  const prependScrollHeightRef = useRef<number | null>(null);
  // Whether the user was at the bottom *before* the next line arrived. If
  // they scrolled up to read history, we don't yank them back down.
  const stickToBottomRef = useRef(true);

  const renderWindow = useCallback((start: number, end: number) => {
    const all = allLinesRef.current;
    const boundedStart = Math.max(0, Math.min(start, all.length));
    const boundedEnd = Math.max(boundedStart, Math.min(end, all.length));
    visibleStartRef.current = boundedStart;
    visibleEndRef.current = boundedEnd;
    setReceivedCount(all.length);
    setHiddenOlderCount(boundedStart);
    setVisibleLines(all.slice(boundedStart, boundedEnd));
  }, []);

  const renderLatestWindow = useCallback(() => {
    const all = allLinesRef.current;
    const currentWindowSize = visibleEndRef.current > visibleStartRef.current
      ? visibleEndRef.current - visibleStartRef.current
      : LOG_PAGE_SIZE;
    const windowSize = Math.max(LOG_PAGE_SIZE, currentWindowSize);
    renderWindow(Math.max(0, all.length - windowSize), all.length);
  }, [renderWindow]);

  const flushReceivedLines = useCallback(() => {
    if (stickToBottomRef.current) {
      renderLatestWindow();
      return;
    }
    setReceivedCount(allLinesRef.current.length);
  }, [renderLatestWindow]);

  useEffect(() => {
    allLinesRef.current = [];
    visibleStartRef.current = 0;
    visibleEndRef.current = 0;
    stickToBottomRef.current = true;
    if (!enabled) return;

    function scheduleFlush() {
      if (frameRef.current !== null) return;
      frameRef.current = window.requestAnimationFrame(() => {
        frameRef.current = null;
        flushReceivedLines();
      });
    }

    const es = new EventSource(logStreamUrl(cluster, jobId, stream));
    es.addEventListener("line", (e: MessageEvent) => {
      // Sample scroll position before the state update — once React
      // re-renders with the new line, scrollHeight has already grown and
      // we can't tell whether the user was at the bottom.
      const el = preRef.current;
      if (el) {
        stickToBottomRef.current =
          el.scrollHeight - el.scrollTop - el.clientHeight < 8;
      }
      allLinesRef.current.push(e.data as string);
      scheduleFlush();
    });
    return () => {
      es.close();
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [cluster, jobId, stream, enabled, flushReceivedLines]);

  useEffect(() => {
    const el = preRef.current;
    if (!el) return;
    if (prependScrollHeightRef.current !== null) {
      el.scrollTop = el.scrollHeight - prependScrollHeightRef.current;
      prependScrollHeightRef.current = null;
      return;
    }
    if (stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [visibleLines]);

  function revealOlderLines() {
    const currentStart = visibleStartRef.current;
    if (currentStart <= 0) return;
    const el = preRef.current;
    prependScrollHeightRef.current = el?.scrollHeight ?? null;
    renderWindow(Math.max(0, currentStart - LOG_PAGE_SIZE), visibleEndRef.current);
  }

  function onScroll() {
    const el = preRef.current;
    if (!el) return;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 8;
    stickToBottomRef.current = isAtBottom;
    if (el.scrollTop <= 0) {
      revealOlderLines();
    } else if (isAtBottom && visibleEndRef.current < allLinesRef.current.length) {
      renderLatestWindow();
    }
  }

  const logText = !enabled
    ? "(logs unavailable - k8s pod has been garbage-collected)"
    : receivedCount === 0
      ? "(waiting for log lines...)"
      : [
          hiddenOlderCount > 0
            ? `(${hiddenOlderCount.toLocaleString()} earlier lines hidden - scroll to top to load ${Math.min(LOG_PAGE_SIZE, hiddenOlderCount)} more)`
            : null,
          ...visibleLines,
        ].filter((line) => line !== null).join("\n");

  return (
    <pre
      ref={preRef}
      onScroll={onScroll}
      aria-label={`Showing ${visibleLines.length} of ${receivedCount} log lines`}
      className="h-96 overflow-auto rounded-md bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-100"
    >
      {logText}
    </pre>
  );
}
