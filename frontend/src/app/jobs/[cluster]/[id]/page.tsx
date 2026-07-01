"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  useIsFetching,
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";
import { toast } from "sonner";
import { ArrowLeft, ExternalLink } from "lucide-react";
import {
  api,
  logStreamUrl,
  type CheckpointCopyRecord,
  type EvalRun,
  type GpuUsage,
  type JobEvalRuns,
  type JobDetails,
  type JobGpu,
  type JobMetadata,
  type JobProgress,
  type PathExistence,
  type WandbStatus,
} from "@/lib/api";
import { hasRenderableProgress, stepEta } from "@/lib/job-progress";
import { formatPct } from "@/lib/format";
import {
  canCopyCheckpoint,
  canResumeJob,
  canRetryJob,
  isCompletedJobState,
  isRunningOrCompletingJobState,
  isTerminalJobState,
  isTrainJobPhase,
  resubmitSourceLabel,
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
import { CheckpointCopyList } from "@/components/checkpoint-copy-history";
import { ResumeJobButton } from "@/components/resume-job-button";
import { RefreshButton } from "@/components/refresh-button";
import { EmptyState, ErrorState, InlineLoading, LoadingState } from "@/components/loading-state";
import { JobStateBadge } from "@/components/job-state-badge";
import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { Th } from "@/components/table";
import { JobTimestamp } from "@/components/job-timestamp";
import { ProgressBar } from "@/components/progress-bar";
import { JobLink } from "@/components/job-link";

const REFRESH_MS = 60_000;
const LOG_PAGE_SIZE = 100;

// Per-job query-key families scoped by [family, cluster, id]. The in-flight
// indicator watches exactly these; refreshAll invalidates them plus the
// checkpoint-copies list (which is deliberately excluded from the indicator).
const JOB_DETAIL_QUERY_FAMILIES = [
  "job",
  "job-details",
  "job-progress",
  "job-flags",
  "job-metadata",
  "job-gpu",
  "job-eval-runs",
] as const;

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
    queryKey: ["job-details", cluster, id],
    queryFn: () => api<JobDetails>(`/api/jobs/${cluster}/${id}/details`),
    refetchInterval: (q) =>
      isTerminalJobState((q.state.data as JobDetails | undefined)?.state)
        ? false
        : REFRESH_MS,
  });

  const progress = useQuery({
    queryKey: ["job-progress", cluster, id],
    queryFn: () => api<JobProgress>(`/api/jobs/${cluster}/${id}/progress`),
    enabled: Boolean(details.data),
    refetchInterval: isTerminalJobState(details.data?.state) ? false : REFRESH_MS,
  });

  const metadata = useQuery({
    queryKey: ["job-metadata", cluster, id],
    queryFn: () => api<JobMetadata>(`/api/jobs/${cluster}/${id}/metadata`),
    enabled: Boolean(details.data),
    staleTime: 5 * 60_000,
  });

  const gpu = useQuery({
    queryKey: ["job-gpu", cluster, id],
    queryFn: () => api<JobGpu>(`/api/jobs/${cluster}/${id}/gpu`),
    enabled: isRunningOrCompletingJobState(details.data?.state),
    refetchInterval: REFRESH_MS,
  });

  const evalRuns = useQuery({
    queryKey: ["job-eval-runs", cluster, id],
    queryFn: () => api<JobEvalRuns>(`/api/jobs/${cluster}/${id}/eval-runs`),
    enabled: details.data?.phase === "eval",
    refetchInterval: isTerminalJobState(details.data?.state) ? false : REFRESH_MS,
  });
  const resumeOf = details.data?.resume_of ?? null;
  const sourceJob = useQuery({
    queryKey: ["job", cluster, resumeOf],
    queryFn: () => api<Record<string, string>>(`/api/jobs/${cluster}/${resumeOf}`),
    enabled: Boolean(resumeOf),
    staleTime: 60_000,
  });

  const stopped = isTerminalJobState(sacct.data?.State) || isTerminalJobState(details.data?.state);

  const refreshAll = () => {
    for (const family of JOB_DETAIL_QUERY_FAMILIES) {
      qc.invalidateQueries({ queryKey: [family, cluster, id] });
    }
    qc.invalidateQueries({ queryKey: ["checkpoint-copies", cluster, id] });
  };

  const isFetching =
    useIsFetching({
      predicate: (q) => {
        const [k0, k1, k2] = q.queryKey as unknown[];
        return (
          JOB_DETAIL_QUERY_FAMILIES.includes(
            k0 as (typeof JOB_DETAIL_QUERY_FAMILIES)[number],
          ) &&
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
  const stateForActions = sacct.data?.State ?? details.data?.state ?? "";
  const checkpointPath = details.data?.paths.ckpt_dir ?? null;
  const checkpointPathExists = useQuery({
    queryKey: ["path-exists", cluster, checkpointPath],
    queryFn: () =>
      api<PathExistence>(
        `/api/clusters/${cluster}/path-exists?path=${encodeURIComponent(checkpointPath!)}`,
      ),
    enabled: Boolean(checkpointPath),
  });
  const checkpointMissing = checkpointPathExists.data?.exists === false;
  const canResume = canResumeJob({ cluster, state: stateForActions });
  const canRetry = canRetryJob({ cluster, state: stateForActions });
  const canCopy = canCopyCheckpoint({ state: sacct.data?.State, phase });
  const detailsError = details.error as Error | null;
  const progressError = progress.error as Error | null;
  const metadataError = metadata.error as Error | null;
  const gpuError = gpu.error as Error | null;
  const evalRunsError = evalRuns.error as Error | null;
  const [stream, setStream] = useState<"out" | "err" | "isaac">("out");
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [copyOpen, setCopyOpen] = useState(false);

  const cancelLabel = cluster === "mlxp" ? "kubectl delete job" : "scancel";

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
                {" "}({resubmitSourceLabel(details.data?.resubmit_action, sourceJob.data?.State)}{" "}
                <JobLink
                  cluster={cluster}
                  jobId={resumeOf}
                  className="font-mono dark:text-blue-400"
                >
                  {resumeOf}
                </JobLink>
                )
              </span>
            )}{" "}
            <span className="text-slate-400">·</span>{" "}
            <span className="text-slate-500">{cluster}</span>
          </h1>
          {details.isLoading && (
            <InlineLoading label="Resolving job details..." className="mt-1" />
          )}
          {details.error && (
            <div className="mt-1 text-sm text-red-600 dark:text-red-400">
              {detailsError?.message}
            </div>
          )}
          {details.data && (
            <div className="mt-1 space-y-1">
              <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                <Badge variant={phase === "eval" ? "outline" : "default"}>{phase ?? "unknown"}</Badge>
                {details.data.variant && (
                  <span className="font-mono text-xs">{details.data.variant}</span>
                )}
              </div>
              {details.data.train_note && (
                <div className="max-w-3xl text-sm text-slate-600 dark:text-slate-400">
                  {details.data.train_note}
                </div>
              )}
              {isEval && details.data.training_job && (
                <div className="flex items-center gap-1.5 text-sm text-slate-600 dark:text-slate-400">
                  <span>Training job:</span>
                  <JobLink
                    cluster={details.data.training_job.cluster}
                    jobId={details.data.training_job.job_id}
                    className="inline-flex items-center gap-1 font-mono dark:text-blue-400"
                    title={details.data.training_job.job_name ?? undefined}
                  >
                    {details.data.training_job.cluster}/{details.data.training_job.job_id}
                    <ExternalLink className="h-3 w-3" />
                  </JobLink>
                  {details.data.training_job.job_name && (
                    <span className="truncate font-mono text-xs text-slate-500">
                      {details.data.training_job.job_name}
                    </span>
                  )}
                </div>
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
              disabled={checkpointMissing}
              title={
                checkpointMissing
                  ? "Checkpoint path is unavailable: moved/deleted or not generated yet."
                  : undefined
              }
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
          {canRetry && (
            <ResumeJobButton
              cluster={cluster}
              jobId={id}
              phase={phase}
              variant={details.data?.variant}
              jobName={details.data?.job_name}
              action="retry"
            />
          )}
          <Button
            variant="destructive"
            size="sm"
            onClick={() => setConfirmCancel(true)}
            disabled={cancel.isPending}
          >
            {cancel.isPending ? "Cancelling..." : cancelLabel}
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
          progress={progress.data}
          progressLoading={progress.isLoading}
          progressError={progressError}
          gpu={gpu.data?.gpu ?? null}
          gpuLoading={gpu.isLoading}
          gpuError={gpuError}
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
                {["State", "ExitCode", "Elapsed", "Start", "End", "Partition", "NodeList", "GPUs", "Reason"].map((k) =>
                  sacct.data[k] ? (
                    <div key={k} className="flex flex-col">
                      <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
                      <dd className="font-mono text-xs">
                        {k === "State" ? <JobStateBadge state={sacct.data[k]} /> : formatSacctValue(k, sacct.data[k])}
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
        effectiveConfigText={metadata.data?.config_snapshot?.text ?? null}
        effectiveConfigPath={metadata.data?.config_snapshot?.path ?? null}
        modelLabel={metadata.data?.config_snapshot?.git_repo_label ?? null}
        modelRepoPath={metadata.data?.config_snapshot?.git_repo_path ?? null}
        effectiveConfigLoading={metadata.isLoading}
        effectiveConfigError={metadataError}
        loading={details.isLoading}
        error={detailsError}
        className="mt-6"
      />

      <DataInterfaceCard
        variantName={details.data?.variant ?? null}
        summaryOverride={metadata.data?.data_interface ?? null}
        loading={details.isLoading || metadata.isLoading}
        error={detailsError ?? metadataError}
        className="mt-6"
      />

      <SubmissionSnapshotCard
        d={details.data}
        snapshot={metadata.data?.config_snapshot ?? null}
        wandbProject={metadata.data?.wandb_project ?? details.data?.wandb_project ?? null}
        isLoading={details.isLoading || metadata.isLoading}
        error={detailsError ?? metadataError}
      />

      <PathsCard
        d={details.data}
        cluster={cluster}
        checkpointPathExists={checkpointPathExists.data?.exists ?? null}
        isLoading={details.isLoading}
        error={detailsError}
      />

      {isEval && (
        <EvalRunsCard
          d={details.data}
          rows={evalRuns.data?.eval_runs ?? []}
          isLoading={details.isLoading || evalRuns.isLoading}
          error={detailsError ?? evalRunsError}
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
  snapshot = null,
  wandbProject = null,
  isLoading = false,
  error,
}: {
  d?: JobDetails;
  snapshot?: JobMetadata["config_snapshot"];
  wandbProject?: string | null;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const isTrain = isTrainJobPhase(d?.phase);
  const shownWandbProject = snapshot?.wandb_project ?? wandbProject;
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
        {!isLoading && !error && d && isTrain && !snapshot && !shownWandbProject && (
          <EmptyState message="No submission snapshot was recorded for this job." />
        )}
        {!isLoading && !error && d && isTrain && (snapshot || shownWandbProject) && (
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
              {shownWandbProject && (
                <SnapshotRow label="wandb project" value={shownWandbProject} />
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

function formatSacctValue(key: string, value: string) {
  if (key !== "Start" && key !== "End") return value;
  return <JobTimestamp iso={value} />;
}

function GpuUsageSection({
  gpu,
  isLoading = false,
  error,
}: {
  gpu?: GpuUsage | null;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const hasUsage = !!gpu?.devices.length;

  return (
    <div className="mt-5 border-t border-slate-100 pt-4 dark:border-slate-900">
      {isLoading && <LoadingState label="Loading GPU sample..." rows={2} />}
      {!isLoading && error && <ErrorState message={error.message} />}
      {!isLoading && !error && !gpu && (
        <p className="text-sm text-slate-500">GPU sample unavailable.</p>
      )}
      {!isLoading && !error && gpu && !hasUsage && (
        <div className="space-y-1 text-sm text-slate-500">
          <p>{gpu.error ?? "GPU utilization is unavailable."}</p>
          {gpu.node && <p className="font-mono text-xs">node: {gpu.node}</p>}
        </div>
      )}
      {!isLoading && !error && gpu && hasUsage && (
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
                <ProgressBar percent={memoryPercent} height="h-2" />
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
  progress,
  progressLoading = false,
  progressError,
  gpu,
  gpuLoading = false,
  gpuError,
  isLoading = false,
  error,
}: {
  d?: JobDetails;
  progress?: JobProgress;
  progressLoading?: boolean;
  progressError?: Error | null;
  gpu?: GpuUsage | null;
  gpuLoading?: boolean;
  gpuError?: Error | null;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const p = progress?.progress ?? d?.progress;
  const phase = progress?.phase ?? d?.phase;
  const state = progress?.state ?? d?.state;
  const elapsed = progress?.elapsed ?? d?.elapsed;
  const wandbUrl = progress?.wandb_url ?? d?.wandb_url;
  const isComplete = isCompletedJobState(state);
  // Treat a completed job as 100% even when the backend signal didn't make
  // it (e.g. wandb run already archived, checkpoint path moved off DDN).
  const effectivePercent = isComplete ? 100 : Math.max(0, Math.min(100, p?.percent ?? 0));
  const showBar = hasRenderableProgress(p, { isComplete });
  // Surface wandb-not-configured as the actionable cause of empty progress,
  // rather than the generic "No progress yet" which looks like a backend bug.
  const wandbStatus = useQuery({
    queryKey: ["wandb-status"],
    queryFn: () => api<WandbStatus>("/api/wandb/status"),
    enabled: !!d && isTrainJobPhase(phase) && !isComplete,
    staleTime: 60_000,
  });
  const isTrain = isTrainJobPhase(phase);
  const wandbMissing =
    !isComplete && isTrain && wandbStatus.data && !wandbStatus.data.logged_in;
  const showGpu = isRunningOrCompletingJobState(state);

  // ETA from elapsed × steps-remaining / current-step. Same linear model the
  // /jobs table uses.
  const eta = isComplete
    ? null
    : stepEta(elapsed, p?.current_step, p?.max_steps, phase);

  // Training jobs deserve a tidy "step N/N" at completion even if the live
  // current_label decayed; eval/other phases already carry a meaningful
  // current_label (e.g. "15/15 runs · episode 70/70") so respect it.
  const isTrainPhase = isTrainJobPhase(phase);
  const label = isComplete && isTrainPhase
    ? p?.max_steps
      ? `step ${p.max_steps.toLocaleString()}/${p.max_steps.toLocaleString()}`
      : "Complete"
    : (p?.current_label ?? (isComplete ? "Complete" : null));

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Progress</CardTitle>
        {wandbUrl && (
          <a
            href={wandbUrl}
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
            {progressLoading && !progress && (
              <LoadingState label="Loading progress..." rows={2} />
            )}
            {progressError && <ErrorState message={progressError.message} />}
            {!showBar && wandbMissing && (
              <p className="text-sm text-slate-600 dark:text-slate-400">
                <Link href="/settings" className="text-blue-600 hover:underline">
                  Sign in
                </Link>{" "}
                to see live progress
              </p>
            )}
            {!progressLoading && !progressError && !showBar && !wandbMissing && (
              <p className="text-sm text-slate-500">{p?.current_label ?? "No progress yet."}</p>
            )}
            {!progressError && showBar && (
              <>
                <div className="flex items-baseline justify-between text-sm">
                  <span className="font-mono">{label}</span>
                  <span className="text-slate-500">{effectivePercent.toFixed(1)}%</span>
                </div>
                <ProgressBar percent={effectivePercent} height="h-2" className="mt-2" />
                {eta && (
                  <ImmediateTooltip content={eta.etaTitle}>
                    <div className="mt-2 text-xs text-slate-500">
                      ~{eta.etaLabel} left
                    </div>
                  </ImmediateTooltip>
                )}
              </>
            )}
            {showGpu && (
              <GpuUsageSection
                gpu={gpu}
                isLoading={gpuLoading}
                error={gpuError}
              />
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function PathsCard({
  d,
  cluster,
  checkpointPathExists,
  isLoading = false,
  error,
}: {
  d?: JobDetails;
  cluster: string;
  checkpointPathExists?: boolean | null;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const copyHistory = useQuery({
    queryKey: ["checkpoint-copies", cluster, d?.job_id],
    queryFn: () =>
      api<CheckpointCopyRecord[]>(
        `/api/jobs/${cluster}/${d!.job_id}/checkpoint-copies`,
      ),
    enabled: Boolean(d?.job_id),
  });
  const checkpointUnavailable = checkpointPathExists === false;
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
              <div key={r.label}>
                <div className="flex items-center justify-between gap-4 py-2">
                  <div className="min-w-[110px] text-xs uppercase tracking-wide text-slate-500">{r.label}</div>
                  <div
                    className={`flex-1 truncate font-mono text-xs ${
                      r.label === "checkpoints" && checkpointUnavailable
                        ? "line-through"
                        : ""
                    }`}
                    title={r.value}
                  >
                    {r.value}
                  </div>
                  {r.label === "checkpoints" && checkpointUnavailable && (
                    <span className="shrink-0 text-xs text-slate-500">
                      unavailable: moved/deleted or not generated
                    </span>
                  )}
                  <CopyButton value={r.value} />
                </div>
                {r.label === "checkpoints" && (
                  <CheckpointCopyHistoryRows history={copyHistory} />
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function CheckpointCopyHistoryRows({
  history,
}: {
  history: UseQueryResult<CheckpointCopyRecord[], Error>;
}) {
  if (history.isLoading) {
    return <InlineLoading label="Loading copied checkpoints..." className="pb-2 pl-[110px]" />;
  }
  if (history.error) {
    return (
      <div className="pb-2 pl-[110px] text-xs text-red-600">
        {(history.error as Error).message}
      </div>
    );
  }
  if (!history.data?.length) return null;

  return (
    <CheckpointCopyList
      records={history.data}
      className="space-y-1 pb-2 pl-[110px]"
      itemClassName="min-w-0 py-1 text-xs"
      showTime={false}
    />
  );
}

function EvalRunsCard({
  d,
  rows,
  isLoading = false,
  error,
}: {
  d?: JobDetails;
  rows: EvalRun[];
  isLoading?: boolean;
  error?: Error | null;
}) {
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
          <EmptyState message="No per-run results found yet." />
        )}
        {!isLoading && !error && rows.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-[900px] w-full table-fixed text-sm">
              <colgroup>
                {hasTask && <col className="w-[190px]" />}
                <col className="w-[56px]" />
                <col className="w-[72px]" />
                <col className="w-[70px]" />
                <col className="w-[160px]" />
                <col />
              </colgroup>
              <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
                <tr>
                  {hasTask && <Th>Task</Th>}
                  <Th>Set</Th>
                  <Th>Run</Th>
                  <Th>Seed</Th>
                  <Th>Success</Th>
                  <Th>Result</Th>
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
        <td className="truncate py-2 pr-4 font-mono text-xs">
          {row.task ?? <span className="text-slate-400">—</span>}
        </td>
      )}
      <td className="whitespace-nowrap py-2 pr-4 font-mono text-xs">{row.eval_set}</td>
      <td className="whitespace-nowrap py-2 pr-4 font-mono text-xs">{row.run}</td>
      <td className="whitespace-nowrap py-2 pr-4 font-mono text-xs">
        {row.seed ?? <span className="text-slate-400">—</span>}
      </td>
      <td className="whitespace-nowrap py-2 pr-4 font-mono text-xs">
        {formatEvalRunSuccess(row) ?? <span className="text-slate-400">—</span>}
      </td>
      <td className="w-full py-2">
        <div className="flex min-w-0 items-center justify-between gap-4">
          <ImmediateTooltip content={row.path} className="min-w-0 flex-1">
            <span className="block w-full truncate font-mono text-xs">{row.path}</span>
          </ImmediateTooltip>
          <span className="shrink-0">
            <CopyButton value={row.path} title="Copy result file" />
          </span>
        </div>
      </td>
    </tr>
  );
}

function formatEvalRunSuccess(row: EvalRun): string | null {
  const rate = row.success_rate == null ? null : formatPct(row.success_rate);
  if (row.success_count != null && row.total_episodes != null) {
    return `${row.success_count}/${row.total_episodes}${rate ? ` (${rate})` : ""}`;
  }
  return rate;
}

function LogStream({
  cluster,
  jobId,
  stream,
}: {
  cluster: string;
  jobId: string;
  stream: "out" | "err" | "isaac";
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
  }, [cluster, jobId, stream, flushReceivedLines]);

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

  const logText =
    receivedCount === 0
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
