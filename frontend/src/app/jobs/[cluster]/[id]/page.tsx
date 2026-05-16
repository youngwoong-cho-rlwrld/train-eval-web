"use client";

import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  useIsFetching,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { toast } from "sonner";
import { ArrowLeft, ExternalLink } from "lucide-react";
import { api, logStreamUrl, type JobDetails, type Variant } from "@/lib/api";
import { formatDuration, parseSlurmDuration } from "@/lib/duration";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConfigCard } from "@/components/config-card";
import { CopyButton } from "@/components/copy-button";
import { RefreshButton } from "@/components/refresh-button";
import { startCopyWatcher } from "@/lib/copy-watcher";

const REFRESH_MS = 60_000;

export default function JobDetail({ params }: { params: Promise<{ cluster: string; id: string }> }) {
  const { cluster, id } = use(params);
  const qc = useQueryClient();

  // Once a job is in a terminal state (COMPLETED / FAILED / CANCELLED /
  // TIMEOUT) none of the data changes, so the refetch loop is wasted work
  // and the countdown on the RefreshButton is misleading.
  const isTerminal = (state: string | undefined) =>
    !!state &&
    /^(COMPLET|FAIL|CANCEL|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPT)/i.test(state);

  const sacct = useQuery({
    queryKey: ["job", cluster, id],
    queryFn: () => api<Record<string, string>>(`/api/jobs/${cluster}/${id}`),
    refetchInterval: (q) =>
      isTerminal((q.state.data as Record<string, string> | undefined)?.State)
        ? false
        : REFRESH_MS,
  });

  const details = useQuery({
    queryKey: ["job-details", cluster, id],
    queryFn: () => api<JobDetails>(`/api/jobs/${cluster}/${id}/details`),
    refetchInterval: (q) =>
      isTerminal((q.state.data as JobDetails | undefined)?.state)
        ? false
        : REFRESH_MS,
  });

  const variantQuery = useQuery({
    queryKey: ["variant", details.data?.variant],
    queryFn: () => api<Variant>(`/api/variants/${details.data!.variant}`),
    enabled: !!details.data?.variant,
  });

  const stopped = isTerminal(sacct.data?.State) || isTerminal(details.data?.state);

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
  const isTrainPhase = phase === "train" || phase === "resume";
  const isComplete = (sacct.data?.State ?? "").toUpperCase().startsWith("COMPLET");
  const canCopy = isTrainPhase && isComplete;
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
            Job <span className="font-mono">{id}</span>{" "}
            <span className="text-slate-400">·</span>{" "}
            <span className="text-slate-500">{cluster}</span>
          </h1>
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
        {details.data && <ProgressCard d={details.data} />}
        <Card>
          <CardHeader>
            <CardTitle>sacct</CardTitle>
          </CardHeader>
          <CardContent>
            {sacct.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
            {sacct.error && <p className="text-sm text-red-600">{(sacct.error as Error).message}</p>}
            {sacct.data && (
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                {["State", "ExitCode", "Elapsed", "Start", "End", "Partition", "NodeList", "Reason"].map((k) =>
                  sacct.data[k] ? (
                    <div key={k} className="flex flex-col">
                      <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
                      <dd className="font-mono text-xs">
                        {k === "State" ? <Badge>{sacct.data[k]}</Badge> : sacct.data[k]}
                      </dd>
                    </div>
                  ) : null,
                )}
              </dl>
            )}
          </CardContent>
        </Card>
      </div>

      {details.data && (
        <ConfigCard
          variantName={details.data.variant ?? null}
          flagsUrl={`/api/jobs/${cluster}/${id}/flags`}
          queryKey={["job-flags", cluster, id]}
          modalityConfigFile={variantQuery.data?.vars.TRAIN_MODALITY_CONFIG ?? null}
          cluster={cluster}
          phase={details.data.phase}
          checkpointOverride={
            details.data.phase === "eval"
              ? details.data.paths.eval_checkpoint
              : null
          }
          className="mt-6"
        />
      )}

      {details.data && <PathsCard d={details.data} />}

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

function ProgressCard({ d }: { d: JobDetails }) {
  const p = d.progress;
  const isComplete = (d.state ?? "").toUpperCase().startsWith("COMPLET");
  // Treat a completed job as 100% even when the backend signal didn't make
  // it (e.g. wandb run already archived, checkpoint path moved off DDN).
  const effectivePercent = isComplete ? 100 : (p.percent ?? 0);
  const showBar =
    isComplete ||
    (p.current_step !== null && p.max_steps) ||
    (p.completed_runs !== null && p.total_runs);
  // Surface wandb-not-configured as the actionable cause of empty progress,
  // rather than the generic "No progress yet" which looks like a backend bug.
  const wandbStatus = useQuery({
    queryKey: ["wandb-status"],
    queryFn: () => api<{ logged_in: boolean; entity: string | null; project: string; error: string | null }>("/api/wandb/status"),
    staleTime: 60_000,
  });
  const isTrain = d.phase === "train" || d.phase === "resume";
  const wandbMissing =
    !isComplete && isTrain && wandbStatus.data && !wandbStatus.data.logged_in;

  // ETA from elapsed × steps-remaining / current-step. Same linear model the
  // /jobs table uses.
  let etaLabel: string | null = null;
  if (!isComplete && p.current_step && p.max_steps && p.current_step < p.max_steps) {
    const elapsedSec = parseSlurmDuration(d.elapsed);
    if (elapsedSec > 0) {
      const etaSec = (elapsedSec * (p.max_steps - p.current_step)) / p.current_step;
      etaLabel = formatDuration(etaSec);
    }
  }

  // Training jobs deserve a tidy "step N/N" at completion even if the live
  // current_label decayed; eval/other phases already carry a meaningful
  // current_label (e.g. "15/15 runs · episode 70/70") so respect it.
  const isTrainPhase = d.phase === "train" || d.phase === "resume";
  const label = isComplete && isTrainPhase
    ? p.max_steps
      ? `step ${p.max_steps.toLocaleString()}/${p.max_steps.toLocaleString()}`
      : "Complete"
    : (p.current_label ?? (isComplete ? "Complete" : null));

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Progress</CardTitle>
        {d.wandb_url && (
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
        {!showBar && wandbMissing && (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            <Link href="/settings" className="text-blue-600 hover:underline">
              Sign in
            </Link>{" "}
            to see live progress
          </p>
        )}
        {!showBar && !wandbMissing && (
          <p className="text-sm text-slate-500">{p.current_label ?? "No progress yet."}</p>
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
              <div className="mt-2 text-xs text-slate-500">
                ~{etaLabel} left
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function PathsCard({ d }: { d: JobDetails }) {
  const rows: { label: string; value: string }[] = [
    { label: "stdout", value: d.paths.stdout },
    { label: "stderr", value: d.paths.stderr },
    { label: "exp dir", value: d.paths.exp_dir },
  ];
  if (d.paths.ckpt_dir) rows.push({ label: "checkpoints", value: d.paths.ckpt_dir });
  if (d.paths.eval_checkpoint) rows.push({ label: "checkpoint", value: d.paths.eval_checkpoint });
  if (d.paths.eval_dir) rows.push({ label: "eval results", value: d.paths.eval_dir });
  if (d.paths.isaac_logs_glob) rows.push({ label: "isaac sim logs", value: d.paths.isaac_logs_glob });

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle>Paths <span className="text-xs font-normal text-slate-500">on {d.cluster}</span></CardTitle>
      </CardHeader>
      <CardContent>
        <div className="divide-y divide-slate-100 dark:divide-slate-900">
          {rows.map((r) => (
            <div key={r.label} className="flex items-center justify-between gap-4 py-2">
              <div className="min-w-[110px] text-xs uppercase tracking-wide text-slate-500">{r.label}</div>
              <div className="flex-1 truncate font-mono text-xs">{r.value}</div>
              <CopyButton value={r.value} />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
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
  const [lines, setLines] = useState<string[]>([]);
  const preRef = useRef<HTMLPreElement>(null);
  // Whether the user was at the bottom *before* the next line arrived. If
  // they scrolled up to read history, we don't yank them back down.
  const stickToBottomRef = useRef(true);

  useEffect(() => {
    stickToBottomRef.current = true;
    if (!enabled) return;
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
      setLines((prev) => prev.concat(e.data as string));
    });
    return () => es.close();
  }, [cluster, jobId, stream, enabled]);

  useEffect(() => {
    if (preRef.current && stickToBottomRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <pre
      ref={preRef}
      className="h-96 overflow-auto rounded-md bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-100"
    >
      {!enabled
        ? "(logs unavailable — k8s pod has been garbage-collected)"
        : lines.length === 0
          ? "(waiting for log lines…)"
          : lines.join("\n")}
    </pre>
  );
}

type CheckpointEntry = { path: string; job_name: string; step: number };

function CopyCheckpointDialog({
  open,
  onOpenChange,
  cluster,
  jobId,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  cluster: string;
  jobId: string;
}) {
  const clusters = useQuery({
    queryKey: ["clusters"],
    queryFn: () =>
      api<{ clusters: string[] }>("/api/clusters").then((d) => d.clusters),
  });
  const checkpoints = useQuery({
    queryKey: ["checkpoints", cluster, jobId],
    queryFn: () =>
      api<CheckpointEntry[]>(`/api/jobs/${cluster}/${jobId}/checkpoints`),
    enabled: open,
  });
  const [destCluster, setDestCluster] = useState<string>("");
  const [destPathRoot, setDestPathRoot] = useState<string>("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleteSource, setDeleteSource] = useState<boolean>(false);

  const copy = useMutation({
    mutationFn: () =>
      api<{ copy_id: string }>(
        `/api/jobs/${cluster}/${jobId}/copy-checkpoint`,
        {
          method: "POST",
          body: JSON.stringify({
            dest_cluster: destCluster,
            dest_path_root: destPathRoot || null,
            sources: Array.from(selected),
            delete_source: deleteSource,
          }),
        },
      ),
    onSuccess: (r) => {
      const dest = destCluster;
      // Close the dialog immediately; track progress as a toast. The
      // watcher writes the copy-id to localStorage so a refresh resumes it.
      resetAndClose();
      startCopyWatcher(r.copy_id, dest);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  function resetAndClose() {
    onOpenChange(false);
    setSelected(new Set());
    setDestCluster("");
    setDestPathRoot("");
    setDeleteSource(false);
  }

  const options = (clusters.data ?? []).filter((c) => c !== cluster);

  function toggle(path: string) {
    const next = new Set(selected);
    if (next.has(path)) next.delete(path);
    else next.add(path);
    setSelected(next);
  }

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? onOpenChange(true) : resetAndClose())}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Copy checkpoint</DialogTitle>
          <DialogDescription>
            Copies the selected <code>checkpoint-N</code> dirs from{" "}
            <span className="font-mono">{cluster}</span> to another cluster.
            Slurm → slurm uses rsync; mlxp transfers use tar piped through
            kubectl exec.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
            <div className="space-y-1.5">
              <Label>Checkpoints</Label>
              {checkpoints.isLoading && (
                <p className="text-sm text-slate-500">Loading…</p>
              )}
              {checkpoints.data && checkpoints.data.length === 0 && (
                <p className="text-sm text-slate-500">
                  No checkpoints found for this variant.
                </p>
              )}
              {checkpoints.data && checkpoints.data.length > 0 && (
                <div className="max-h-56 overflow-y-auto rounded-md border border-slate-200 dark:border-slate-800">
                  {checkpoints.data.map((c) => (
                    <label
                      key={c.path}
                      className="flex cursor-pointer items-center gap-2 border-b border-slate-100 px-3 py-1.5 text-xs last:border-0 hover:bg-slate-50 dark:border-slate-900 dark:hover:bg-slate-900/40"
                    >
                      <input
                        type="checkbox"
                        checked={selected.has(c.path)}
                        onChange={() => toggle(c.path)}
                        className="h-4 w-4 rounded border-slate-300 dark:border-slate-700"
                      />
                      <span className="font-mono">step {c.step.toLocaleString()}</span>
                      <span className="ml-auto truncate font-mono text-[10px] text-slate-500">
                        {c.job_name}
                      </span>
                    </label>
                  ))}
                </div>
              )}
            </div>
            <div className="space-y-1.5">
              <Label>Destination cluster</Label>
              <Select value={destCluster} onValueChange={setDestCluster}>
                <SelectTrigger>
                  <SelectValue placeholder="pick a cluster…" />
                </SelectTrigger>
                <SelectContent>
                  {options.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Destination directory (optional)</Label>
              <Input
                value={destPathRoot}
                onChange={(e) => setDestPathRoot(e.target.value)}
                placeholder="/abs/dir (each checkpoint-N is created under it)"
                className="font-mono text-xs"
              />
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={deleteSource}
                onChange={(e) => setDeleteSource(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 dark:border-slate-700"
              />
              <span>Remove checkpoint after copy</span>
            </label>
          </div>

        <DialogFooter>
          <Button variant="outline" onClick={resetAndClose}>
            Cancel
          </Button>
          <Button
            onClick={() => copy.mutate()}
            disabled={!destCluster || selected.size === 0 || copy.isPending}
          >
            {copy.isPending
              ? "Starting…"
              : selected.size > 1
                ? `Copy ${selected.size}`
                : "Copy"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
