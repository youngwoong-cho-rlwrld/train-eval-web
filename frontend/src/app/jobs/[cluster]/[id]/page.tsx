"use client";

import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ArrowLeft, ExternalLink } from "lucide-react";
import { api, logStreamUrl, type JobDetails } from "@/lib/api";
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
import { CopyButton } from "@/components/copy-button";

export default function JobDetail({ params }: { params: Promise<{ cluster: string; id: string }> }) {
  const { cluster, id } = use(params);
  const qc = useQueryClient();

  const sacct = useQuery({
    queryKey: ["job", cluster, id],
    queryFn: () => api<Record<string, string>>(`/api/jobs/${cluster}/${id}`),
    refetchInterval: 5000,
  });

  const details = useQuery({
    queryKey: ["job-details", cluster, id],
    queryFn: () => api<JobDetails>(`/api/jobs/${cluster}/${id}/details`),
    refetchInterval: 10000,
  });

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
  const [stream, setStream] = useState<"out" | "err" | "isaac">("out");
  const [confirmCancel, setConfirmCancel] = useState(false);

  const cancelLabel = cluster === "mlxp" ? "kubectl delete job" : "scancel";

  return (
    <div className="mx-auto max-w-5xl px-8 py-12">
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
        <Button
          variant="destructive"
          size="sm"
          onClick={() => setConfirmCancel(true)}
          disabled={cancel.isPending}
        >
          {cancel.isPending ? "Cancelling…" : cancelLabel}
        </Button>
      </div>

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

      {details.data && <ProgressCard d={details.data} />}

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>sacct</CardTitle>
        </CardHeader>
        <CardContent>
          {sacct.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
          {sacct.error && <p className="text-sm text-red-600">{(sacct.error as Error).message}</p>}
          {sacct.data && (
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
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
          <LogStream cluster={cluster} jobId={id} stream={stream} />
        </CardContent>
      </Card>
    </div>
  );
}

function ProgressCard({ d }: { d: JobDetails }) {
  const p = d.progress;
  const percent = p.percent ?? 0;
  const showBar = (p.current_step !== null && p.max_steps) || (p.completed_runs !== null && p.total_runs);
  return (
    <Card className="mt-6">
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
        {!showBar && <p className="text-sm text-slate-500">{p.current_label ?? "No progress yet."}</p>}
        {showBar && (
          <>
            <div className="flex items-baseline justify-between text-sm">
              <span className="font-mono">{p.current_label}</span>
              <span className="text-slate-500">{percent.toFixed(1)}%</span>
            </div>
            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
              <div
                className="h-full rounded-full bg-slate-900 transition-all dark:bg-slate-50"
                style={{ width: `${Math.max(0, Math.min(100, percent))}%` }}
              />
            </div>
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

function LogStream({ cluster, jobId, stream }: { cluster: string; jobId: string; stream: "out" | "err" | "isaac" }) {
  const [lines, setLines] = useState<string[]>([]);
  const preRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    setLines([]);
    const es = new EventSource(logStreamUrl(cluster, jobId, stream));
    es.addEventListener("line", (e: MessageEvent) => {
      setLines((prev) => {
        const next = prev.concat(e.data as string);
        return next.length > 2000 ? next.slice(-2000) : next;
      });
    });
    return () => es.close();
  }, [cluster, jobId, stream]);

  useEffect(() => {
    if (preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight;
  }, [lines]);

  return (
    <pre
      ref={preRef}
      className="h-96 overflow-auto rounded-md bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-100"
    >
      {lines.length === 0 ? "(waiting for log lines…)" : lines.join("\n")}
    </pre>
  );
}
