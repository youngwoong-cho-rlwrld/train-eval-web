"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  api,
  type CheckpointCopyRecord,
  type CheckpointEntry,
} from "@/lib/api";
import { startCopyWatcher } from "@/lib/copy-watcher";
import { Button } from "@/components/ui/button";
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
import { CheckpointCopyList } from "@/components/checkpoint-copy-history";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";

export function CopyCheckpointDialog({
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
    enabled: open,
  });
  const checkpoints = useQuery({
    queryKey: ["checkpoints", cluster, jobId],
    queryFn: () =>
      api<CheckpointEntry[]>(`/api/jobs/${cluster}/${jobId}/checkpoints`),
    enabled: open,
  });
  const copyHistory = useQuery({
    queryKey: ["checkpoint-copies", cluster, jobId],
    queryFn: () =>
      api<CheckpointCopyRecord[]>(
        `/api/jobs/${cluster}/${jobId}/checkpoint-copies`,
      ),
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

  const clusterOptions = Array.isArray(clusters.data) ? clusters.data : [];
  const options = clusterOptions.filter((c) => c !== cluster);
  const checkpointOptions = useMemo(
    () => dedupeCheckpoints(checkpoints.data ?? []),
    [checkpoints.data],
  );

  function toggle(path: string) {
    const next = new Set(selected);
    if (next.has(path)) next.delete(path);
    else next.add(path);
    setSelected(next);
  }

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? onOpenChange(true) : resetAndClose())}>
      <DialogContent className="w-[calc(100vw-2rem)] max-w-xl overflow-hidden">
        <DialogHeader>
          <DialogTitle>Copy checkpoint</DialogTitle>
          <DialogDescription>
            Copies the selected <code>checkpoint-N</code> dirs from{" "}
            <span className="font-mono">{cluster}</span> to another cluster.
            Selected step dirs are copied under the owning run name when one
            exists.
          </DialogDescription>
        </DialogHeader>

        <div className="min-w-0 space-y-3">
          <div className="min-w-0 space-y-1.5">
            <Label>Checkpoints</Label>
            {checkpoints.isLoading && (
              <LoadingState label="Loading checkpoints..." rows={3} />
            )}
            {checkpoints.error && (
              <ErrorState message={(checkpoints.error as Error).message} />
            )}
            {checkpoints.data && checkpointOptions.length === 0 && (
              <EmptyState message="No checkpoints found for this experiment." />
            )}
            {checkpoints.data && checkpointOptions.length > 0 && (
              <div className="max-h-56 min-w-0 overflow-hidden overflow-y-auto rounded-md border border-slate-200 dark:border-slate-800">
                {checkpointOptions.map((c) => (
                  <label
                    key={c.path}
                    className="grid min-w-0 cursor-pointer grid-cols-[auto_auto_minmax(0,1fr)] items-center gap-2 border-b border-slate-100 px-3 py-1.5 text-xs last:border-0 hover:bg-slate-50 dark:border-slate-900 dark:hover:bg-slate-900/40"
                  >
                    <input
                      type="checkbox"
                      checked={selected.has(c.path)}
                      onChange={() => toggle(c.path)}
                      className="h-4 w-4 rounded border-slate-300 dark:border-slate-700"
                    />
                    <span className="whitespace-nowrap font-mono">
                      step {c.step.toLocaleString()}
                    </span>
                    <span
                      className="min-w-0 truncate text-right font-mono text-[10px] text-slate-500"
                      title={c.job_name}
                    >
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
              <SelectTrigger className="min-w-0">
                <SelectValue placeholder="pick a cluster..." />
              </SelectTrigger>
              <SelectContent>
                {options.map((c) => (
                  <SelectItem key={c} value={c}>
                    {c}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {clusters.isLoading && (
              <LoadingState label="Loading destination clusters..." rows={1} />
            )}
            {clusters.error && (
              <ErrorState message={(clusters.error as Error).message} />
            )}
          </div>
          <div className="space-y-1.5">
            <Label>Destination directory (optional)</Label>
            <Input
              value={destPathRoot}
              onChange={(e) => setDestPathRoot(e.target.value)}
              placeholder="/abs/dir (each selected checkpoint is created under it)"
              className="min-w-0 font-mono text-xs"
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
          <PreviousCopies history={copyHistory} />
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
              ? "Starting..."
              : selected.size > 1
                ? `Copy ${selected.size}`
                : "Copy"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function dedupeCheckpoints(rows: CheckpointEntry[]) {
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = row.path.replace(/\/+$/, "");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function PreviousCopies({
  history,
}: {
  history: UseQueryResult<CheckpointCopyRecord[], Error>;
}) {
  return (
    <div className="min-w-0 border-t border-slate-200 pt-3 dark:border-slate-800">
      <Label>Previous copies</Label>
      <div className="mt-1.5">
        {history.isLoading && <LoadingState label="Loading previous copies..." rows={1} />}
        {history.error && <ErrorState message={history.error.message} />}
        {history.data && history.data.length === 0 && (
          <EmptyState message="No copied checkpoints recorded for this job." />
        )}
        {history.data && history.data.length > 0 && (
          <CheckpointCopyList
            records={history.data}
            className="max-h-36 min-w-0 overflow-x-hidden overflow-y-auto rounded-md border border-slate-200 dark:border-slate-800"
            itemClassName="min-w-0 border-b border-slate-100 px-3 py-2 text-xs last:border-0 dark:border-slate-900"
          />
        )}
      </div>
    </div>
  );
}
