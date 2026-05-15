"use client";

import { useIsFetching, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type MlxpNode, type Partition } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { RefreshButton } from "@/components/refresh-button";
import { useMyMlxpNode } from "@/hooks/use-my-mlxp-node";

const REFRESH_MS = 60_000;

export default function MonitorPage() {
  const qc = useQueryClient();

  const clusters = useQuery({
    queryKey: ["clusters"],
    queryFn: () => api<{ clusters: string[] }>("/api/clusters").then((d) => d.clusters),
  });
  // mlxp is k8s, not slurm — it has its own panel below.
  const slurm = (clusters.data ?? []).filter((c) => c !== "mlxp");

  const refreshAll = () => {
    qc.invalidateQueries({ queryKey: ["clusters"] });
    for (const c of slurm) qc.invalidateQueries({ queryKey: ["partitions", c] });
    qc.invalidateQueries({ queryKey: ["mlxp-gpus"] });
  };

  const isFetching =
    useIsFetching({
      predicate: (q) => {
        const k = q.queryKey[0];
        return k === "clusters" || k === "partitions" || k === "mlxp-gpus";
      },
    }) > 0;

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">GPU monitor</h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            All clusters in one view. Auto-refresh every {REFRESH_MS / 1000}s.
          </p>
        </div>
        <RefreshButton isFetching={isFetching} onRefresh={refreshAll} />
      </div>

      <div className="mt-8 space-y-6">
        {slurm.map((c) => (
          <SlurmClusterPanel key={c} cluster={c} />
        ))}
        <MlxpPanel />
      </div>
    </div>
  );
}

function SlurmClusterPanel({ cluster }: { cluster: string }) {
  const q = useQuery({
    queryKey: ["partitions", cluster],
    queryFn: () => api<Partition[]>(`/api/clusters/${cluster}/partitions`),
    refetchInterval: REFRESH_MS,
  });
  const ps = q.data ?? [];
  const idle = ps.reduce((s, p) => s + p.gpu_idle, 0);
  const total = ps.reduce((s, p) => s + p.gpu_total, 0);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-baseline justify-between">
          <CardTitle className="text-base">{cluster} <span className="ml-1 text-xs font-normal text-slate-500">slurm</span></CardTitle>
          <span className="font-mono text-sm">
            <span className={idle > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>{idle}</span>
            <span className="text-slate-400"> / {total} GPU free</span>
          </span>
        </div>
        <CardDescription className="text-xs">{ps.length} partitions</CardDescription>
      </CardHeader>
      <CardContent>
        {q.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
        {q.error && <p className="text-sm text-red-600">{(q.error as Error).message}</p>}
        {ps.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
                <tr>
                  <th className="py-2 pr-4 font-medium">Partition</th>
                  <th className="py-2 pr-4 font-medium">GPUs free / total</th>
                  <th className="py-2 pr-4 font-medium">Nodes idle / total</th>
                  <th className="py-2 pr-4 font-medium">States</th>
                </tr>
              </thead>
              <tbody>
                {ps.map((p) => (
                  <tr key={p.name} className="border-b border-slate-100 last:border-0 dark:border-slate-900">
                    <td className="py-2 pr-4 font-mono text-xs">
                      {p.name}
                      {p.is_default && <Badge variant="secondary" className="ml-1 text-[10px]">default</Badge>}
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">
                      <span className={p.gpu_idle > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>
                        {p.gpu_idle}
                      </span>
                      <span className="text-slate-400"> / {p.gpu_total}</span>
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">
                      {p.idle_nodes} / {p.total_nodes}
                    </td>
                    <td className="py-2 pr-4 text-xs">
                      {Object.entries(p.states).map(([st, n]) => (
                        <span key={st} className="mr-2 font-mono">
                          <span className={st === "idle" ? "text-green-600 dark:text-green-400" : "text-slate-500"}>{st}</span>:{n}
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function MlxpPanel() {
  const q = useQuery({
    queryKey: ["mlxp-gpus"],
    queryFn: () => api<MlxpNode[]>("/api/mlxp/gpus"),
    refetchInterval: REFRESH_MS,
    retry: false,
  });
  const nodes = q.data ?? [];
  // Same source of truth as the /submit page's Node picker.
  const [yoursNode, setYoursNode] = useMyMlxpNode();
  const idle = nodes.reduce((s, n) => s + n.gpu_free, 0);
  const total = nodes.reduce((s, n) => s + n.gpu_total, 0);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-baseline justify-between">
          <CardTitle className="text-base">MLXP <span className="ml-1 text-xs font-normal text-slate-500">naver, k8s</span></CardTitle>
          <span className="font-mono text-sm">
            <span className={idle > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>{idle}</span>
            <span className="text-slate-400"> / {total} GPU free</span>
          </span>
        </div>
        <CardDescription className="flex items-center gap-2 text-xs">
          <span>your node:</span>
          <Select value={yoursNode} onValueChange={setYoursNode}>
            <SelectTrigger className="h-7 w-auto min-w-[200px] gap-1 px-2 text-xs">
              <SelectValue placeholder="select…" />
            </SelectTrigger>
            <SelectContent>
              {nodes.map((n) => (
                <SelectItem key={n.name} value={n.name}>
                  <span className="font-mono">{n.name}</span>
                  <span className={`ml-2 text-[10px] ${n.gpu_free > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}`}>
                    {n.gpu_free}/{n.gpu_total} free
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="text-slate-400">(persists across sessions + syncs with the Submit page)</span>
        </CardDescription>
      </CardHeader>
      <CardContent>
        {q.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
        {q.error && <p className="text-sm text-red-600">{(q.error as Error).message}</p>}
        {nodes.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
                <tr>
                  <th className="py-2 pr-4 font-medium">Node</th>
                  <th className="py-2 pr-4 font-medium">GPUs free / total</th>
                </tr>
              </thead>
              <tbody>
                {nodes.map((n) => (
                  <tr key={n.name} className="border-b border-slate-100 last:border-0 dark:border-slate-900">
                    <td className="py-2 pr-4 font-mono text-xs">
                      {n.name}
                      {n.name === yoursNode && <Badge variant="default" className="ml-1 text-[10px]">yours</Badge>}
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs">
                      <span className={n.gpu_free > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>
                        {n.gpu_free}
                      </span>
                      <span className="text-slate-400"> / {n.gpu_total}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
