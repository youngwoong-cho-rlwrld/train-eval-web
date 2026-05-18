"use client";

import { useMemo, useState } from "react";
import { useIsFetching, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Check, Copy, Database, Trophy } from "lucide-react";
import { toast } from "sonner";
import { api, type ResultCell, type ResultsResponse, type ResultTask, type ResultVariant } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { RefreshButton } from "@/components/refresh-button";

const REFRESH_MS = 120_000;

export default function ResultsPage() {
  const qc = useQueryClient();
  const [cluster, setCluster] = useState("all");

  const clustersQuery = useQuery({
    queryKey: ["clusters"],
    queryFn: () => api<{ clusters: string[] }>("/api/clusters").then((d) => d.clusters),
    staleTime: 5 * 60_000,
  });
  const clusterOptions = (clustersQuery.data ?? []).filter((c) => c !== "mlxp");

  const resultsQuery = useQuery({
    queryKey: ["results", cluster],
    queryFn: () =>
      api<ResultsResponse>(
        cluster === "all" ? "/api/results" : `/api/results?cluster=${encodeURIComponent(cluster)}`,
      ),
    refetchInterval: REFRESH_MS,
  });

  const isFetching =
    useIsFetching({
      predicate: (q) => q.queryKey[0] === "results" || q.queryKey[0] === "clusters",
    }) > 0;

  const variants = useMemo(() => resultsQuery.data?.variants ?? [], [resultsQuery.data?.variants]);
  const { singleTask, multiTask, taskCount, evalCellCount } = useMemo(() => {
    const singleTask = variants.filter((v) => v.tasks.length <= 1);
    const multiTask = variants.filter((v) => v.tasks.length > 1);
    return {
      singleTask,
      multiTask,
      taskCount: variants.reduce((sum, v) => sum + v.tasks.length, 0),
      evalCellCount: variants.reduce(
        (sum, v) => sum + v.tasks.reduce((inner, t) => inner + t.eval_sets.length, 0),
        0,
      ),
    };
  }, [variants]);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["results"] });
    qc.invalidateQueries({ queryKey: ["clusters"] });
  };

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <div className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Results</h1>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
            <span>{variants.length} variants</span>
            <span className="text-slate-300 dark:text-slate-700">/</span>
            <span>{taskCount} tasks</span>
            <span className="text-slate-300 dark:text-slate-700">/</span>
            <span>{evalCellCount} eval-set summaries</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Select value={cluster} onValueChange={setCluster}>
            <SelectTrigger className="h-8 w-[150px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All clusters</SelectItem>
              {clusterOptions.map((c) => (
                <SelectItem key={c} value={c}>
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <RefreshButton isFetching={isFetching} onRefresh={refresh} intervalMs={REFRESH_MS} />
        </div>
      </div>

      {resultsQuery.error && (
        <ErrorBanner message={(resultsQuery.error as Error).message} />
      )}
      {resultsQuery.data?.errors.map((err) => (
        <ErrorBanner key={err.cluster} message={`${err.cluster}: ${err.error}`} />
      ))}

      {resultsQuery.isLoading && (
        <Card className="mt-8">
          <CardContent className="py-8 text-sm text-slate-500">Loading results...</CardContent>
        </Card>
      )}

      {!resultsQuery.isLoading && variants.length === 0 && !resultsQuery.error && (
        <Card className="mt-8">
          <CardContent className="flex items-center gap-3 py-8 text-sm text-slate-500">
            <Database className="h-4 w-4" />
            No eval result artifacts found for the selected cluster.
          </CardContent>
        </Card>
      )}

      {multiTask.length > 0 && (
        <ResultSection title="Multitask" variants={multiTask} />
      )}
      {singleTask.length > 0 && (
        <ResultSection title="Single Task" variants={singleTask} />
      )}
    </div>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="mt-6 flex items-start gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-300">
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

function ResultSection({ title, variants }: { title: string; variants: ResultVariant[] }) {
  return (
    <section className="mt-8">
      <div className="mb-3 flex items-center gap-2">
        <Trophy className="h-4 w-4 text-slate-500" />
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {title}
        </h2>
      </div>
      <div className="space-y-5">
        {variants.map((variant) => (
          <ResultCard key={`${variant.cluster}-${variant.variant}`} variant={variant} />
        ))}
      </div>
    </section>
  );
}

function ResultCard({ variant }: { variant: ResultVariant }) {
  const evalSets = evalSetColumns(variant.tasks);
  const nRuns = variant.n_runs ?? maxExpectedRuns(variant.tasks);
  const nEpisodes = variant.n_episodes ?? maxEpisodeCount(variant.tasks);

  return (
    <Card>
      <CardHeader className="gap-3 sm:flex-row sm:items-start sm:justify-between sm:space-y-0">
        <div className="min-w-0">
          <CardTitle className="truncate font-mono text-base">{variant.variant}</CardTitle>
          <CardDescription className="mt-1">
            {variant.note || variant.experiment || "eval results"}
          </CardDescription>
        </div>
        <div className="flex shrink-0 flex-wrap justify-end gap-2">
          <CopyResultTableButton variant={variant} evalSets={evalSets} />
          <Badge variant="outline">{variant.cluster}</Badge>
          {variant.model_version && <Badge variant="secondary">{variant.model_version}</Badge>}
          <Badge variant={variant.tasks.length > 1 ? "default" : "outline"}>
            {variant.tasks.length > 1 ? "multitask" : "single"}
          </Badge>
          {nRuns != null && <Badge variant="secondary">{nRuns} runs</Badge>}
          {nEpisodes != null && <Badge variant="secondary">{nEpisodes} episodes</Badge>}
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800">
              <tr>
                <Th>Task</Th>
                {evalSets.map((evalSet) => (
                  <Th key={evalSet}>{evalSet}</Th>
                ))}
                <Th>Ave</Th>
              </tr>
            </thead>
            <tbody>
              {variant.tasks.map((task) => (
                <ResultRow key={task.task} task={task} evalSets={evalSets} />
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 grid gap-2 text-xs text-slate-500 dark:text-slate-400 md:grid-cols-2">
          {variant.checkpoint && (
            <Meta label="checkpoint" value={basename(variant.checkpoint)} title={variant.checkpoint} />
          )}
          {variant.source && (
            <Meta label="source" value={basename(variant.source)} title={variant.source} />
          )}
          {variant.num_envs_per_gpu != null && (
            <Meta label="num envs / gpu" value={String(variant.num_envs_per_gpu)} />
          )}
          {variant.total_num_envs != null && (
            <Meta label="total num envs" value={String(variant.total_num_envs)} />
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function CopyResultTableButton({
  variant,
  evalSets,
}: {
  variant: ResultVariant;
  evalSets: string[];
}) {
  const [copied, setCopied] = useState(false);

  async function copyTable() {
    try {
      await navigator.clipboard.writeText(resultTableTsv(variant, evalSets));
      setCopied(true);
      toast.success("Result table copied");
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      toast.error(`Copy failed: ${(e as Error).message}`);
    }
  }

  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className="h-6 gap-1.5 px-2 text-xs"
      onClick={copyTable}
      title="Copy TSV table for Google Sheets or Notion"
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      Copy table
    </Button>
  );
}

function ResultRow({ task, evalSets }: { task: ResultTask; evalSets: string[] }) {
  const byEvalSet = new Map(task.eval_sets.map((cell) => [cell.eval_set, cell]));
  const average = mean(task.eval_sets.map((cell) => cell.mean_success_rate));

  return (
    <tr className="border-b border-slate-100 last:border-0 dark:border-slate-900">
      <td className="min-w-[260px] py-3 pr-4 align-top">
        <div className="font-medium">{displayTaskName(task)}</div>
        {task.instruction && (
          <div className="mt-1 max-w-[520px] text-xs text-slate-500 dark:text-slate-400">
            {task.instruction}
          </div>
        )}
      </td>
      {evalSets.map((evalSet) => (
        <td key={evalSet} className="min-w-[138px] py-3 pr-4 align-top">
          <ResultCellView cell={byEvalSet.get(evalSet)} />
        </td>
      ))}
      <td className="min-w-[90px] py-3 pr-4 align-top font-mono text-sm">
        {average == null ? <span className="text-slate-400">--</span> : formatPct(average)}
      </td>
    </tr>
  );
}

function ResultCellView({ cell }: { cell?: ResultCell }) {
  if (!cell) return <span className="text-slate-400">--</span>;
  const incomplete =
    cell.expected_runs != null &&
    cell.expected_runs > 0 &&
    cell.completed_runs < cell.expected_runs;
  return (
    <div title={cell.per_run_success_rate.map((v) => formatPct(v)).join(", ")}>
      <div className="font-mono">{formatMeanStd(cell)}</div>
      <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
        <span>{cell.completed_runs}{cell.expected_runs ? `/${cell.expected_runs}` : ""} runs</span>
        {incomplete && <Badge variant="warning" className="px-1 py-0 text-[10px]">partial</Badge>}
      </div>
    </div>
  );
}

function Meta({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="min-w-0">
      <span className="mr-1 text-slate-400">{label}:</span>
      <span className="font-mono" title={title ?? value}>{value}</span>
    </div>
  );
}

function evalSetColumns(tasks: ResultTask[]) {
  const cols: string[] = [];
  for (const task of tasks) {
    for (const cell of task.eval_sets) {
      if (!cols.includes(cell.eval_set)) cols.push(cell.eval_set);
    }
  }
  return cols;
}

function maxExpectedRuns(tasks: ResultTask[]) {
  const vals = tasks.flatMap((task) => task.eval_sets.map((cell) => cell.expected_runs ?? cell.completed_runs));
  return vals.length ? Math.max(...vals) : null;
}

function maxEpisodeCount(tasks: ResultTask[]) {
  const vals = tasks.flatMap((task) =>
    task.eval_sets.flatMap((cell) => cell.episode_counts.filter((v): v is number => v != null)),
  );
  return vals.length ? Math.max(...vals) : null;
}

function displayTaskName(task: ResultTask) {
  return (task.task_name || task.task).replace(/^task-/, "");
}

function basename(path: string) {
  const parts = path.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? path;
}

function resultTableTsv(variant: ResultVariant, evalSets: string[]) {
  const rows = [
    ["Task", ...evalSets, "Ave"],
    ...variant.tasks.map((task) => {
      const byEvalSet = new Map(task.eval_sets.map((cell) => [cell.eval_set, cell]));
      const average = mean(task.eval_sets.map((cell) => cell.mean_success_rate));
      return [
        displayTaskName(task),
        ...evalSets.map((evalSet) => {
          const cell = byEvalSet.get(evalSet);
          return cell ? formatMeanStd(cell) : "";
        }),
        average == null ? "" : formatPct(average),
      ];
    }),
  ];
  return rows.map((row) => row.map(tsvCell).join("\t")).join("\n");
}

function tsvCell(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function formatMeanStd(cell: ResultCell) {
  return `${formatPct(cell.mean_success_rate)} ± ${formatPct(cell.std_success_rate)}`;
}

function formatPct(value: number) {
  return (value * 100).toFixed(2);
}

function mean(vals: number[]) {
  if (vals.length === 0) return null;
  return vals.reduce((sum, v) => sum + v, 0) / vals.length;
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="py-2 pr-4 font-medium whitespace-nowrap">{children}</th>;
}
