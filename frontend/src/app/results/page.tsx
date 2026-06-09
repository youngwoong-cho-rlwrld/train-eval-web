"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useIsFetching, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Check, CircleHelp, Copy, Database, ExternalLink, Trophy } from "lucide-react";
import { toast } from "sonner";
import { api, type ResultCell, type ResultsResponse, type ResultTask, type ResultVariant } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { RefreshButton } from "@/components/refresh-button";
import { LoadingState } from "@/components/loading-state";
import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { JobStateBadge } from "@/components/job-state-badge";
import { Th } from "@/components/table";
import { jobDetailHref } from "@/lib/job-links";
import { formatPct } from "@/lib/format";

const REFRESH_MS = 120_000;
const AVERAGE_HELP =
  "Ave is total successes divided by total episodes across all displayed eval sets for this task. It is computed from per-run eval_results/**/run_*/results.json files, not the top-level aggregate results.json.";

export default function ResultsPage() {
  const qc = useQueryClient();
  const [cluster, setCluster] = useState("all");
  const [nameFilter, setNameFilter] = useState("");

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

  const allVariants = useMemo(() => resultsQuery.data?.variants ?? [], [resultsQuery.data?.variants]);
  const variants = useMemo(
    () => allVariants.filter((variant) => resultVariantMatchesName(variant, nameFilter)),
    [allVariants, nameFilter],
  );
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
            <span>{variants.length} experiments</span>
            <span className="text-slate-300 dark:text-slate-700">/</span>
            <span>{taskCount} tasks</span>
            <span className="text-slate-300 dark:text-slate-700">/</span>
            <span>{evalCellCount} eval-set summaries</span>
          </div>
        </div>
        <div className="flex flex-wrap items-end justify-end gap-3">
          <div className="space-y-1">
            <Label htmlFor="result-name" className="text-xs text-slate-500">Name</Label>
            <Input
              id="result-name"
              value={nameFilter}
              onChange={(e) => setNameFilter(e.target.value)}
              placeholder="experiment"
              className="h-8 w-[220px] font-mono text-xs"
            />
          </div>
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
          <CardContent className="py-8">
            <LoadingState label="Loading results..." rows={5} />
          </CardContent>
        </Card>
      )}

      {!resultsQuery.isLoading && allVariants.length === 0 && !resultsQuery.error && (
        <Card className="mt-8">
          <CardContent className="flex items-center gap-3 py-8 text-sm text-slate-500">
            <Database className="h-4 w-4" />
            No eval result artifacts found for the selected cluster.
          </CardContent>
        </Card>
      )}

      {!resultsQuery.isLoading && allVariants.length > 0 && variants.length === 0 && !resultsQuery.error && (
        <Card className="mt-8">
          <CardContent className="flex items-center gap-3 py-8 text-sm text-slate-500">
            <Database className="h-4 w-4" />
            No results match this name filter.
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

function resultVariantMatchesName(variant: ResultVariant, filter: string) {
  const needle = filter.trim().toLowerCase();
  if (!needle) return true;
  return [
    variant.variant,
    variant.experiment ?? "",
    variant.note ?? "",
    variant.job_state ?? "",
    variant.checkpoint ?? "",
    variant.checkpoint_job_cluster ?? "",
    variant.checkpoint_job_name ?? "",
    variant.source ?? "",
  ].join(" ").toLowerCase().includes(needle);
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
          <ResultCard
            key={resultVariantKey(variant)}
            variant={variant}
          />
        ))}
      </div>
    </section>
  );
}

function ResultCard({ variant }: { variant: ResultVariant }) {
  const evalSets = evalSetColumns(variant.tasks);
  const nRuns = variant.n_runs ?? maxExpectedRuns(variant.tasks);
  const nEpisodes = variant.n_episodes ?? maxEpisodeCount(variant.tasks);
  const checkpointName = variant.checkpoint ? checkpointDisplayName(variant.checkpoint) : null;
  const checkpointLabel = checkpointName ?? variant.checkpoint ?? "";
  const checkpointJobHref = jobDetailHref(
    variant.checkpoint_job_cluster ?? variant.cluster,
    variant.checkpoint_job_id,
  );

  return (
    <Card>
      <CardHeader className="gap-3 sm:flex-row sm:items-start sm:justify-between sm:space-y-0">
        <div className="min-w-0">
          <CardTitle className="font-mono text-base">
            <ResultTitle variant={variant} />
          </CardTitle>
          <CardDescription className="mt-1">
            {variant.note || "eval results"}
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
                <Th>
                  <span className="inline-flex items-center gap-1">
                    Ave
                    <ImmediateTooltip content={AVERAGE_HELP}>
                      <CircleHelp className="h-3.5 w-3.5 text-slate-400" />
                    </ImmediateTooltip>
                  </span>
                </Th>
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
            <Meta
              label="checkpoint"
              value={
                variant.checkpoint_job_id
                  ? `${checkpointLabel} (${variant.checkpoint_job_id})`
                  : checkpointLabel
              }
              title={variant.checkpoint_job_name ? `Open ${variant.checkpoint_job_name}` : variant.checkpoint}
              href={checkpointJobHref}
            />
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

function ResultTitle({ variant }: { variant: ResultVariant }) {
  if (!variant.job_id) return variant.variant;
  return (
    <span className="inline-flex max-w-full min-w-0 items-center gap-2">
      <Link
        href={jobDetailHref(variant.cluster, variant.job_id)!}
        target="_blank"
        rel="noreferrer"
        className="inline-flex min-w-0 items-center gap-1.5 text-blue-600 hover:underline dark:text-blue-400"
        title={variant.job_name ? `Open ${variant.job_name}` : "Open job detail"}
      >
        <span className="truncate">{variant.variant}</span>
        <span className="shrink-0">({variant.job_id})</span>
        <ExternalLink className="h-3.5 w-3.5 shrink-0" />
      </Link>
      {variant.job_state && <JobStateBadge state={variant.job_state} />}
    </span>
  );
}

function resultVariantKey(variant: ResultVariant) {
  return [
    variant.cluster,
    variant.variant,
    variant.source,
    variant.experiment,
    variant.job_id,
    variant.checkpoint,
  ].filter(Boolean).join(":");
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
      await writeResultTableToClipboard(variant, evalSets);
      setCopied(true);
      toast.success("Result table copied");
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      toast.error(`Copy failed: ${(e as Error).message}`);
    }
  }

  return (
    <ImmediateTooltip content="Copy TSV table for Google Sheets or Notion">
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-6 gap-1.5 px-2 text-xs"
        onClick={copyTable}
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
        Copy table
      </Button>
    </ImmediateTooltip>
  );
}

async function writeResultTableToClipboard(variant: ResultVariant, evalSets: string[]) {
  const tsv = resultTableTsv(variant, evalSets);
  if (!("ClipboardItem" in window) || !navigator.clipboard.write) {
    await navigator.clipboard.writeText(tsv);
    return;
  }

  const html = resultTableHtml(variant, evalSets);
  await navigator.clipboard.write([
    new ClipboardItem({
      "text/html": new Blob([html], { type: "text/html" }),
      "text/plain": new Blob([tsv], { type: "text/plain" }),
    }),
  ]);
}

function ResultRow({ task, evalSets }: { task: ResultTask; evalSets: string[] }) {
  const byEvalSet = new Map(task.eval_sets.map((cell) => [cell.eval_set, cell]));
  const average = episodeWeightedAverage(task.eval_sets);

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
  const episodes = totalEpisodes(cell);
  const tooltip = cell.per_run_success_rate.length
    ? cell.per_run_success_rate.map((v) => formatPct(v)).join(", ")
    : "partial";

  return (
    <ImmediateTooltip content={tooltip} className="inline-flex">
      <div>
        <div className="font-mono">{formatMeanStd(cell)}</div>
        <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
          <span>{cell.completed_runs}{cell.expected_runs ? `/${cell.expected_runs}` : ""} runs</span>
          {episodes != null && <span>{episodes.toLocaleString()} episodes</span>}
          {incomplete && <Badge variant="warning" className="px-1 py-0 text-[10px]">partial</Badge>}
        </div>
      </div>
    </ImmediateTooltip>
  );
}

function Meta({ label, value, title, href }: { label: string; value: string; title?: string; href?: string }) {
  const body = href ? (
    <Link
      href={href}
      target="_blank"
      rel="noreferrer"
      className="font-mono text-blue-600 hover:underline dark:text-blue-400"
    >
      {value}
    </Link>
  ) : (
    <span className="font-mono">{value}</span>
  );

  return (
    <div className="min-w-0">
      <span className="mr-1 text-slate-400">{label}:</span>
      <ImmediateTooltip content={title ?? value}>
        {body}
      </ImmediateTooltip>
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

function checkpointDisplayName(path: string) {
  const parts = path.split("/").filter(Boolean);
  const leaf = parts[parts.length - 1] ?? path;
  if (!leaf.startsWith("checkpoint-")) return leaf;
  return parts[parts.length - 2] ?? leaf;
}

function resultTableTsv(variant: ResultVariant, evalSets: string[]) {
  return resultTableRows(variant, evalSets).map((row) => row.map(tsvCell).join("\t")).join("\n");
}

function resultTableHtml(variant: ResultVariant, evalSets: string[]) {
  const [header, ...body] = resultTableRows(variant, evalSets);
  return [
    '<table border="1" cellspacing="0" cellpadding="4">',
    "<thead>",
    `<tr>${header.map((value) => `<th>${htmlCell(value)}</th>`).join("")}</tr>`,
    "</thead>",
    "<tbody>",
    ...body.map((row) => `<tr>${row.map((value) => `<td>${htmlCell(value)}</td>`).join("")}</tr>`),
    "</tbody>",
    "</table>",
  ].join("");
}

function resultTableRows(variant: ResultVariant, evalSets: string[]) {
  const rows = [
    ["Task", ...evalSets, "Ave"],
    ...variant.tasks.map((task) => {
      const byEvalSet = new Map(task.eval_sets.map((cell) => [cell.eval_set, cell]));
      const average = episodeWeightedAverage(task.eval_sets);
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
  return rows;
}

function tsvCell(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function htmlCell(value: string) {
  return tsvCell(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatMeanStd(cell: ResultCell) {
  if (cell.mean_success_rate == null) return "--";
  return `${formatPct(cell.mean_success_rate)} ± ${formatPct(cell.std_success_rate ?? 0)}`;
}

function totalEpisodes(cell: ResultCell) {
  const total = cell.episode_counts.reduce<number>((sum, v) => sum + (v ?? 0), 0);
  return total > 0 ? total : null;
}

function episodeWeightedAverage(cells: ResultCell[]) {
  let successes = 0;
  let episodes = 0;

  for (const cell of cells) {
    cell.episode_counts.forEach((episodeCount, idx) => {
      if (episodeCount == null || episodeCount <= 0) return;
      const successCount = cell.success_counts[idx];
      if (successCount != null) {
        episodes += episodeCount;
        successes += successCount;
        return;
      }
      const runRate = cell.per_run_success_rate[idx];
      if (runRate == null) return;
      episodes += episodeCount;
      successes += runRate * episodeCount;
    });
  }

  if (episodes > 0) return successes / episodes;
  return null;
}

