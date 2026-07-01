"use client";

import { useState } from "react";
import Link from "next/link";
import { keepPreviousData, useIsFetching, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, ChevronRight, CircleHelp, Copy, Database, ExternalLink, Table2 } from "lucide-react";
import { toast } from "sonner";
import { api, type ResultCell, type ResultsResponse, type ResultTask, type ResultVariant } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { RefreshButton } from "@/components/refresh-button";
import { ErrorState, InlineLoading, LoadingState } from "@/components/loading-state";
import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { JobStateBadge } from "@/components/job-state-badge";
import { isActiveJobState, primaryJobState } from "@/lib/job-status";
import { Th } from "@/components/table";
import { jobDetailHref } from "@/lib/job-links";
import { basename, formatPct } from "@/lib/format";
import { copyRich } from "@/lib/clipboard";

const REFRESH_MS = 120_000;
const AVERAGE_HELP =
  "Ave is total successes divided by total episodes across all displayed eval sets for this task. It is computed from per-run eval_results/**/run_*/results.json files, not the top-level aggregate results.json.";

// The standalone results-sheet-viewer runs on :3001 alongside this app, so
// default to the same host on that port (works on localhost and the deployed
// tailnet host alike). Override with NEXT_PUBLIC_RESULTS_VIEWER_URL.
function resultsViewerUrl(): string {
  const override = process.env.NEXT_PUBLIC_RESULTS_VIEWER_URL;
  if (override) return override;
  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:3001`;
  }
  return "http://localhost:3001";
}

export default function ResultsPage() {
  const qc = useQueryClient();
  const [cluster, setCluster] = useState("all");
  const [nameFilter, setNameFilter] = useState("");
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  const clustersQuery = useQuery({
    queryKey: ["clusters"],
    queryFn: () => api<{ clusters: string[] }>("/api/clusters").then((d) => d.clusters),
    staleTime: 5 * 60_000,
  });
  const clusterNames = clustersQuery.data ?? [];
  const clusterOptions = clusterNames.filter((c) => c !== "mlxp");

  // One query per cluster so each cluster's results render as soon as its
  // probe returns, instead of first paint waiting on the slowest cluster.
  // The cluster dropdown filters the merged data client-side.
  const resultQueries = useQueries({
    queries: clusterNames.map((c) => ({
      queryKey: ["results", c],
      queryFn: () => api<ResultsResponse>(`/api/results?cluster=${encodeURIComponent(c)}`),
      refetchInterval: REFRESH_MS,
      placeholderData: keepPreviousData,
    })),
  });

  const isFetching =
    useIsFetching({
      predicate: (q) => q.queryKey[0] === "results" || q.queryKey[0] === "clusters",
    }) > 0;

  const anyLoaded = resultQueries.some((q) => q.data !== undefined);
  const probing = clusterNames.filter((_, i) => resultQueries[i]?.isLoading);
  const initialLoading =
    clustersQuery.isLoading || (clusterNames.length > 0 && !anyLoaded && probing.length > 0);

  const queryErrors = clusterNames.flatMap((c, i) => {
    const error = resultQueries[i]?.error;
    return error ? [{ cluster: c, error: (error as Error).message }] : [];
  });
  const backendErrors = resultQueries.flatMap((q) => q.data?.errors ?? []);

  const allVariants = resultQueries.flatMap((q) => q.data?.variants ?? []);
  const variants = allVariants.filter(
    (variant) =>
      (cluster === "all" || variant.cluster === cluster) &&
      resultVariantMatchesName(variant, nameFilter),
  );
  const taskCount = variants.reduce((sum, v) => sum + v.tasks.length, 0);
  const evalCellCount = variants.reduce(
    (sum, v) => sum + v.tasks.reduce((inner, t) => inner + t.eval_sets.length, 0),
    0,
  );

  const groups = groupByExperiment(variants);
  // A narrow name filter means the user is hunting something specific —
  // open the few matching groups so their tables are visible immediately.
  const autoOpen = nameFilter.trim().length > 0 && groups.length <= 2;

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
            <span>{groups.length} experiments</span>
            <span className="text-slate-300 dark:text-slate-700">/</span>
            <span>{variants.length} results</span>
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
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 gap-1.5"
            onClick={() => window.open(resultsViewerUrl(), "_blank", "noopener,noreferrer")}
          >
            <Table2 className="h-4 w-4" />
            Table View
          </Button>
          {probing.length > 0 && (
            <InlineLoading label={`Loading ${probing.join(", ")}...`} className="h-8" />
          )}
          <RefreshButton isFetching={isFetching} onRefresh={refresh} intervalMs={REFRESH_MS} />
        </div>
      </div>

      {queryErrors.map((err) => (
        <ErrorState key={`query-${err.cluster}`} message={`${err.cluster}: ${err.error}`} className="mt-6" />
      ))}
      {backendErrors.map((err) => (
        <ErrorState key={err.cluster} message={`${err.cluster}: ${err.error}`} className="mt-6" />
      ))}

      {initialLoading && (
        <Card className="mt-8">
          <CardContent className="py-8">
            <LoadingState label="Loading results..." rows={5} />
          </CardContent>
        </Card>
      )}

      {!initialLoading && anyLoaded && probing.length === 0 && allVariants.length === 0 && (
        <Card className="mt-8">
          <CardContent className="flex items-center gap-3 py-8 text-sm text-slate-500">
            <Database className="h-4 w-4" />
            No eval result artifacts found.
          </CardContent>
        </Card>
      )}

      {!initialLoading && allVariants.length > 0 && variants.length === 0 && (
        <Card className="mt-8">
          <CardContent className="flex items-center gap-3 py-8 text-sm text-slate-500">
            <Database className="h-4 w-4" />
            No results match the current name/cluster filter.
          </CardContent>
        </Card>
      )}

      {groups.length > 0 && (
        <div className="mt-8 space-y-3">
          {groups.map((group) => (
            <ExperimentGroup
              key={group.name}
              group={group}
              expanded={openGroups[group.name] ?? autoOpen}
              onToggle={() =>
                setOpenGroups((prev) => ({
                  ...prev,
                  [group.name]: !(prev[group.name] ?? autoOpen),
                }))
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

function resultVariantMatchesName(variant: ResultVariant, filter: string) {
  const needle = filter.trim().toLowerCase();
  if (!needle) return true;
  return variant.variant.toLowerCase().includes(needle);
}

type ResultGroup = {
  name: string;
  members: ResultVariant[];
  latest: number;
};

function groupByExperiment(variants: ResultVariant[]): ResultGroup[] {
  const byName = new Map<string, ResultVariant[]>();
  for (const variant of variants) {
    const name = variant.variant || "(unknown experiment)";
    const members = byName.get(name);
    if (members) members.push(variant);
    else byName.set(name, [variant]);
  }
  return [...byName.entries()]
    .map(([name, members]) => {
      const sorted = [...members].sort((a, b) => resultRecency(b) - resultRecency(a));
      return { name, members: sorted, latest: resultRecency(sorted[0]) };
    })
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
}

// Best-effort recency: prefer the eval completion time reported by the
// backend, then the YYYYMMDD_HHMMSS stamp jobs carry in their names or
// namespaces (epoch ms), falling back to the numeric job id. Epoch values
// dwarf job ids, so stamped results always sort above unstamped ones.
function resultRecency(variant: ResultVariant): number {
  if (variant.completed_at != null) return variant.completed_at * 1000;
  for (const text of [variant.job_name, variant.experiment, variant.source]) {
    const m = text?.match(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
    if (m) {
      const ts = Date.parse(`${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}`);
      if (!Number.isNaN(ts)) return ts;
    }
  }
  const id = Number(variant.job_id);
  return Number.isFinite(id) ? id : 0;
}

function formatCompletedAt(seconds: number): string {
  return new Intl.DateTimeFormat(undefined, {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(seconds * 1000));
}

function formatGroupRecency(ts: number): string | null {
  if (ts < 1e12) return null; // job-id fallback, not a real timestamp
  return new Intl.DateTimeFormat(undefined, {
    timeZone: "Asia/Seoul",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(ts));
}

function ExperimentGroup({
  group,
  expanded,
  onToggle,
}: {
  group: ResultGroup;
  expanded: boolean;
  onToggle: () => void;
}) {
  const clusters = [...new Set(group.members.map((m) => m.cluster))];
  const modelVersions = [
    ...new Set(group.members.map((m) => m.model_version).filter((v): v is string => !!v)),
  ];
  const kinds = [...new Set(group.members.map((m) => taskKind(m.tasks)))];
  const latest = formatGroupRecency(group.latest);

  // Per-state counts across this group's results ("CANCELLED by N" → "CANCELLED"
  // so they aggregate). Results whose eval job is unknown carry no state badge;
  // they remain covered by the total result count.
  const stateCounts = new Map<string, number>();
  for (const m of group.members) {
    const state = primaryJobState(m.job_state);
    if (!state) continue;
    stateCounts.set(state, (stateCounts.get(state) ?? 0) + 1);
  }
  const states = [...stateCounts.entries()].sort(
    (a, b) =>
      Number(isActiveJobState(b[0])) - Number(isActiveJobState(a[0])) ||
      a[0].localeCompare(b[0]),
  );

  return (
    <section className="overflow-hidden rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors hover:bg-slate-50 dark:hover:bg-slate-900"
      >
        <span className="flex min-w-0 items-center gap-2">
          {expanded ? (
            <ChevronDown className="h-4 w-4 shrink-0 text-slate-400" />
          ) : (
            <ChevronRight className="h-4 w-4 shrink-0 text-slate-400" />
          )}
          <span className="truncate font-mono text-sm font-medium">{group.name}</span>
          {states.map(([state, count]) => (
            <span key={state} className="inline-flex shrink-0 items-center gap-0.5">
              <JobStateBadge state={state} />
              {count > 1 && <span className="text-[10px] text-slate-500">×{count}</span>}
            </span>
          ))}
        </span>
        <span className="flex shrink-0 flex-wrap items-center justify-end gap-2">
          {modelVersions.map((mv) => (
            <Badge key={mv} variant="secondary">{mv}</Badge>
          ))}
          {kinds.map((kind) => (
            <Badge key={kind} variant={kind === "multitask" ? "default" : "outline"}>
              {kind}
            </Badge>
          ))}
          {clusters.map((c) => (
            <Badge key={c} variant="outline">{c}</Badge>
          ))}
          <span className="text-xs text-slate-500">
            {group.members.length} result{group.members.length === 1 ? "" : "s"}
          </span>
          {latest && <span className="text-xs text-slate-400">latest {latest}</span>}
        </span>
      </button>
      {expanded && (
        <div className="divide-y divide-slate-200 border-t border-slate-200 dark:divide-slate-800 dark:border-slate-800">
          {group.members.map((variant) => (
            <ResultCard
              key={resultVariantKey(variant)}
              variant={variant}
              className="rounded-none border-0 shadow-none"
            />
          ))}
        </div>
      )}
    </section>
  );
}

function ResultCard({ variant, className }: { variant: ResultVariant; className?: string }) {
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
    <Card className={className}>
      <CardHeader className="gap-3 sm:flex-row sm:items-start sm:justify-between sm:space-y-0">
        <div className="min-w-0">
          <CardTitle className="font-mono text-base">
            <ResultTitle variant={variant} />
          </CardTitle>
          {variant.completed_at != null && (
            <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              completed {formatCompletedAt(variant.completed_at)}
            </div>
          )}
          <CardDescription className="mt-1">
            {variant.note || "eval results"}
          </CardDescription>
        </div>
        <div className="flex shrink-0 flex-wrap justify-end gap-2">
          <CopyResultTableButton variant={variant} evalSets={evalSets} />
          <Badge variant="outline">{variant.cluster}</Badge>
          {variant.model_version && <Badge variant="secondary">{variant.model_version}</Badge>}
          <Badge variant={taskKind(variant.tasks) === "multitask" ? "default" : "outline"}>
            {taskKind(variant.tasks)}
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
              value={withId(checkpointLabel, variant.checkpoint_job_id)}
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
      {variant.job_state && (
        <JobStateBadge state={primaryJobState(variant.job_state)} />
      )}
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
  const html = resultTableHtml(variant, evalSets);
  await copyRich(html, tsv);
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

function checkpointDisplayName(path: string) {
  const parts = path.split("/").filter(Boolean);
  const leaf = basename(path);
  if (!leaf.startsWith("checkpoint-")) return leaf;
  return parts[parts.length - 2] ?? leaf;
}

/** "name (id)" when an id is present, else just the name. */
function withId(name: string, id?: string | null): string {
  return id ? `${name} (${id})` : name;
}

/** Single- vs multi-task classification from a result's task list. */
function taskKind(tasks: ResultTask[]): "multitask" | "single" {
  return tasks.length > 1 ? "multitask" : "single";
}

function resultTableTitle(variant: ResultVariant) {
  return withId(variant.variant, variant.job_id);
}

function resultTableTsv(variant: ResultVariant, evalSets: string[]) {
  const table = resultTableRows(variant, evalSets).map((row) => row.map(tsvCell).join("\t")).join("\n");
  return `${tsvCell(resultTableTitle(variant))}\n${table}`;
}

function resultTableHtml(variant: ResultVariant, evalSets: string[]) {
  const [header, ...body] = resultTableRows(variant, evalSets);
  return [
    `<p><strong>${htmlCell(resultTableTitle(variant))}</strong></p>`,
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

