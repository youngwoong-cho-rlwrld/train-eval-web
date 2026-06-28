"use client";

import { useMemo, useRef, useState } from "react";
import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import {
  api,
  type Variant,
  type SubmitResponse,
  type SubmitConfigPreview,
  type DataInterfaceSummary,
  type Partition,
  type Dataset,
  type DexjocoTasks,
  type MlxpNode,
  type MlxpSettings,
  type GitStatus,
  type GitCommitOption,
  type PathExistence,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ChevronRight, Loader2 } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useMyMlxpNode } from "@/hooks/use-my-mlxp-node";
import { DatasetField } from "@/components/submit/dataset-field";
import { MlxpCard } from "@/components/submit/mlxp-card";
import { AvailabilityCard } from "@/components/submit/availability-card";
import {
  ConfigCard,
  DataInterfaceCard,
  type ExtraFlagRow,
  type FlagEditor,
} from "@/components/config-card";
import { useDatasetDir } from "@/hooks/use-dataset-dir";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";
import { evalHarness, resolveModel } from "@/lib/variant-model";

type Phase = "train" | "eval";
type EvalConfigEdit = {
  scope: string;
  nEpisodes: string;
  nRuns: string;
  evalSets: string;
  dexjocoTask: string;
};
type TrainConfigEdit = {
  scope: string;
  numGpus: string;
  batchSize: string;
  maxSteps: string;
  saveSteps: string;
  numWorkers: string;
  actionHorizon: string;
  gitCommit: string;
};
type TrainConfigValues = Omit<TrainConfigEdit, "scope">;
type TrainNoteEdit = {
  scope: string;
  value: string;
};
type SubmitStep = "job" | "config" | "modality";
const TRAIN_CONFIG_FIELDS: readonly (keyof TrainConfigValues)[] = [
  "numGpus",
  "batchSize",
  "maxSteps",
  "saveSteps",
  "numWorkers",
  "actionHorizon",
  "gitCommit",
] as const;

function buildDefaultJobName(phase: Phase, variant: string): string {
  if (!variant) return "";
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const ts =
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return `${phase}_${variant}_${ts}`;
}

function ellipsize(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, Math.max(0, maxLength - 3))}...`;
}

function makeTrainConfigEdit(
  scope: string,
  defaults: TrainConfigValues,
  patch: Partial<TrainConfigValues> = {},
): TrainConfigEdit {
  return { scope, ...defaults, ...patch };
}

function trainConfigHasChanges(
  edit: TrainConfigEdit | null,
  defaults: TrainConfigValues,
): boolean {
  return (
    edit !== null &&
    TRAIN_CONFIG_FIELDS.some((field) => edit[field] !== defaults[field])
  );
}

function trainConfigFieldChanged(
  edit: TrainConfigEdit | null,
  defaults: TrainConfigValues,
  field: keyof TrainConfigValues,
): boolean {
  return edit !== null && edit[field] !== defaults[field];
}

function submittedGitCommitValue(
  configEditActive: boolean,
  gitCommit: string,
): string | null {
  const trimmed = gitCommit.trim();
  return configEditActive ? trimmed : trimmed || null;
}

function submitGitQueryParams(
  cluster: string,
  variant: string,
  gitCommit: string,
  includeCommit: boolean,
): URLSearchParams {
  const qs = new URLSearchParams({ cluster, variant });
  if (includeCommit) qs.set("commit", gitCommit.trim());
  return qs;
}

export default function SubmitPage() {
  const router = useRouter();
  const qc = useQueryClient();

  const [cluster, setCluster] = useState<string>("kakao");
  const [variantName, setVariantName] = useState<string>("");
  const [variantFilter, setVariantFilter] = useState("");
  const variantFilterRef = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<Phase>("train");
  const [partition, setPartition] = useState<string>("");
  const mlxpSettings = useQuery({
    queryKey: ["mlxp-settings"],
    queryFn: () => api<MlxpSettings>("/api/mlxp/settings"),
  });
  // Persisted across sessions + synced across pages via useMyMlxpNode.
  const [mlxpNode, setMlxpNode] = useMyMlxpNode(mlxpSettings.data?.default_node ?? "");
  const [mlxpJobClass, setMlxpJobClass] = useState<"dedicated" | "normal" | "background">("normal");
  const [extraArgs, setExtraArgs] = useState<string>("");
  const [evalOverwriteResults, setEvalOverwriteResults] = useState<boolean>(false);
  const [evalConfigEdit, setEvalConfigEdit] = useState<EvalConfigEdit | null>(null);
  const [trainConfigEdit, setTrainConfigEdit] = useState<TrainConfigEdit | null>(null);
  const [checkpointEdit, setCheckpointEdit] = useState<{
    scope: string;
    value: string;
  } | null>(null);
  const [trainNoteEdit, setTrainNoteEdit] = useState<TrainNoteEdit | null>(null);
  const [jobName, setJobName] = useState<string>("");
  const [jobNameTouched, setJobNameTouched] = useState<boolean>(false);
  const [gitDialogOpen, setGitDialogOpen] = useState<boolean>(false);
  const [dirtyGitStatus, setDirtyGitStatus] = useState<GitStatus | null>(null);
  const [preflightPending, setPreflightPending] = useState<boolean>(false);
  const [submitStep, setSubmitStep] = useState<SubmitStep>("job");
  const [datasetDialogOpen, setDatasetDialogOpen] = useState<boolean>(false);
  const [gitCommitDialogOpen, setGitCommitDialogOpen] = useState<boolean>(false);

  const changeCluster = (next: string) => {
    setCluster(next);
    setSubmitStep("job");
  };
  const changeVariantName = (next: string) => {
    setVariantName(next);
    setSubmitStep("job");
  };
  const changePhase = (next: Phase) => {
    setPhase(next);
    setSubmitStep("job");
  };

  // Dataset override state. For single-task variants, `singleDataset` holds
  // the chosen name. For multi-task, `multiDatasets` holds the array of
  // "name|cfg|weight" strings (N1.5) or plain "name" strings (N1.6). Both
  // are initialized from the variant's own config when it loads; user can
  // edit before submit.
  const [datasetEdit, setDatasetEdit] = useState<{
    variant: string;
    single: string;
    multi: string[];
  } | null>(null);

  const clusters = useQuery({
    queryKey: ["clusters"],
    queryFn: () =>
      api<{ clusters: string[] }>("/api/clusters").then((d) => d.clusters),
  });
  const variantNames = useQuery({
    queryKey: ["variants"],
    queryFn: () =>
      api<{ variants: string[] }>("/api/variants").then((d) => d.variants),
  });
  // Substring filter for the experiment dropdown. The selected experiment
  // always stays in the list so the trigger label keeps rendering while open.
  const variantFilterNeedle = variantFilter.trim().toLowerCase();
  const filteredVariantNames = (variantNames.data ?? []).filter(
    (v) => v === variantName || v.toLowerCase().includes(variantFilterNeedle),
  );
  const variant = useQuery({
    queryKey: ["variant", variantName],
    queryFn: () =>
      variantName ? api<Variant>(`/api/variants/${variantName}`) : null,
    enabled: !!variantName,
  });
  const dataInterface = useQuery({
    queryKey: ["variant-data-interface", variantName],
    queryFn: () =>
      api<DataInterfaceSummary>(`/api/variants/${variantName}/data-interface`),
    enabled: !!variantName,
  });
  const partitions = useQuery({
    queryKey: ["partitions", cluster],
    queryFn: () => api<Partition[]>(`/api/clusters/${cluster}/partitions`),
    refetchInterval: 30_000,
    enabled: !!cluster && cluster !== "mlxp",
  });
  const isSlurm = cluster !== "mlxp";
  const checkpointScope = `${cluster}:${variantName}:${phase}`;
  const trainNoteScope = `${checkpointScope}:train-note`;
  const evalConfigScope = `${checkpointScope}:eval-config`;
  const trainConfigScope = `${checkpointScope}:train-config`;

  const defaultDatasetDir = cluster === "mlxp" ? mlxpSettings.data?.datasets_dir : undefined;
  const [datasetDir, setDatasetDir] = useDatasetDir(cluster, defaultDatasetDir);
  const datasets = useQuery({
    queryKey: ["datasets", cluster, datasetDir],
    queryFn: () =>
      api<Dataset[]>(
        `/api/clusters/${cluster}/datasets?path=${encodeURIComponent(datasetDir)}`,
      ),
    enabled: !!cluster,
    retry: false,
  });
  const mlxp = useQuery({
    queryKey: ["mlxp-gpus"],
    queryFn: () => api<MlxpNode[]>("/api/mlxp/gpus"),
    refetchInterval: 60_000,
    retry: false,
    enabled: !isSlurm,
  });
  const wantsCheckpoint = phase === "eval" && !!variantName;
  const isDexjoco = evalHarness(variant.data?.vars) === "dexjoco";
  const dexjocoTasks = useQuery({
    queryKey: ["dexjoco-tasks", cluster],
    queryFn: () =>
      api<DexjocoTasks>(
        `/api/dexjoco/tasks?cluster=${encodeURIComponent(cluster)}`,
      ),
    enabled: isDexjoco && !!cluster,
    retry: false,
  });
  const activeCheckpointEdit =
    checkpointEdit?.scope === checkpointScope ? checkpointEdit : null;
  const checkpointPath = activeCheckpointEdit ? activeCheckpointEdit.value : "";
  const trimmedCkpt = checkpointPath.trim();
  const checkpointExists = useQuery({
    queryKey: ["path-exists", cluster, trimmedCkpt],
    queryFn: () =>
      api<PathExistence>(
        `/api/clusters/${cluster}/path-exists?path=${encodeURIComponent(trimmedCkpt)}`,
      ),
    enabled: wantsCheckpoint && !!trimmedCkpt,
  });
  const checkpointExistsValue: boolean | null = !trimmedCkpt
    ? null
    : checkpointExists.isLoading || checkpointExists.data === undefined
      ? null
      : checkpointExists.data.exists && checkpointExists.data.kind === "dir";

  const defaultJobName = useMemo(
    () => buildDefaultJobName(phase, variantName),
    [phase, variantName],
  );
  const shownJobName = jobNameTouched ? jobName : defaultJobName;
  const trainNoteDefault = variant.data?.vars.TRAIN_NOTE ?? "";
  const activeTrainNoteEdit =
    trainNoteEdit?.scope === trainNoteScope ? trainNoteEdit : null;
  const trainNote = activeTrainNoteEdit ? activeTrainNoteEdit.value : trainNoteDefault;
  const submittedTrainNote = trainNote.trim() || null;
  const effectiveTrainNote = submittedTrainNote ?? trainNoteDefault.trim();
  const trainNoteValid = effectiveTrainNote.length > 0;
  const evalConfigDefaults = useMemo(
    () => ({
      nEpisodes: variant.data?.vars.N_EPISODES ?? "",
      nRuns: variant.data?.vars.N_RUNS ?? "",
      evalSets: formatEvalSetsInput(variant.data?.arrays.EVAL_SETS ?? []),
      dexjocoTask: variant.data?.vars.DEXJOCO_TASK ?? "",
    }),
    [variant.data],
  );
  const activeEvalConfigEdit =
    evalConfigEdit?.scope === evalConfigScope ? evalConfigEdit : null;
  const evalNEpisodes = activeEvalConfigEdit
    ? activeEvalConfigEdit.nEpisodes
    : evalConfigDefaults.nEpisodes;
  const evalNRuns = activeEvalConfigEdit
    ? activeEvalConfigEdit.nRuns
    : evalConfigDefaults.nRuns;
  const evalSetsText = activeEvalConfigEdit
    ? activeEvalConfigEdit.evalSets
    : evalConfigDefaults.evalSets;
  const dexjocoTask = activeEvalConfigEdit
    ? activeEvalConfigEdit.dexjocoTask
    : evalConfigDefaults.dexjocoTask;
  const dexjocoTaskValid =
    !(wantsCheckpoint && isDexjoco) || dexjocoTask.trim().length > 0;
  const evalSetValues = parseEvalSetsInput(evalSetsText);
  const evalSetOptions = variant.data?.arrays.EVAL_SETS ?? [];
  const evalNEpisodesTrimmed = evalNEpisodes.trim();
  const evalNRunsTrimmed = evalNRuns.trim();
  const evalNEpisodesParsed = Number.parseInt(evalNEpisodesTrimmed, 10);
  const evalNRunsParsed = Number.parseInt(evalNRunsTrimmed, 10);
  const evalNEpisodesValid =
    !wantsCheckpoint || /^[1-9]\d*$/.test(evalNEpisodesTrimmed);
  const evalNRunsValid =
    !wantsCheckpoint || /^[1-9]\d*$/.test(evalNRunsTrimmed);
  const evalSetsCharactersValid = evalSetValues.every((v) =>
    /^[A-Za-z0-9_.-]+$/.test(v),
  );
  const evalSetsValid =
    !wantsCheckpoint || (evalSetValues.length > 0 && evalSetsCharactersValid);
  const evalTotalRuns =
    wantsCheckpoint && evalNEpisodesValid && evalNRunsValid && evalSetsValid
      ? evalNRunsParsed * evalSetValues.length
      : null;
  const evalTotalEpisodes =
    evalTotalRuns !== null ? evalTotalRuns * evalNEpisodesParsed : null;
  const updateEvalConfig = (patch: Partial<Omit<EvalConfigEdit, "scope">>) => {
    const base = activeEvalConfigEdit ?? {
      scope: evalConfigScope,
      nEpisodes: evalConfigDefaults.nEpisodes,
      nRuns: evalConfigDefaults.nRuns,
      evalSets: evalConfigDefaults.evalSets,
      dexjocoTask: evalConfigDefaults.dexjocoTask,
    };
    setEvalConfigEdit({ ...base, ...patch, scope: evalConfigScope });
  };
  const trainConfigDefaults = useMemo<TrainConfigValues>(() => {
    const vars = variant.data?.vars;
    const numGpus = vars?.TRAIN_NUM_GPUS ?? "2";
    const { model } = resolveModel(vars);
    const perGpuBatch = vars?.TRAIN_BATCH_SIZE ?? "";
    const globalBatch =
      vars?.TRAIN_GLOBAL_BATCH_SIZE ?? vars?.GLOBAL_BATCH_SIZE ?? "";
    let batchSize =
      model === "n1.5"
        ? perGpuBatch
        : globalBatch;
    const parsedNumGpus = Number.parseInt(numGpus, 10);
    const parsedPerGpuBatch = Number.parseInt(perGpuBatch, 10);
    const parsedGlobalBatch = Number.parseInt(globalBatch, 10);
    if (!batchSize && parsedNumGpus > 0 && parsedPerGpuBatch > 0) {
      batchSize =
        model === "n1.5"
          ? String(parsedPerGpuBatch)
          : String(parsedNumGpus * parsedPerGpuBatch);
    }
    if (
      !batchSize &&
      model === "n1.5" &&
      parsedNumGpus > 0 &&
      parsedGlobalBatch > 0 &&
      parsedGlobalBatch % parsedNumGpus === 0
    ) {
      batchSize = String(parsedGlobalBatch / parsedNumGpus);
    }
    return {
      numGpus,
      batchSize,
      maxSteps: vars?.MAX_STEPS ?? "",
      saveSteps: vars?.SAVE_STEPS ?? "",
      numWorkers: vars?.TRAIN_NUM_WORKERS ?? "16",
      actionHorizon:
        vars?.TRAIN_ACTION_HORIZON ??
        (dataInterface.data?.action_horizon != null
          ? String(dataInterface.data.action_horizon)
          : ""),
      gitCommit: vars?.TRAIN_GIT_COMMIT ?? "",
    };
  }, [variant.data, dataInterface.data?.action_horizon]);
  const activeTrainConfigEdit =
    trainConfigEdit?.scope === trainConfigScope ? trainConfigEdit : null;
  const trainNumGpus = activeTrainConfigEdit
    ? activeTrainConfigEdit.numGpus
    : trainConfigDefaults.numGpus;
  const trainBatchSize = activeTrainConfigEdit
    ? activeTrainConfigEdit.batchSize
    : trainConfigDefaults.batchSize;
  const trainMaxSteps = activeTrainConfigEdit
    ? activeTrainConfigEdit.maxSteps
    : trainConfigDefaults.maxSteps;
  const trainSaveSteps = activeTrainConfigEdit
    ? activeTrainConfigEdit.saveSteps
    : trainConfigDefaults.saveSteps;
  const trainNumWorkers = activeTrainConfigEdit
    ? activeTrainConfigEdit.numWorkers
    : trainConfigDefaults.numWorkers;
  const trainActionHorizon = activeTrainConfigEdit
    ? activeTrainConfigEdit.actionHorizon
    : trainConfigDefaults.actionHorizon;
  const trainGitCommit = activeTrainConfigEdit
    ? activeTrainConfigEdit.gitCommit
    : trainConfigDefaults.gitCommit;
  const trainGitCommitTrimmed = trainGitCommit.trim();
  const trainNumGpusParsed = Number.parseInt(trainNumGpus.trim(), 10);
  const trainBatchSizeParsed = Number.parseInt(
    trainBatchSize.trim(),
    10,
  );
  const trainMaxStepsParsed = Number.parseInt(trainMaxSteps.trim(), 10);
  const trainSaveStepsParsed = Number.parseInt(trainSaveSteps.trim(), 10);
  const trainNumWorkersParsed = Number.parseInt(trainNumWorkers.trim(), 10);
  const trainActionHorizonParsed = Number.parseInt(
    trainActionHorizon.trim(),
    10,
  );
  const isPositiveInteger = (value: string) => /^[1-9]\d*$/.test(value.trim());
  const { model: trainModel } = resolveModel(variant.data?.vars);
  const wantsTrainConfig = phase === "train" && !!variantName;
  const wantsGpuConfig = !!variantName;
  const wantsGitCommitConfig = !!variantName;
  const trainActionHorizonEnabled = wantsTrainConfig && trainModel === "n1.6";
  const modalityActionHorizon = dataInterface.data?.action_horizon ?? null;
  const trainNumGpusValid =
    !wantsGpuConfig ||
    (isPositiveInteger(trainNumGpus) &&
      (isSlurm || [1, 2, 4, 8].includes(trainNumGpusParsed)));
  const trainBatchSizeValid =
    !wantsTrainConfig || isPositiveInteger(trainBatchSize);
  const trainMaxStepsValid =
    !wantsTrainConfig || isPositiveInteger(trainMaxSteps);
  const trainSaveStepsValid =
    !wantsTrainConfig || isPositiveInteger(trainSaveSteps);
  const trainNumWorkersValid =
    !wantsTrainConfig || isPositiveInteger(trainNumWorkers);
  const trainActionHorizonValid =
    !trainActionHorizonEnabled ||
    isPositiveInteger(trainActionHorizon);
  const trainGitCommitValid =
    !wantsGitCommitConfig ||
    !trainGitCommitTrimmed ||
    /^[0-9a-fA-F]{7,40}$/.test(trainGitCommitTrimmed);
  const trainConfigValid =
    trainNumGpusValid &&
    trainBatchSizeValid &&
    trainMaxStepsValid &&
    trainSaveStepsValid &&
    trainNumWorkersValid &&
    trainActionHorizonValid &&
    trainGitCommitValid;
  const updateTrainConfig = (
    patch: Partial<Omit<TrainConfigEdit, "scope">>,
  ) => {
    const base =
      activeTrainConfigEdit ??
      makeTrainConfigEdit(trainConfigScope, trainConfigDefaults);
    setTrainConfigEdit({ ...base, ...patch, scope: trainConfigScope });
  };

  const datasetDefaults = useMemo(() => {
    if (!variant.data) return { single: "", multi: [] as string[] };
    // Priority: N1.6 multi (TRAIN_DATASET_NAMES, name-only) → N1.5 multi
    // (DATASETS, "name|cfg|weight") → single (DATASET_NAME).
    if (variant.data.arrays.TRAIN_DATASET_NAMES) {
      return { single: "", multi: variant.data.arrays.TRAIN_DATASET_NAMES };
    }
    if (variant.data.arrays.DATASETS) {
      return { single: "", multi: variant.data.arrays.DATASETS };
    }
    if (variant.data.vars.DATASET_NAME) {
      return { single: variant.data.vars.DATASET_NAME, multi: [] as string[] };
    }
    return { single: "", multi: [] as string[] };
  }, [variant.data]);

  const activeDatasetEdit =
    datasetEdit?.variant === variantName ? datasetEdit : null;
  const datasetTouched = activeDatasetEdit !== null;
  const singleDataset = activeDatasetEdit
    ? activeDatasetEdit.single
    : datasetDefaults.single;
  const multiDatasets = activeDatasetEdit
    ? activeDatasetEdit.multi
    : datasetDefaults.multi;

  const selectedPartitionName = useMemo(() => {
    if (!partitions.data) return "";
    if (partition && partitions.data.some((p) => p.name === partition)) return partition;
    return (partitions.data.find((p) => p.is_default) ?? partitions.data[0])?.name ?? "";
  }, [partitions.data, partition]);
  const selectedPartition = partitions.data?.find((p) => p.name === selectedPartitionName);
  const variantError = variant.error as Error | null;
  const submittedTrainGlobalBatchSize =
    phase === "train"
      ? trainModel === "n1.5"
        ? trainBatchSizeParsed * trainNumGpusParsed
        : trainBatchSizeParsed
      : null;
  const submittedTrainActionHorizon =
    trainActionHorizonEnabled ? trainActionHorizonParsed : null;
  const trainConfigEditActive = activeTrainConfigEdit !== null;
  const trainGitCommitChanged = trainConfigFieldChanged(
    activeTrainConfigEdit,
    trainConfigDefaults,
    "gitCommit",
  );
  const trainOverridesChanged = trainConfigHasChanges(
    activeTrainConfigEdit,
    trainConfigDefaults,
  );
  const includeGitCommitParam = trainConfigEditActive || !!trainGitCommitTrimmed;
  const submittedGitCommit = submittedGitCommitValue(
    trainConfigEditActive,
    trainGitCommit,
  );

  const buildSubmitBody = (commitDirtyChanges: boolean) => {
    return {
      cluster,
      variant: variantName,
      phase: phase,
      train_note: submittedTrainNote,
      partition: isSlurm ? selectedPartitionName : null,
      node: isSlurm || mlxpJobClass !== "dedicated" ? null : mlxpNode,
      job_class: isSlurm ? null : mlxpJobClass,
      dataset_override: resolveDatasetOverride({
        touched: datasetTouched,
        variant: variant.data,
        singleDataset,
        multiDatasets,
      }),
      extra_args: phase === "eval" && !isSlurm ? [] : splitArgs(extraArgs),
      train_num_gpus: wantsGpuConfig ? trainNumGpusParsed : null,
      train_global_batch_size: submittedTrainGlobalBatchSize,
      train_max_steps: phase === "train" ? trainMaxStepsParsed : null,
      train_save_steps: phase === "train" ? trainSaveStepsParsed : null,
      train_num_workers: phase === "train" ? trainNumWorkersParsed : null,
      train_action_horizon: submittedTrainActionHorizon,
      train_git_commit: submittedGitCommit,
      eval_num_envs_per_gpu: null,
      eval_n_episodes: wantsCheckpoint ? evalNEpisodesParsed : null,
      eval_n_runs: wantsCheckpoint ? evalNRunsParsed : null,
      eval_sets: wantsCheckpoint ? evalSetValues : null,
      eval_overwrite_results: wantsCheckpoint ? evalOverwriteResults : false,
      dexjoco_task:
        wantsCheckpoint && isDexjoco ? dexjocoTask.trim() || null : null,
      checkpoint_path: wantsCheckpoint ? trimmedCkpt : null,
      job_name: shownJobName.trim() || null,
      commit_dirty_changes: commitDirtyChanges,
    };
  };

  const configPreviewEnabled =
    !!variantName &&
    !variant.isLoading &&
    !variantError &&
    trainNoteValid &&
    trainConfigValid &&
    (!wantsCheckpoint ||
      (!!trimmedCkpt &&
        checkpointExistsValue === true &&
        evalNEpisodesValid &&
        evalNRunsValid &&
        evalSetsValid &&
        dexjocoTaskValid));
  const configPreview = useQuery({
    queryKey: [
      "submit-config-preview",
      cluster,
      variantName,
      phase,
      selectedPartitionName,
      mlxpNode,
      datasetTouched,
      singleDataset,
      multiDatasets,
      extraArgs,
      trainNote,
      trainNumGpus,
      trainBatchSize,
      trainMaxSteps,
      trainSaveSteps,
      trainNumWorkers,
      trainActionHorizon,
      submittedGitCommit,
      evalNEpisodes,
      evalNRuns,
      evalSetValues,
      dexjocoTask,
      evalOverwriteResults,
      checkpointPath,
      shownJobName,
    ],
    queryFn: () =>
      api<SubmitConfigPreview>("/api/submit/config-preview", {
        method: "POST",
        body: JSON.stringify(buildSubmitBody(false)),
      }),
    enabled: configPreviewEnabled,
    placeholderData: keepPreviousData,
    retry: false,
  });
  // Only surface the preview when the query is actually applicable to the
  // current inputs. `placeholderData: keepPreviousData` otherwise keeps the
  // previous phase's preview around while the eval preview is disabled (it
  // requires a checkpoint), which made the flag table show stale *train* flags
  // until a checkpoint was entered. Falling back to null lets ConfigCard render
  // the live per-phase flags, so the flag set stays stable before/after the
  // checkpoint is set.
  const displayedConfigPreview = configPreviewEnabled
    ? (configPreview.data ?? null)
    : null;
  const jobGitStatus = useQuery({
    queryKey: ["submit-git-status", cluster, variantName, submittedGitCommit],
    queryFn: () => {
      const qs = submitGitQueryParams(
        cluster,
        variantName,
        trainGitCommit,
        includeGitCommitParam,
      );
      return api<GitStatus>(`/api/submit/git-status?${qs}`);
    },
    enabled:
      !!variantName &&
      !variant.isLoading &&
      !variantError &&
      configPreview.isSuccess &&
      !configPreview.data.model_repo_error,
    retry: false,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    placeholderData: keepPreviousData,
  });
  const gitCommitOptions = useQuery({
    queryKey: [
      "submit-git-commits",
      cluster,
      variantName,
      trainGitCommitValid ? trainGitCommitTrimmed : "",
    ],
    queryFn: () => {
      const qs = new URLSearchParams({
        cluster,
        variant: variantName,
        limit: "25",
      });
      if (trainGitCommitValid && trainGitCommitTrimmed) {
        qs.set("selected", trainGitCommitTrimmed);
      }
      return api<{ commits: GitCommitOption[] }>(`/api/submit/git-commits?${qs}`);
    },
    enabled:
      gitCommitDialogOpen &&
      !!variantName &&
      !variant.isLoading &&
      !variantError,
    retry: false,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
  const jobGitStatusFetchError = jobGitStatus.error as Error | null;
  const modelRepoError =
    displayedConfigPreview?.model_repo_error ?? jobGitStatus.data?.error ?? null;
  const modelRepoMessage =
    jobGitStatusFetchError ? "Git status will be checked again when submitting." : null;
  const selectedGitCommitValue =
    trainGitCommitValid
      ? trainGitCommitTrimmed || jobGitStatus.data?.commit || undefined
      : undefined;

  const submit = useMutation({
    mutationFn: ({ commitDirtyChanges = false }: { commitDirtyChanges?: boolean } = {}) => {
      return api<SubmitResponse>("/api/submit", {
        method: "POST",
        body: JSON.stringify(buildSubmitBody(commitDirtyChanges)),
      });
    },
    onSuccess: (data) => {
      toast.success(`Submitted job ${data.job_id} on ${cluster}`);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      router.push(`/jobs/${cluster}/${data.job_id}`);
    },
    onError: (err: Error) => toast.error(`Submit failed: ${err.message}`),
  });

  async function handleSubmitClick() {
    if (phase !== "train") {
      submit.mutate({ commitDirtyChanges: false });
      return;
    }
    setPreflightPending(true);
    try {
      const qs = submitGitQueryParams(
        cluster,
        variantName,
        trainGitCommit,
        includeGitCommitParam,
      );
      const status = await api<GitStatus>(`/api/submit/git-status?${qs}`);
      if (status.error) {
        toast.error(`Git status failed: ${status.error}`);
        return;
      }
      if (status.dirty && !trainGitCommitTrimmed) {
        setDirtyGitStatus(status);
        setGitDialogOpen(true);
        return;
      }
      submit.mutate({ commitDirtyChanges: false });
    } catch (err) {
      toast.error(`Git preflight failed: ${(err as Error).message}`);
    } finally {
      setPreflightPending(false);
    }
  }

  const canSubmit =
    !!variantName &&
    trainNoteValid &&
    (isSlurm ? !!selectedPartitionName : mlxpJobClass !== "dedicated" || !!mlxpNode) &&
    (!wantsCheckpoint ||
      (!!trimmedCkpt &&
        checkpointExistsValue !== false &&
        evalNEpisodesValid &&
        evalNRunsValid &&
        evalSetsValid &&
        dexjocoTaskValid)) &&
    trainConfigValid &&
    !modelRepoError &&
    !configPreview.error &&
    !preflightPending &&
    !submit.isPending;
  const datasetField =
    variantName && variant.isLoading ? (
      <Field label="Dataset override">
        <LoadingState label="Loading dataset defaults..." rows={3} />
      </Field>
    ) : variantName && variantError ? (
      <Field label="Dataset override">
        <ErrorState message={variantError.message} />
      </Field>
    ) : variant.data ? (
      <DatasetField
        variant={variant.data}
        datasets={datasets.data ?? []}
        single={singleDataset}
        multi={multiDatasets}
        onSingleChange={(v) => {
          setDatasetEdit({ variant: variantName, single: v, multi: multiDatasets });
        }}
        onMultiChange={(v) => {
          setDatasetEdit({ variant: variantName, single: singleDataset, multi: v });
        }}
        touched={datasetTouched}
        cluster={cluster}
        datasetDir={datasetDir}
        onDatasetDirChange={setDatasetDir}
        datasetsError={datasets.error as Error | null}
      />
    ) : null;

  const datasetEditor: FlagEditor | undefined = datasetField
    ? (
        <Button
          variant={datasetTouched ? "default" : "outline"}
          size="sm"
          onClick={() => setDatasetDialogOpen(true)}
        >
          {datasetTouched ? "Edit override" : "Override datasets"}
        </Button>
      )
    : undefined;
  const evalSetsEditor: FlagEditor | undefined = wantsCheckpoint
    ? {
        wide: true,
        content: (
          <div className="space-y-3">
            {evalSetOptions.length > 0 && (
              <div className="grid gap-2 sm:grid-cols-3">
                {evalSetOptions.map((evalSet) => {
                  const checked = evalSetValues.includes(evalSet);
                  return (
                    <label
                      key={evalSet}
                      className="flex h-9 items-center justify-between gap-3 rounded border border-slate-200 px-2 text-xs dark:border-slate-800"
                    >
                      <span className="font-mono">{evalSet}</span>
                      <Switch
                        checked={checked}
                        onCheckedChange={(nextChecked) => {
                          const next = nextChecked
                            ? [...evalSetValues, evalSet]
                            : evalSetValues.filter((v) => v !== evalSet);
                          updateEvalConfig({
                            evalSets: formatEvalSetsInput(next),
                          });
                        }}
                      />
                    </label>
                  );
                })}
              </div>
            )}
            <Input
              value={evalSetsText}
              onChange={(e) =>
                updateEvalConfig({ evalSets: e.target.value })
              }
              placeholder="0cm 1cm 3cm 5cm 7cm"
              className="font-mono text-xs"
            />
            {!evalSetsValid && (
              <p className="text-xs text-red-600 dark:text-red-400">
                Choose at least one eval set. Use letters, numbers, dot,
                underscore, or hyphen.
              </p>
            )}
            {evalTotalRuns !== null && evalTotalEpisodes !== null && (
              <p className="text-xs text-slate-500">
                Total: <code>{evalTotalRuns}</code> runs ·{" "}
                <code>{evalTotalEpisodes}</code> episodes
              </p>
            )}
            {activeEvalConfigEdit && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setEvalConfigEdit(null)}
              >
                Reset eval defaults
              </Button>
            )}
          </div>
        ),
      }
    : undefined;
  const checkpointEditor: FlagEditor | undefined = wantsCheckpoint
    ? {
        wide: true,
        content: (
          <div className="space-y-2">
            <Input
              value={checkpointPath}
              onChange={(e) => {
                setCheckpointEdit({
                  scope: checkpointScope,
                  value: e.target.value,
                });
              }}
              placeholder="/absolute/path/to/checkpoint-directory"
              className={
                trimmedCkpt && checkpointExistsValue === false
                  ? "font-mono text-xs border-red-500 focus-visible:ring-red-500"
                  : "font-mono text-xs"
              }
            />
            {!trimmedCkpt && (
              <p className="text-xs text-red-600 dark:text-red-400">
                Choose a checkpoint.
              </p>
            )}
            {trimmedCkpt && checkpointExistsValue === false && (
              <p className="text-xs text-red-600 dark:text-red-400">
                Path not found as a directory on <code>{cluster}</code>.
              </p>
            )}
          </div>
        ),
      }
    : undefined;
  const dexjocoTaskOptions = dexjocoTasks.data?.tasks ?? [];
  const dexjocoTaskError = dexjocoTasks.error as Error | null;
  const dexjocoTaskEditor: FlagEditor | undefined =
    wantsCheckpoint && isDexjoco
      ? {
          wide: true,
          content: (
            <div className="space-y-2">
              <Select
                value={dexjocoTask}
                onValueChange={(value) =>
                  updateEvalConfig({ dexjocoTask: value })
                }
              >
                <SelectTrigger className="gap-2">
                  <SelectValue placeholder="select a task..." />
                  {dexjocoTasks.isFetching && (
                    <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-slate-400" />
                  )}
                </SelectTrigger>
                <SelectContent>
                  {dexjocoTaskOptions.map((task) => (
                    <SelectItem key={task} value={task}>
                      <span className="font-mono text-xs">{task}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {dexjocoTaskError && (
                <ErrorState message={dexjocoTaskError.message} />
              )}
              {!dexjocoTaskValid && (
                <p className="text-xs text-red-600 dark:text-red-400">
                  Choose a DexJoCo task.
                </p>
              )}
            </div>
          ),
        }
      : undefined;
  const flagEditors: Record<string, FlagEditor> = {};
  if (datasetEditor) {
    flagEditors["--dataset-path"] = datasetEditor;
    flagEditors["--data-config"] = datasetEditor;
  }
  const gpuCountEditor =
    wantsGpuConfig && variant.data ? (
      <NumberCellEditor
        value={trainNumGpus}
        onChange={(value) => updateTrainConfig({ numGpus: value })}
        valid={trainNumGpusValid}
        invalidMessage={isSlurm ? "Positive integer." : "Use 1, 2, 4, or 8."}
      />
    ) : undefined;
  if (gpuCountEditor) {
    flagEditors["--num-gpus"] = gpuCountEditor;
  }
  if (wantsTrainConfig && variant.data) {
    const batchEditor = (
      <NumberCellEditor
        value={trainBatchSize}
        onChange={(value) => updateTrainConfig({ batchSize: value })}
        valid={trainBatchSizeValid}
        invalidMessage="Positive integer."
      />
    );
    flagEditors["--global-batch-size"] = batchEditor;
    flagEditors["--batch-size"] = batchEditor;
    flagEditors["--max-steps"] = (
      <NumberCellEditor
        value={trainMaxSteps}
        onChange={(value) => updateTrainConfig({ maxSteps: value })}
        valid={trainMaxStepsValid}
        invalidMessage="Positive integer."
      />
    );
    flagEditors["--save-steps"] = (
      <NumberCellEditor
        value={trainSaveSteps}
        onChange={(value) => updateTrainConfig({ saveSteps: value })}
        valid={trainSaveStepsValid}
        invalidMessage="Positive integer."
      />
    );
    const workersEditor = (
      <NumberCellEditor
        value={trainNumWorkers}
        onChange={(value) => updateTrainConfig({ numWorkers: value })}
        valid={trainNumWorkersValid}
        invalidMessage="Positive integer."
      />
    );
    flagEditors["--dataloader_num_workers"] = workersEditor;
    flagEditors["--dataloader-num-workers"] = workersEditor;
  }
  if (wantsCheckpoint) {
    flagEditors["--n-episodes"] = (
      <NumberCellEditor
        value={evalNEpisodes}
        onChange={(value) => updateEvalConfig({ nEpisodes: value })}
        valid={evalNEpisodesValid}
        invalidMessage="Positive integer."
      />
    );
    flagEditors["--n-runs"] = (
      <NumberCellEditor
        value={evalNRuns}
        onChange={(value) => updateEvalConfig({ nRuns: value })}
        valid={evalNRunsValid}
        invalidMessage="Positive integer."
      />
    );
    if (evalSetsEditor) flagEditors["(eval_sets)"] = evalSetsEditor;
  }
  const extraFlagRows: ExtraFlagRow[] = [];
  if (wantsTrainConfig) {
    if (trainActionHorizonEnabled) {
      extraFlagRows.push({
        key: "train-action-horizon",
        flag: "TRAIN_ACTION_HORIZON",
        value: trainActionHorizon || "(unset)",
        editor: (
          <NumberCellEditor
            value={trainActionHorizon}
            onChange={(value) => updateTrainConfig({ actionHorizon: value })}
            valid={trainActionHorizonValid}
            invalidMessage="Positive integer."
            hint={
              modalityActionHorizon != null && trainActionHorizonParsed !== modalityActionHorizon
                ? `Will stage modality delta_indices=list(range(${trainActionHorizonParsed})).`
                : undefined
            }
          />
        ),
      });
    }
  }
  if (phase === "eval" && wantsGpuConfig) {
    extraFlagRows.push({
      key: "eval-num-gpus",
      flag: "TRAIN_NUM_GPUS",
      value: trainNumGpus || "(unset)",
      editor: gpuCountEditor,
    });
  }
  if (wantsGitCommitConfig) {
    extraFlagRows.push({
      key: "train-git-commit",
      flag: "TRAIN_GIT_COMMIT",
      value: trainGitCommitTrimmed || "(current HEAD)",
      editor: (
        <Button
          variant="outline"
          size="sm"
          onClick={() => setGitCommitDialogOpen(true)}
        >
          {trainGitCommitChanged ? "Edit override" : "Override commit"}
        </Button>
      ),
    });
  }
  if (wantsCheckpoint) {
    extraFlagRows.push({
      key: "checkpoint",
      flag: "checkpoint",
      value:
        checkpointPath.trim() ||
        "(required)",
      editor: checkpointEditor,
    });
    if (isDexjoco && dexjocoTaskEditor) {
      extraFlagRows.push({
        key: "dexjoco-task",
        flag: "DEXJOCO_TASK",
        value: dexjocoTask.trim() || "(required)",
        editor: dexjocoTaskEditor,
      });
    }
    extraFlagRows.push({
      key: "overwrite-results",
      flag: "overwrite results",
      value: evalOverwriteResults ? "true" : "false",
      editor: (
        <Switch
          checked={evalOverwriteResults}
          onCheckedChange={setEvalOverwriteResults}
        />
      ),
    });
  }
  if (phase === "train" || isSlurm) {
    extraFlagRows.push({
      key: "extra-args",
      flag: phase === "train" ? "extra train args" : "extra sbatch args",
      value: extraArgs.trim() || "(none)",
      editor: {
        wide: true,
        content: (
          <Input
            placeholder={
              phase === "train"
                ? "--state-part-mode random_balanced --state-part-token-count 7"
                : "--exclusive --nice=100"
            }
            value={extraArgs}
            onChange={(e) => setExtraArgs(e.target.value)}
          />
        ),
      },
    });
  }
  const selectedVariantName = variant.data?.name ?? variantName;
  const canReviewConfig = !!variantName && trainNoteValid;
  const canReviewModality =
    canReviewConfig &&
    (!wantsCheckpoint ||
      (!!trimmedCkpt &&
        checkpointExistsValue === true &&
        evalNEpisodesValid &&
        evalNRunsValid &&
        evalSetsValid &&
        dexjocoTaskValid));
  const selectedMlxpGpuType =
    mlxp.data?.find((n) => n.name === mlxpNode)?.gpu_type ||
    mlxpSettings.data?.gpu_type ||
    "GPU";
  const submitButtonLabel =
    submit.isPending || preflightPending
      ? "Submitting..."
      : isSlurm
        ? `Submit ${phase} -> ${cluster}/${selectedPartitionName || "?"}`
        : mlxpJobClass === "dedicated"
          ? `Submit ${phase} -> mlxp/${mlxpNode}/${trainNumGpus || "?"}x${selectedMlxpGpuType}`
          : `Submit ${phase} -> mlxp/${mlxpJobClass} queue/${trainNumGpus || "?"}x${selectedMlxpGpuType}`;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 lg:py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Submit a job</h1>
      <SubmitStepper
        activeStep={submitStep}
        configEnabled={canReviewConfig}
        modalityEnabled={canReviewModality}
        onStepChange={setSubmitStep}
      />

      <div className="mt-8 grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-6">
          {submitStep === "job" && (
          <Card>
            <CardHeader>
              <CardTitle>Job</CardTitle>
              <CardDescription>
                Choose where the job runs and how it is identified.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <Field label="Cluster">
                <Select value={cluster} onValueChange={changeCluster}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {clusters.data?.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {clusters.isLoading && <LoadingState label="Loading clusters..." rows={1} />}
                {clusters.error && <ErrorState message={(clusters.error as Error).message} />}
              </Field>

              <Field label="Experiment">
                <Select
                  value={variantName}
                  onValueChange={changeVariantName}
                  onOpenChange={(open) => {
                    if (!open) {
                      setVariantFilter("");
                      return;
                    }
                    // Radix focuses the selected item right after open; queue
                    // our focus behind that so the filter box wins.
                    setTimeout(() => variantFilterRef.current?.focus(), 0);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="select an experiment..." />
                  </SelectTrigger>
                  <SelectContent>
                    <div className="pb-1">
                      <Input
                        ref={variantFilterRef}
                        value={variantFilter}
                        onChange={(e) => setVariantFilter(e.target.value)}
                        onKeyDown={(e) => {
                          // Type in the box instead of triggering the Select's
                          // item typeahead; still let Escape close the menu.
                          if (e.key !== "Escape") e.stopPropagation();
                        }}
                        placeholder="filter experiments..."
                        className="h-8 border-0 font-mono text-xs shadow-none focus-visible:ring-0"
                      />
                    </div>
                    {filteredVariantNames.map((v) => (
                      <SelectItem key={v} value={v}>
                        {v}
                      </SelectItem>
                    ))}
                    {filteredVariantNames.length === 0 && (
                      <div className="px-3 py-2 text-sm text-slate-500">
                        no experiments match
                      </div>
                    )}
                  </SelectContent>
                </Select>
                {variantNames.isLoading && <LoadingState label="Loading experiments..." rows={1} />}
                {variantNames.error && <ErrorState message={(variantNames.error as Error).message} />}
              </Field>

              <PhaseField value={phase} onChange={changePhase} />

              {isSlurm ? (
                <Field label="Partition">
                  <Select value={selectedPartitionName} onValueChange={setPartition}>
                    <SelectTrigger>
                      <SelectValue placeholder="loading partitions…" />
                    </SelectTrigger>
                    <SelectContent>
                      {partitions.data?.map((p) => (
                        <SelectItem key={p.name} value={p.name}>
                          <span className="flex items-center gap-2">
                            <span>{p.name}</span>
                            {p.is_default && (
                              <Badge
                                variant="secondary"
                                className="text-[10px]"
                              >
                                default
                              </Badge>
                            )}
                            {p.is_background && (
                              <Badge
                                variant="outline"
                                className="text-[10px]"
                              >
                                preemptible
                              </Badge>
                            )}
                            <span className="ml-2 text-xs text-slate-500">
                              {p.gpu_idle}/{p.gpu_total} {p.gpu_type ?? "GPU"} available
                            </span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selectedPartition?.is_background && (
                    <p className="text-xs text-slate-500">
                      Preemptible partition — submit auto-adds{" "}
                      <code>--requeue</code>; train_body resumes from latest
                      checkpoint after preemption.
                    </p>
                  )}
                  {partitions.isLoading && <LoadingState label="Loading partitions..." rows={2} />}
                  {partitions.error && <ErrorState message={(partitions.error as Error).message} />}
                </Field>
              ) : (
                <>
                  <Field label="Job class">
                    <Select
                      value={mlxpJobClass}
                      onValueChange={(v) =>
                        setMlxpJobClass(v as "dedicated" | "normal" | "background")
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="dedicated">dedicated</SelectItem>
                        <SelectItem value="normal">normal</SelectItem>
                        <SelectItem value="background">background</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-slate-500">
                      {mlxpJobClass === "dedicated" ? (
                        <>
                          Pins the selected node with top priority (hostname{" "}
                          <code>In</code> affinity + <code>mlxp/job-class</code> label).
                        </>
                      ) : (
                        <>
                          Queue job: placed on any free GPU in the zone. May be
                          suspended for higher classes and auto-resumes (~5 min
                          cycle); training continues from the latest checkpoint.
                        </>
                      )}
                    </p>
                  </Field>
                  {mlxpJobClass === "dedicated" && (
                <Field label="Node">
                  <Select value={mlxpNode} onValueChange={setMlxpNode}>
                    <SelectTrigger>
                      <SelectValue placeholder="select a default node..." />
                    </SelectTrigger>
                    <SelectContent>
                      {mlxp.data?.map((n) => (
                        <SelectItem key={n.name} value={n.name}>
                          <span className="flex items-center gap-2">
                            <span className="font-mono">{n.name}</span>
                            <span
                              className={`text-xs ${n.gpu_free > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}`}
                            >
                              {n.gpu_free}/{n.gpu_total} available
                            </span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {mlxp.isLoading && <LoadingState label="Loading MLXP nodes..." rows={2} />}
                  {mlxp.error && <ErrorState message={(mlxp.error as Error).message} />}
                  <p className="text-xs text-slate-500">
                    Pick the MLXP node this app should target by default.
                    Your selection is saved locally for next time.
                  </p>
                </Field>
                  )}
                </>
              )}

              <JobNameField
                value={shownJobName}
                defaultValue={defaultJobName}
                variantName={variantName}
                touched={jobNameTouched}
                description={
                  isSlurm ? (
                    <>
                      Used as <code>--job-name</code> and the wandb run id.
                    </>
                  ) : (
                    <>
                      Carried as the MLXP <code>display-name</code>{" "}
                      annotation and the wandb run id.
                    </>
                  )
                }
                onChange={(value) => {
                  setJobName(value);
                  setJobNameTouched(true);
                }}
                onReset={() => setJobNameTouched(false)}
              />
              <TrainNoteField
                value={trainNote}
                defaultValue={trainNoteDefault}
                variantName={variantName}
                touched={activeTrainNoteEdit !== null}
                valid={trainNoteValid}
                onChange={(value) => {
                  setTrainNoteEdit({ scope: trainNoteScope, value });
                }}
                onReset={() => setTrainNoteEdit(null)}
              />
              <div className="flex justify-end border-t border-slate-100 pt-5 dark:border-slate-900">
                <Button
                  onClick={() => setSubmitStep("config")}
                  disabled={!canReviewConfig}
                  className="gap-1"
                >
                  Review config.sh
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
          )}

          {submitStep === "config" && variantName && (
            <>
              <ConfigCard
                variantName={selectedVariantName}
                flagsUrl={`/api/variants/${variantName}/flags?cluster=${cluster}&phase=${phase}`}
                queryKey={["variant-flags", variantName, cluster, phase]}
                cluster={cluster}
                phase={phase}
                checkpointOverride={wantsCheckpoint ? checkpointPath : null}
                checkpointOverrideExists={checkpointExistsValue}
                effectiveConfigText={displayedConfigPreview?.text ?? null}
                effectiveConfigPath={displayedConfigPreview?.path ?? null}
                modelLabel={displayedConfigPreview?.model_label ?? null}
                modelRepoPath={displayedConfigPreview?.model_repo_path ?? null}
                modelGitCommit={jobGitStatus.data?.commit ?? null}
                modelRepoError={modelRepoError}
                modelRepoMessage={modelRepoMessage}
                modelRepoChecking={jobGitStatus.isLoading}
                effectiveConfigLoading={configPreview.isLoading && !displayedConfigPreview}
                effectiveConfigError={configPreview.error as Error | null}
                flagsOverride={displayedConfigPreview?.flags ?? null}
                flagEditors={flagEditors}
                extraFlagRows={extraFlagRows}
                showCheckpointPathRow={false}
                showEffectiveConfigPathRows={false}
                loading={variant.isLoading}
                error={variantError}
              />
              <div className="flex justify-between">
                <Button variant="outline" onClick={() => setSubmitStep("job")}>
                  Back to Job
                </Button>
                <div className="flex items-center gap-2">
                  {trainOverridesChanged && (
                    <Button
                      variant="destructiveOutline"
                      onClick={() => setTrainConfigEdit(null)}
                    >
                      Reset defaults
                    </Button>
                  )}
                  <Button
                    onClick={() => setSubmitStep("modality")}
                    disabled={!canReviewModality}
                    className="gap-1"
                  >
                    Review modality.py
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </>
          )}

          {submitStep === "modality" && variantName && (
            <>
              <DataInterfaceCard
                variantName={selectedVariantName}
                loading={variant.isLoading}
                error={variantError}
              />
              <div className="flex justify-between">
                <Button variant="outline" onClick={() => setSubmitStep("config")}>
                  Back to config.sh
                </Button>
                <Button onClick={handleSubmitClick} disabled={!canSubmit}>
                  {submitButtonLabel}
                </Button>
              </div>
            </>
          )}
        </div>

        <aside className="space-y-4 lg:sticky lg:top-6 lg:self-start">
          {isSlurm && (
            partitions.data ? (
              <AvailabilityCard cluster={cluster} partitions={partitions.data} />
            ) : (
              <AsideStatusCard
                title="GPU availability"
                description={`${cluster} · refreshes every 30s`}
                loading={partitions.isLoading}
                error={partitions.error as Error | null}
              />
            )
          )}
          {!isSlurm && (
            mlxp.data ? (
              <MlxpCard nodes={mlxp.data} />
            ) : (
              <AsideStatusCard
                title="MLXP (Naver, k8s)"
                description="GPU availability"
                loading={mlxp.isLoading}
                error={mlxp.error as Error | null}
              />
            )
          )}
        </aside>
      </div>

      <Dialog open={datasetDialogOpen} onOpenChange={setDatasetDialogOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Dataset override</DialogTitle>
            <DialogDescription>
              Edit the dataset values used for <code>--dataset-path</code> in
              this submission.
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[70vh] overflow-y-auto pr-1">
            {datasetField ?? (
              <EmptyState message="Pick an experiment before editing datasets." />
            )}
          </div>
          <DialogFooter>
            {datasetTouched && (
              <Button
                variant="outline"
                onClick={() => {
                  setDatasetEdit(null);
                  setDatasetDialogOpen(false);
                }}
              >
                Reset override
              </Button>
            )}
            <Button onClick={() => setDatasetDialogOpen(false)}>Done</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={gitCommitDialogOpen} onOpenChange={setGitCommitDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Training commit</DialogTitle>
            <DialogDescription>
              Set <code>TRAIN_GIT_COMMIT</code> for this submission.
              Leave it empty to use the repo HEAD at submit time.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Select
              value={selectedGitCommitValue}
              onValueChange={(value) => updateTrainConfig({ gitCommit: value })}
            >
              <SelectTrigger className="gap-2">
                <SelectValue placeholder="select a commit..." />
                {gitCommitOptions.isFetching && (
                  <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-slate-400" />
                )}
              </SelectTrigger>
              <SelectContent className="w-[var(--radix-select-trigger-width)] max-w-[var(--radix-select-trigger-width)] min-w-0">
                {gitCommitOptions.data?.commits.map((commit) => (
                  <SelectItem
                    key={commit.commit}
                    value={commit.commit}
                    className="w-full max-w-full overflow-hidden"
                  >
                    <span className="flex min-w-0 max-w-full items-center gap-2 overflow-hidden">
                      <span className="shrink-0 font-mono text-xs">
                        {commit.short_commit}
                      </span>
                      <span
                        className="min-w-0 truncate text-xs text-slate-500"
                        title={commit.subject}
                      >
                        {ellipsize(commit.subject, 40)}
                      </span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {gitCommitOptions.error && (
              <ErrorState message={(gitCommitOptions.error as Error).message} />
            )}
            {!trainGitCommitValid && (
              <p className="text-xs text-red-600 dark:text-red-400">
                The configured TRAIN_GIT_COMMIT is invalid. Pick a commit from
                the list.
              </p>
            )}
          </div>
          <DialogFooter>
            {trainGitCommitTrimmed && (
              <Button
                variant="outline"
                onClick={() => updateTrainConfig({ gitCommit: "" })}
              >
                Use current HEAD
              </Button>
            )}
            <Button onClick={() => setGitCommitDialogOpen(false)}>Done</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={gitDialogOpen} onOpenChange={setGitDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Commit uncommitted changes before training?</DialogTitle>
            <DialogDescription>
              Training submissions record the selected model-code git commit.
              The current {dirtyGitStatus?.repo_label ?? "training repo"} working
              tree is dirty, so the backend will commit these changes before
              submitting and store that hash in the job snapshot.
            </DialogDescription>
          </DialogHeader>
          {dirtyGitStatus?.repo_path && (
            <div className="rounded border border-slate-200 bg-slate-50 p-3 text-xs dark:border-slate-800 dark:bg-slate-950">
              <div className="text-slate-500 dark:text-slate-400">Repo</div>
              <div className="break-all font-mono">{dirtyGitStatus.repo_path}</div>
            </div>
          )}
          <div className="max-h-56 overflow-auto rounded border border-slate-200 bg-slate-50 p-3 font-mono text-xs dark:border-slate-800 dark:bg-slate-950">
            {dirtyGitStatus?.files.map((line) => (
              <div key={line}>{line}</div>
            ))}
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setGitDialogOpen(false)}
              disabled={submit.isPending}
            >
              Cancel
            </Button>
            <Button
              onClick={() => {
                setGitDialogOpen(false);
                submit.mutate({ commitDirtyChanges: true });
              }}
              disabled={submit.isPending}
            >
              {submit.isPending ? "Submitting..." : "Commit and submit"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function AsideStatusCard({
  title,
  description,
  loading,
  error,
}: {
  title: string;
  description: string;
  loading: boolean;
  error: Error | null;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {loading && <LoadingState label="Loading..." rows={3} />}
        {error && <ErrorState message={error.message} />}
        {!loading && !error && <EmptyState message="No data available." />}
      </CardContent>
    </Card>
  );
}

function SubmitStepper({
  activeStep,
  configEnabled,
  modalityEnabled,
  onStepChange,
}: {
  activeStep: SubmitStep;
  configEnabled: boolean;
  modalityEnabled: boolean;
  onStepChange: (step: SubmitStep) => void;
}) {
  const steps: Array<{ key: SubmitStep; label: string; enabled: boolean }> = [
    { key: "job", label: "Job", enabled: true },
    { key: "config", label: "config.sh", enabled: configEnabled },
    { key: "modality", label: "modality.py", enabled: modalityEnabled },
  ];
  return (
    <div className="mt-6 rounded-lg border border-slate-200 bg-white p-2 shadow-sm dark:border-slate-800 dark:bg-slate-950">
      <div className="grid gap-2 sm:grid-cols-3">
        {steps.map((step, index) => {
          const active = step.key === activeStep;
          return (
            <button
              key={step.key}
              type="button"
              disabled={!step.enabled}
              onClick={() => onStepChange(step.key)}
              aria-current={active ? "step" : undefined}
              className={[
                "flex min-h-11 items-center gap-3 rounded-md border border-transparent px-3 text-left text-sm transition",
                active
                  ? "text-slate-950 dark:text-slate-50"
                  : "text-slate-400 hover:bg-slate-50 hover:text-slate-600 dark:text-slate-600 dark:hover:bg-slate-900 dark:hover:text-slate-300",
                !step.enabled ? "cursor-not-allowed opacity-50" : "",
              ].join(" ")}
            >
              <span
                className={[
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                  active
                    ? "bg-slate-950 text-white dark:bg-slate-50 dark:text-slate-950"
                    : "bg-slate-100 text-slate-500 dark:bg-slate-900 dark:text-slate-400",
                ].join(" ")}
              >
                {index + 1}
              </span>
              <span className="font-medium">{step.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function PhaseField({
  value,
  onChange,
}: {
  value: Phase;
  onChange: (value: Phase) => void;
}) {
  return (
    <Field label="Phase">
      <Select value={value} onValueChange={(next) => onChange(next as Phase)}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="train">train</SelectItem>
          <SelectItem value="eval">eval</SelectItem>
        </SelectContent>
      </Select>
    </Field>
  );
}

function TrainNoteField({
  value,
  defaultValue,
  variantName,
  touched,
  valid,
  onChange,
  onReset,
}: {
  value: string;
  defaultValue: string;
  variantName: string;
  touched: boolean;
  valid: boolean;
  onChange: (value: string) => void;
  onReset: () => void;
}) {
  return (
    <Field
      label={
        <span className="flex items-center gap-2">
          Train note
          <span className="font-mono text-xs font-normal text-slate-500">
            {value.length}
          </span>
        </span>
      }
    >
      <div className="flex gap-2">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={
            variantName
              ? defaultValue || "Describe this submission"
              : "pick an experiment first"
          }
          className={
            variantName && !valid
              ? "flex-1 font-mono text-xs border-red-500 focus-visible:ring-red-500"
              : "flex-1 font-mono text-xs"
          }
        />
        {touched && (
          <Button variant="outline" size="sm" onClick={onReset}>
            Reset to default
          </Button>
        )}
      </div>
      <p className="text-xs text-slate-500">
        Stored in the submitted <code>config.sh</code> and shown on Results.
      </p>
      {variantName && !valid && (
        <p className="text-xs text-red-600 dark:text-red-400">
          TRAIN_NOTE is required before reviewing config.sh.
        </p>
      )}
    </Field>
  );
}

function JobNameField({
  value,
  defaultValue,
  variantName,
  touched,
  description,
  onChange,
  onReset,
}: {
  value: string;
  defaultValue: string;
  variantName: string;
  touched: boolean;
  description: React.ReactNode;
  onChange: (value: string) => void;
  onReset: () => void;
}) {
  return (
    <Field
      label={
        <span className="flex items-center gap-2">
          Job name
          <span className="font-mono text-xs font-normal text-slate-500">
            {value.length}
          </span>
        </span>
      }
    >
      <div className="flex gap-2">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={variantName ? defaultValue : "pick an experiment first"}
          className="flex-1 font-mono text-xs"
        />
        {touched && (
          <Button variant="outline" size="sm" onClick={onReset}>
            Reset to default
          </Button>
        )}
      </div>
      <p className="text-xs text-slate-500">{description}</p>
    </Field>
  );
}

function NumberCellEditor({
  value,
  onChange,
  valid,
  invalidMessage,
  hint,
}: {
  value: string;
  onChange: (value: string) => void;
  valid: boolean;
  invalidMessage: string;
  hint?: string;
}) {
  return (
    <div className="space-y-1">
      <Input
        type="number"
        min={1}
        step={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 text-xs"
      />
      {!valid && (
        <p className="text-xs text-red-600 dark:text-red-400">
          {invalidMessage}
        </p>
      )}
      {valid && hint && (
        <p className="text-xs text-slate-500">{hint}</p>
      )}
    </div>
  );
}

function parseEvalSetsInput(value: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of value.split(/[\s,]+/)) {
    const item = part.trim();
    if (!item || seen.has(item)) continue;
    seen.add(item);
    out.push(item);
  }
  return out;
}

function resolveDatasetOverride({
  touched,
  variant,
  singleDataset,
  multiDatasets,
}: {
  touched: boolean;
  variant?: Variant | null;
  singleDataset: string;
  multiDatasets: string[];
}): string | string[] | null {
  if (!touched || !variant) return null;
  if (variant.arrays.TRAIN_DATASET_NAMES || variant.arrays.DATASETS) {
    return multiDatasets;
  }
  if (variant.vars.DATASET_NAME) {
    return singleDataset;
  }
  return null;
}

function splitArgs(value: string): string[] {
  return value.split(/\s+/).filter(Boolean);
}

function formatEvalSetsInput(values: string[]): string {
  return parseEvalSetsInput(values.join(" ")).join(" ");
}
