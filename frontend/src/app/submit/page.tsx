"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import {
  api,
  type Variant,
  type SubmitResponse,
  type SubmitConfigPreview,
  type Partition,
  type Dataset,
  type MlxpNode,
  type GitStatus,
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
  type ExtraFlagRow,
  type FlagEditor,
} from "@/components/config-card";
import { useDatasetDir } from "@/hooks/use-dataset-dir";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";

type Phase = "train" | "eval";
type EvalConfigEdit = {
  scope: string;
  nEpisodes: string;
  nRuns: string;
  evalSets: string;
};
type TrainConfigEdit = {
  scope: string;
  numGpus: string;
  batchSize: string;
  maxSteps: string;
  saveSteps: string;
};

function buildDefaultJobName(phase: Phase, variant: string): string {
  if (!variant) return "";
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const ts =
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}` +
    `_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return `${phase}_${variant}_${ts}`;
}

export default function SubmitPage() {
  const router = useRouter();
  const qc = useQueryClient();

  const [cluster, setCluster] = useState<string>("kakao");
  const [variantName, setVariantName] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("train");
  const [partition, setPartition] = useState<string>("");
  // Persisted across sessions + synced across pages via useMyMlxpNode.
  const [mlxpNode, setMlxpNode] = useMyMlxpNode();
  const [extraArgs, setExtraArgs] = useState<string>("");
  const [evalOverwriteResults, setEvalOverwriteResults] = useState<boolean>(false);
  const [evalConfigEdit, setEvalConfigEdit] = useState<EvalConfigEdit | null>(null);
  const [trainConfigEdit, setTrainConfigEdit] = useState<TrainConfigEdit | null>(null);
  const [checkpointEdit, setCheckpointEdit] = useState<{
    scope: string;
    value: string;
  } | null>(null);
  const [jobName, setJobName] = useState<string>("");
  const [jobNameTouched, setJobNameTouched] = useState<boolean>(false);
  const [gitDialogOpen, setGitDialogOpen] = useState<boolean>(false);
  const [dirtyGitStatus, setDirtyGitStatus] = useState<GitStatus | null>(null);
  const [preflightPending, setPreflightPending] = useState<boolean>(false);

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
  const variant = useQuery({
    queryKey: ["variant", variantName],
    queryFn: () =>
      variantName ? api<Variant>(`/api/variants/${variantName}`) : null,
    enabled: !!variantName,
  });
  const partitions = useQuery({
    queryKey: ["partitions", cluster],
    queryFn: () => api<Partition[]>(`/api/clusters/${cluster}/partitions`),
    refetchInterval: 30_000,
    enabled: !!cluster && cluster !== "mlxp",
  });
  const isSlurm = cluster !== "mlxp";
  const submitPhase: Phase = isSlurm ? phase : "train";
  const checkpointScope = `${cluster}:${variantName}:${submitPhase}`;
  const evalConfigScope = `${checkpointScope}:eval-config`;
  const trainConfigScope = `${checkpointScope}:train-config`;

  const [datasetDir, setDatasetDir] = useDatasetDir(cluster);
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
  const wantsCheckpoint = phase === "eval" && isSlurm && !!variantName;
  const selectedCkpt = useQuery({
    queryKey: ["selected-checkpoint", variantName, cluster],
    queryFn: () =>
      api<{ path: string | null; step: number | null }>(
        `/api/variants/${variantName}/selected-checkpoint?cluster=${cluster}`,
      ),
    enabled: wantsCheckpoint,
  });
  const activeCheckpointEdit =
    checkpointEdit?.scope === checkpointScope ? checkpointEdit : null;
  const checkpointPath = activeCheckpointEdit
    ? activeCheckpointEdit.value
    : selectedCkpt.data?.path ?? "";
  const trimmedCkpt = checkpointPath.trim();
  const checkpointExists = useQuery({
    queryKey: ["path-exists", cluster, trimmedCkpt],
    queryFn: () =>
      api<{ exists: boolean; kind: "dir" | "file" | null }>(
        `/api/clusters/${cluster}/path-exists?path=${encodeURIComponent(trimmedCkpt)}`,
      ),
    enabled: wantsCheckpoint && !!trimmedCkpt,
  });
  const checkpointExistsValue: boolean | null = !trimmedCkpt
    ? null
    : checkpointExists.isLoading || checkpointExists.data === undefined
      ? null
      : checkpointExists.data.exists;

  const defaultJobName = useMemo(
    () => buildDefaultJobName(submitPhase, variantName),
    [submitPhase, variantName],
  );
  const shownJobName = jobNameTouched ? jobName : defaultJobName;
  const evalConfigDefaults = useMemo(
    () => ({
      nEpisodes: variant.data?.vars.N_EPISODES ?? "",
      nRuns: variant.data?.vars.N_RUNS ?? "",
      evalSets: formatEvalSetsInput(variant.data?.arrays.EVAL_SETS ?? []),
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
    };
    setEvalConfigEdit({ ...base, ...patch, scope: evalConfigScope });
  };
  const trainConfigDefaults = useMemo(() => {
    const vars = variant.data?.vars;
    const numGpus = vars?.TRAIN_NUM_GPUS ?? "2";
    const model = vars?.MODEL_VERSION ?? "n1.5";
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
    };
  }, [variant.data]);
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
  const trainNumGpusParsed = Number.parseInt(trainNumGpus.trim(), 10);
  const trainBatchSizeParsed = Number.parseInt(
    trainBatchSize.trim(),
    10,
  );
  const trainMaxStepsParsed = Number.parseInt(trainMaxSteps.trim(), 10);
  const trainSaveStepsParsed = Number.parseInt(trainSaveSteps.trim(), 10);
  const isPositiveInteger = (value: string) => /^[1-9]\d*$/.test(value.trim());
  const trainModel = variant.data?.vars.MODEL_VERSION ?? "n1.5";
  const wantsTrainConfig = submitPhase === "train" && !!variantName;
  const trainNumGpusValid =
    !wantsTrainConfig ||
    (isPositiveInteger(trainNumGpus) &&
      (isSlurm || [1, 2, 4, 8].includes(trainNumGpusParsed)));
  const trainBatchSizeValid =
    !wantsTrainConfig || isPositiveInteger(trainBatchSize);
  const trainMaxStepsValid =
    !wantsTrainConfig || isPositiveInteger(trainMaxSteps);
  const trainSaveStepsValid =
    !wantsTrainConfig || isPositiveInteger(trainSaveSteps);
  const trainConfigValid =
    trainNumGpusValid &&
    trainBatchSizeValid &&
    trainMaxStepsValid &&
    trainSaveStepsValid;
  const updateTrainConfig = (
    patch: Partial<Omit<TrainConfigEdit, "scope">>,
  ) => {
    const base = activeTrainConfigEdit ?? {
      scope: trainConfigScope,
      numGpus: trainConfigDefaults.numGpus,
      batchSize: trainConfigDefaults.batchSize,
      maxSteps: trainConfigDefaults.maxSteps,
      saveSteps: trainConfigDefaults.saveSteps,
    };
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
    submitPhase === "train"
      ? trainModel === "n1.5"
        ? trainBatchSizeParsed * trainNumGpusParsed
        : trainBatchSizeParsed
      : null;

  const buildSubmitBody = (commitDirtyChanges: boolean) => {
      let dataset_override: string | string[] | null = null;
      if (datasetTouched) {
        if (variant.data?.arrays.TRAIN_DATASET_NAMES) dataset_override = multiDatasets;
        else if (variant.data?.arrays.DATASETS) dataset_override = multiDatasets;
        else if (variant.data?.vars.DATASET_NAME) dataset_override = singleDataset;
      }
      return {
        cluster,
        variant: variantName,
        phase: submitPhase,
        partition: isSlurm ? selectedPartitionName : null,
        node: isSlurm ? null : mlxpNode,
        dataset_override,
        extra_args: extraArgs.split(/\s+/).filter(Boolean),
        train_num_gpus: submitPhase === "train" ? trainNumGpusParsed : null,
        train_global_batch_size: submittedTrainGlobalBatchSize,
        train_max_steps: submitPhase === "train" ? trainMaxStepsParsed : null,
        train_save_steps: submitPhase === "train" ? trainSaveStepsParsed : null,
        eval_num_envs_per_gpu: null,
        eval_n_episodes: wantsCheckpoint ? evalNEpisodesParsed : null,
        eval_n_runs: wantsCheckpoint ? evalNRunsParsed : null,
        eval_sets: wantsCheckpoint ? evalSetValues : null,
        eval_overwrite_results: wantsCheckpoint ? evalOverwriteResults : false,
        checkpoint_path: wantsCheckpoint ? checkpointPath.trim() : null,
        job_name: shownJobName.trim() || null,
        commit_dirty_changes: commitDirtyChanges,
      };
  };

  const configPreviewEnabled =
    !!variantName &&
    !variant.isLoading &&
    !variantError &&
    (submitPhase !== "train" || trainConfigValid) &&
    (!wantsCheckpoint || (evalNEpisodesValid && evalNRunsValid && evalSetsValid));
  const configPreview = useQuery({
    queryKey: [
      "submit-config-preview",
      cluster,
      variantName,
      submitPhase,
      selectedPartitionName,
      mlxpNode,
      datasetTouched,
      singleDataset,
      multiDatasets,
      extraArgs,
      trainNumGpus,
      trainBatchSize,
      trainMaxSteps,
      trainSaveSteps,
      evalNEpisodes,
      evalNRuns,
      evalSetValues,
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
    retry: false,
  });
  const trainGitStatus = useQuery({
    queryKey: ["submit-git-status", cluster, variantName],
    queryFn: () => {
      const qs = new URLSearchParams({ cluster, variant: variantName });
      return api<GitStatus>(`/api/submit/git-status?${qs}`);
    },
    enabled:
      submitPhase === "train" &&
      !!variantName &&
      !variant.isLoading &&
      !variantError &&
      configPreview.isSuccess &&
      !configPreview.data.model_repo_error,
    retry: false,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
  const modelRepoError =
    configPreview.data?.model_repo_error ??
    trainGitStatus.data?.error ??
    ((trainGitStatus.error as Error | null)?.message ?? null);
  const modelRepoMessage =
    !modelRepoError && trainGitStatus.data
      ? trainGitStatus.data.dirty
        ? `Git ${trainGitStatus.data.short_commit ?? "unknown"} · uncommitted changes`
        : `Git ${trainGitStatus.data.short_commit ?? "unknown"} · clean`
      : null;

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
    if (submitPhase !== "train") {
      submit.mutate({ commitDirtyChanges: false });
      return;
    }
    setPreflightPending(true);
    try {
      const qs = new URLSearchParams({ cluster, variant: variantName });
      const status = await api<GitStatus>(`/api/submit/git-status?${qs}`);
      if (status.error) {
        toast.error(`Git status failed: ${status.error}`);
        return;
      }
      if (status.dirty) {
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
    (isSlurm ? !!selectedPartitionName : !!mlxpNode) &&
    (!wantsCheckpoint ||
      (!!trimmedCkpt &&
        checkpointExistsValue !== false &&
        evalNEpisodesValid &&
        evalNRunsValid &&
        evalSetsValid)) &&
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
    ? { content: datasetField, wide: true }
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
              placeholder={
                selectedCkpt.isLoading
                  ? "looking up auto-pick..."
                  : "/absolute/path/to/checkpoint-N"
              }
              className={
                trimmedCkpt && checkpointExistsValue === false
                  ? "font-mono text-xs border-red-500 focus-visible:ring-red-500"
                  : "font-mono text-xs"
              }
            />
            {!trimmedCkpt && !selectedCkpt.isLoading && (
              <p className="text-xs text-red-600 dark:text-red-400">
                Choose a checkpoint.
              </p>
            )}
            {trimmedCkpt && checkpointExistsValue === false && (
              <p className="text-xs text-red-600 dark:text-red-400">
                Path not found on <code>{cluster}</code>.
              </p>
            )}
            {trimmedCkpt && selectedCkpt.data?.path &&
              trimmedCkpt !== selectedCkpt.data.path && (
                <p className="text-xs text-slate-500">
                  Auto-pick was <code>{selectedCkpt.data.path}</code> -
                  overriding.
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
  if (wantsTrainConfig && variant.data) {
    flagEditors["--num-gpus"] = (
      <NumberCellEditor
        value={trainNumGpus}
        onChange={(value) => updateTrainConfig({ numGpus: value })}
        valid={trainNumGpusValid}
        invalidMessage={isSlurm ? "Positive integer." : "Use 1, 2, 4, or 8."}
      />
    );
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
  if (wantsTrainConfig && activeTrainConfigEdit) {
    extraFlagRows.push({
      key: "reset-train",
      flag: "train overrides",
      value: "modified",
      editor: (
        <Button
          variant="outline"
          size="sm"
          onClick={() => setTrainConfigEdit(null)}
        >
          Reset train defaults
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
        selectedCkpt.data?.path ||
        (selectedCkpt.isLoading ? "..." : "(none found - eval will fail)"),
      editor: checkpointEditor,
    });
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
  extraFlagRows.push({
    key: "extra-sbatch-args",
    flag: "extra sbatch args",
    value: extraArgs.trim() || "(none)",
    editor: {
      wide: true,
      content: (
        <Input
          placeholder={
            isSlurm
              ? "--exclusive --nice=100"
              : "--exclude=rlwrld-gpu-260504-260803-st-p5en-48xl-3"
          }
          value={extraArgs}
          onChange={(e) => setExtraArgs(e.target.value)}
        />
      ),
    },
  });

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Submit a job</h1>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Job</CardTitle>
              <CardDescription>
                Choose where the job runs and how it is identified.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <Field label="Cluster">
                <Select value={cluster} onValueChange={setCluster}>
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
                <Select value={variantName} onValueChange={setVariantName}>
                  <SelectTrigger>
                    <SelectValue placeholder="select an experiment..." />
                  </SelectTrigger>
                  <SelectContent>
                    {variantNames.data?.map((v) => (
                      <SelectItem key={v} value={v}>
                        {v}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {variantNames.isLoading && <LoadingState label="Loading experiments..." rows={1} />}
                {variantNames.error && <ErrorState message={(variantNames.error as Error).message} />}
              </Field>

              {isSlurm && (
                <>
                  <Field label="Phase">
                    <Select
                      value={phase}
                      onValueChange={(v) => setPhase(v as Phase)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="train">train</SelectItem>
                        <SelectItem value="eval">eval</SelectItem>
                      </SelectContent>
                    </Select>
                  </Field>

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
                                {p.gpu_idle}/{p.gpu_total} GPU
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

                  <Field
                    label={
                      <span className="flex items-center gap-2">
                        Job name
                        <span className="font-mono text-xs font-normal text-slate-500">
                          {shownJobName.length}
                        </span>
                      </span>
                    }
                  >
                    <div className="flex gap-2">
                      <Input
                        value={shownJobName}
                        onChange={(e) => {
                          setJobName(e.target.value);
                          setJobNameTouched(true);
                        }}
                        placeholder={
                          variantName
                            ? defaultJobName
                            : "pick an experiment first"
                        }
                        className="flex-1 font-mono text-xs"
                      />
                      {jobNameTouched && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setJobNameTouched(false);
                          }}
                        >
                          Reset to default
                        </Button>
                      )}
                    </div>
                    <p className="text-xs text-slate-500">
                      Used as <code>--job-name</code> and the wandb run id.
                    </p>
                  </Field>

                </>
              )}

              {!isSlurm && (
                <>
                  <Field label="Phase">
                    <Select value="train" onValueChange={() => {}}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="train">train</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-slate-500">
                      MLXP currently supports <code>train</code> only.
                    </p>
                  </Field>

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
                                {n.gpu_free}/{n.gpu_total} free
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

                  <Field
                    label={
                      <span className="flex items-center gap-2">
                        Job name
                        <span className="font-mono text-xs font-normal text-slate-500">
                          {shownJobName.length}
                        </span>
                      </span>
                    }
                  >
                    <div className="flex gap-2">
                      <Input
                        value={shownJobName}
                        onChange={(e) => {
                          setJobName(e.target.value);
                          setJobNameTouched(true);
                        }}
                        placeholder={
                          variantName
                            ? defaultJobName
                            : "pick an experiment first"
                        }
                        className="flex-1 font-mono text-xs"
                      />
                      {jobNameTouched && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setJobNameTouched(false);
                          }}
                        >
                          Reset to default
                        </Button>
                      )}
                    </div>
                    <p className="text-xs text-slate-500">
                      Carried as the MLXP <code>display-name</code> annotation
                      and the wandb run id.
                    </p>
                  </Field>

                </>
              )}
            </CardContent>
          </Card>

          {variantName && (
            <ConfigCard
              variantName={variant.data?.name ?? variantName}
              flagsUrl={`/api/variants/${variantName}/flags?cluster=${cluster}&phase=${submitPhase}`}
              queryKey={["variant-flags", variantName, cluster, submitPhase]}
              modalityConfigFile={variant.data?.vars.TRAIN_MODALITY_CONFIG ?? null}
              cluster={cluster}
              phase={submitPhase}
              checkpointOverride={wantsCheckpoint ? checkpointPath : null}
              checkpointOverrideExists={checkpointExistsValue}
              effectiveConfigText={configPreview.data?.text ?? null}
              effectiveConfigPath={configPreview.data?.path ?? null}
              modelLabel={configPreview.data?.model_label ?? null}
              modelRepoPath={configPreview.data?.model_repo_path ?? null}
              modelRepoError={modelRepoError}
              modelRepoMessage={modelRepoMessage}
              modelRepoChecking={submitPhase === "train" && trainGitStatus.isLoading}
              effectiveConfigLoading={configPreview.isLoading}
              effectiveConfigError={configPreview.error as Error | null}
              flagsOverride={configPreview.data?.flags ?? null}
              flagEditors={flagEditors}
              extraFlagRows={extraFlagRows}
              showCheckpointPathRow={false}
              loading={variant.isLoading}
              error={variantError}
              className="mt-6"
            />
          )}

          <div className="flex justify-end">
            <Button onClick={handleSubmitClick} disabled={!canSubmit}>
              {submit.isPending || preflightPending
                ? "Submitting…"
                : isSlurm
                  ? `Submit ${phase} → ${cluster}/${selectedPartitionName || "?"}`
                  : `Submit train → mlxp/${mlxpNode}/${trainNumGpus || "?"}×H200`}
            </Button>
          </div>
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
              <MlxpCard nodes={mlxp.data} yoursNode={mlxpNode} />
            ) : (
              <AsideStatusCard
                title="MLXP (Naver, k8s)"
                description={`your node: ${mlxpNode || "-"}`}
                loading={mlxp.isLoading}
                error={mlxp.error as Error | null}
              />
            )
          )}
        </aside>
      </div>

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

function formatEvalSetsInput(values: string[]): string {
  return parseEvalSetsInput(values.join(" ")).join(" ");
}
