"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import {
  api,
  type Variant,
  type SubmitResponse,
  type Partition,
  type Dataset,
  type MlxpNode,
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
import { useMyMlxpNode } from "@/hooks/use-my-mlxp-node";
import { DatasetField } from "@/components/submit/dataset-field";
import { MlxpCard } from "@/components/submit/mlxp-card";
import { AvailabilityCard } from "@/components/submit/availability-card";
import { ConfigCard } from "@/components/config-card";
import { useDatasetDir } from "@/hooks/use-dataset-dir";

type Phase = "train" | "resume" | "eval";

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
  const [checkpointPath, setCheckpointPath] = useState<string>("");
  const [checkpointTouched, setCheckpointTouched] = useState<boolean>(false);
  const [jobName, setJobName] = useState<string>("");
  const [jobNameTouched, setJobNameTouched] = useState<boolean>(false);

  // Dataset override state. For single-task variants, `singleDataset` holds
  // the chosen name. For multi-task, `multiDatasets` holds the array of
  // "name|cfg|weight" strings (N1.5) or plain "name" strings (N1.6). Both
  // are initialized from the variant's own config when it loads; user can
  // edit before submit.
  const [singleDataset, setSingleDataset] = useState<string>("");
  const [multiDatasets, setMultiDatasets] = useState<string[]>([]);
  const [datasetTouched, setDatasetTouched] = useState<boolean>(false);

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

  // Prefill checkpoint with the auto-detected one until the user types over it.
  useEffect(() => {
    if (!checkpointTouched && selectedCkpt.data?.path) {
      setCheckpointPath(selectedCkpt.data.path);
    }
  }, [selectedCkpt.data?.path, checkpointTouched]);

  // Auto-fill job_name with the default until the user types over it.
  // For MLXP, phase is locked to "train" — defaults reflect that.
  useEffect(() => {
    if (jobNameTouched) return;
    const effectivePhase = isSlurm ? phase : "train";
    setJobName(buildDefaultJobName(effectivePhase, variantName));
  }, [phase, variantName, isSlurm, jobNameTouched]);
  // Reset prefill state when variant / cluster / phase changes.
  useEffect(() => {
    setCheckpointPath("");
    setCheckpointTouched(false);
  }, [variantName, cluster, phase]);

  // Whenever a new variant loads (or the user picks a different one), reset
  // dataset state to that variant's defaults. `datasetTouched` is reset so
  // changing variant doesn't carry over a previous override.
  useEffect(() => {
    if (!variant.data) return;
    // Priority: N1.6 multi (TRAIN_DATASET_NAMES, name-only) → N1.5 multi
    // (DATASETS, "name|cfg|weight") → single (DATASET_NAME).
    if (variant.data.arrays.TRAIN_DATASET_NAMES) {
      setMultiDatasets(variant.data.arrays.TRAIN_DATASET_NAMES);
      setSingleDataset("");
    } else if (variant.data.arrays.DATASETS) {
      setMultiDatasets(variant.data.arrays.DATASETS);
      setSingleDataset("");
    } else if (variant.data.vars.DATASET_NAME) {
      setSingleDataset(variant.data.vars.DATASET_NAME);
      setMultiDatasets([]);
    }
    setDatasetTouched(false);
  }, [variant.data]);

  // Pick a sensible default partition when the cluster's partitions arrive
  // (or when the cluster changes and the previously-selected partition no
  // longer exists in the new cluster).
  useEffect(() => {
    if (!partitions.data) return;
    if (partition && partitions.data.some((p) => p.name === partition)) return;
    const def = partitions.data.find((p) => p.is_default) ?? partitions.data[0];
    setPartition(def?.name ?? "");
  }, [partitions.data, partition]);

  const submit = useMutation({
    mutationFn: () => {
      let dataset_override: string | string[] | null = null;
      if (datasetTouched) {
        if (variant.data?.arrays.TRAIN_DATASET_NAMES) dataset_override = multiDatasets;
        else if (variant.data?.arrays.DATASETS) dataset_override = multiDatasets;
        else if (variant.data?.vars.DATASET_NAME) dataset_override = singleDataset;
      }
      return api<SubmitResponse>("/api/submit", {
        method: "POST",
        body: JSON.stringify({
          cluster,
          variant: variantName,
          phase,
          partition: isSlurm ? partition : null,
          node: isSlurm ? null : mlxpNode,
          dataset_override,
          extra_args: extraArgs.split(/\s+/).filter(Boolean),
          checkpoint_path: wantsCheckpoint ? checkpointPath.trim() : null,
          job_name: jobNameTouched ? jobName.trim() : null,
        }),
      });
    },
    onSuccess: (data) => {
      toast.success(`Submitted job ${data.job_id} on ${cluster}`);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      router.push(`/jobs/${cluster}/${data.job_id}`);
    },
    onError: (err: Error) => toast.error(`Submit failed: ${err.message}`),
  });

  const canSubmit =
    !!variantName &&
    (isSlurm ? !!partition : !!mlxpNode) &&
    (!wantsCheckpoint ||
      (!!trimmedCkpt && checkpointExistsValue !== false)) &&
    !submit.isPending;
  const selectedPartition = partitions.data?.find((p) => p.name === partition);

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Submit a job</h1>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
              <CardDescription>
                All fields source from your local configs/ tree.
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
              </Field>

              <Field label="Variant">
                <Select value={variantName} onValueChange={setVariantName}>
                  <SelectTrigger>
                    <SelectValue placeholder="select a variant…" />
                  </SelectTrigger>
                  <SelectContent>
                    {variantNames.data?.map((v) => (
                      <SelectItem key={v} value={v}>
                        {v}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
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
                        <SelectItem value="resume">
                          resume (requires existing checkpoint)
                        </SelectItem>
                        <SelectItem value="eval">eval</SelectItem>
                      </SelectContent>
                    </Select>
                  </Field>

                  <Field label="Partition">
                    <Select value={partition} onValueChange={setPartition}>
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
                  </Field>

                  {variant.data && (
                    <DatasetField
                      variant={variant.data}
                      datasets={datasets.data ?? []}
                      single={singleDataset}
                      multi={multiDatasets}
                      onSingleChange={(v) => {
                        setSingleDataset(v);
                        setDatasetTouched(true);
                      }}
                      onMultiChange={(v) => {
                        setMultiDatasets(v);
                        setDatasetTouched(true);
                      }}
                      touched={datasetTouched}
                      cluster={cluster}
                      datasetDir={datasetDir}
                      onDatasetDirChange={setDatasetDir}
                      datasetsError={datasets.error as Error | null}
                    />
                  )}

                  <Field
                    label={
                      <span className="flex items-center gap-2">
                        Job name
                        <span className="font-mono text-xs font-normal text-slate-500">
                          {jobName.length}
                        </span>
                      </span>
                    }
                  >
                    <div className="flex gap-2">
                      <Input
                        value={jobName}
                        onChange={(e) => {
                          setJobName(e.target.value);
                          setJobNameTouched(true);
                        }}
                        placeholder={
                          variantName
                            ? buildDefaultJobName(phase, variantName)
                            : "pick a variant first"
                        }
                        className="flex-1 font-mono text-xs"
                      />
                      {jobNameTouched && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setJobNameTouched(false);
                            setJobName(buildDefaultJobName(phase, variantName));
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

                  {wantsCheckpoint && (
                    <Field label="Checkpoint">
                      <Input
                        value={checkpointPath}
                        onChange={(e) => {
                          setCheckpointPath(e.target.value);
                          setCheckpointTouched(true);
                        }}
                        placeholder={
                          selectedCkpt.isLoading
                            ? "looking up auto-pick…"
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
                          Choose a checkpoint
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
                            Auto-pick was{" "}
                            <code>{selectedCkpt.data.path}</code> — overriding.
                          </p>
                        )}
                    </Field>
                  )}

                  <Field label="Extra sbatch args (optional)">
                    <Input
                      placeholder="--exclusive --nice=100"
                      value={extraArgs}
                      onChange={(e) => setExtraArgs(e.target.value)}
                    />
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
                        <SelectValue placeholder="select your sanctioned node…" />
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
                    <p className="text-xs text-slate-500">
                      Each rlwrld member is sanctioned for a specific node —
                      check the GPU Resource Schedule sheet. Wrong-node
                      submission has triggered admin job deletions. Your
                      selection is saved locally for next time.
                    </p>
                  </Field>

                  {variant.data && (
                    <DatasetField
                      variant={variant.data}
                      datasets={datasets.data ?? []}
                      single={singleDataset}
                      multi={multiDatasets}
                      onSingleChange={(v) => {
                        setSingleDataset(v);
                        setDatasetTouched(true);
                      }}
                      onMultiChange={(v) => {
                        setMultiDatasets(v);
                        setDatasetTouched(true);
                      }}
                      touched={datasetTouched}
                      cluster={cluster}
                      datasetDir={datasetDir}
                      onDatasetDirChange={setDatasetDir}
                      datasetsError={datasets.error as Error | null}
                    />
                  )}

                  <Field
                    label={
                      <span className="flex items-center gap-2">
                        Job name
                        <span className="font-mono text-xs font-normal text-slate-500">
                          {jobName.length}
                        </span>
                      </span>
                    }
                  >
                    <div className="flex gap-2">
                      <Input
                        value={jobName}
                        onChange={(e) => {
                          setJobName(e.target.value);
                          setJobNameTouched(true);
                        }}
                        placeholder={
                          variantName
                            ? buildDefaultJobName("train", variantName)
                            : "pick a variant first"
                        }
                        className="flex-1 font-mono text-xs"
                      />
                      {jobNameTouched && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setJobNameTouched(false);
                            setJobName(buildDefaultJobName("train", variantName));
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

                  <Field label="Extra gr00t_finetune.py args (optional)">
                    <Input
                      placeholder="--tune-visual --random-diffusion"
                      value={extraArgs}
                      onChange={(e) => setExtraArgs(e.target.value)}
                    />
                  </Field>
                </>
              )}
            </CardContent>
          </Card>

          {variant.data && (
            <ConfigCard
              variantName={variant.data.name}
              flagsUrl={`/api/variants/${variant.data.name}/flags?cluster=${cluster}&phase=${phase}`}
              queryKey={["variant-flags", variant.data.name, cluster, phase]}
              modalityConfigFile={variant.data.vars.TRAIN_MODALITY_CONFIG ?? null}
              cluster={cluster}
              phase={phase}
              checkpointOverride={wantsCheckpoint ? checkpointPath : null}
              checkpointOverrideExists={checkpointExistsValue}
              className="mt-6"
            />
          )}

          <div className="flex justify-end">
            <Button onClick={() => submit.mutate()} disabled={!canSubmit}>
              {submit.isPending
                ? "Submitting…"
                : isSlurm
                  ? `Submit ${phase} → ${cluster}/${partition || "?"}`
                  : `Submit train → mlxp/${mlxpNode}/${variant.data?.vars.TRAIN_NUM_GPUS ?? "?"}×H200`}
            </Button>
          </div>
        </div>

        <aside className="space-y-4 lg:sticky lg:top-6 lg:self-start">
          {isSlurm && partitions.data && (
            <AvailabilityCard cluster={cluster} partitions={partitions.data} />
          )}
          {!isSlurm && mlxp.data && (
            <MlxpCard nodes={mlxp.data} yoursNode={mlxpNode} />
          )}
        </aside>
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
