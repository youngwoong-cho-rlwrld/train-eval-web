"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { api, type Variant, type SubmitResponse, type Partition, type Dataset, type MlxpNode } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronRight, Plus, X } from "lucide-react";

type Phase = "train" | "resume" | "eval";

export default function SubmitPage() {
  const router = useRouter();
  const qc = useQueryClient();

  const [cluster, setCluster] = useState<string>("kakao");
  const [variantName, setVariantName] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("train");
  const [partition, setPartition] = useState<string>("");
  const [extraArgs, setExtraArgs] = useState<string>("");

  // Dataset override state. For single-task variants, `singleDataset` holds
  // the chosen name. For multi-task, `multiDatasets` holds the array of
  // "name|cfg|weight" strings. Both are initialized from the variant's own
  // config when it loads; user can edit before submit.
  const [singleDataset, setSingleDataset] = useState<string>("");
  const [multiDatasets, setMultiDatasets] = useState<string[]>([]);
  const [datasetTouched, setDatasetTouched] = useState<boolean>(false);

  const clusters = useQuery({
    queryKey: ["clusters"],
    queryFn: () => api<{ clusters: string[] }>("/api/clusters").then((d) => d.clusters),
  });
  const variantNames = useQuery({
    queryKey: ["variants"],
    queryFn: () => api<{ variants: string[] }>("/api/variants").then((d) => d.variants),
  });
  const variant = useQuery({
    queryKey: ["variant", variantName],
    queryFn: () => (variantName ? api<Variant>(`/api/variants/${variantName}`) : null),
    enabled: !!variantName,
  });
  const partitions = useQuery({
    queryKey: ["partitions", cluster],
    queryFn: () => api<Partition[]>(`/api/clusters/${cluster}/partitions`),
    refetchInterval: 30_000,
    enabled: !!cluster,
  });
  const datasets = useQuery({
    queryKey: ["datasets", cluster],
    queryFn: () => api<Dataset[]>(`/api/clusters/${cluster}/datasets`),
    enabled: !!cluster,
  });
  const mlxp = useQuery({
    queryKey: ["mlxp-gpus"],
    queryFn: () => api<MlxpNode[]>("/api/mlxp/gpus"),
    refetchInterval: 60_000,
    retry: false,  // surfaces "kubectl not available" without retry storm
  });

  // Whenever a new variant loads (or the user picks a different one), reset
  // dataset state to that variant's defaults. `datasetTouched` is reset so
  // changing variant doesn't carry over a previous override.
  useEffect(() => {
    if (!variant.data) return;
    if (variant.data.vars.DATASET_NAME) {
      setSingleDataset(variant.data.vars.DATASET_NAME);
      setMultiDatasets([]);
    } else if (variant.data.arrays.DATASETS) {
      setMultiDatasets(variant.data.arrays.DATASETS);
      setSingleDataset("");
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
        if (variant.data?.vars.DATASET_NAME) dataset_override = singleDataset;
        else if (variant.data?.arrays.DATASETS) dataset_override = multiDatasets;
      }
      return api<SubmitResponse>("/api/submit", {
        method: "POST",
        body: JSON.stringify({
          cluster,
          variant: variantName,
          phase,
          partition,
          dataset_override,
          extra_args: extraArgs.split(/\s+/).filter(Boolean),
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

  const canSubmit = !!variantName && !!partition && !submit.isPending;
  const selectedPartition = partitions.data?.find((p) => p.name === partition);

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Submit a job</h1>
      <p className="mt-2 text-slate-600 dark:text-slate-400">
        Pick cluster + variant + partition, send to sbatch.
      </p>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
          <CardDescription>All fields source from your local configs/ tree.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <Field label="Cluster">
            <Select value={cluster} onValueChange={setCluster}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {clusters.data?.map((c) => (
                  <SelectItem key={c} value={c}>{c}</SelectItem>
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
                  <SelectItem key={v} value={v}>{v}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field label="Phase">
            <Select value={phase} onValueChange={(v) => setPhase(v as Phase)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="train">train</SelectItem>
                <SelectItem value="resume">resume (requires existing checkpoint)</SelectItem>
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
                      {p.is_default && <Badge variant="secondary" className="text-[10px]">default</Badge>}
                      {p.is_background && <Badge variant="outline" className="text-[10px]">preemptible</Badge>}
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
                Preemptible partition — submit auto-adds <code>--requeue</code>; train_body resumes from latest checkpoint after preemption.
              </p>
            )}
          </Field>

          {variant.data && (
            <DatasetField
              variant={variant.data}
              datasets={datasets.data ?? []}
              single={singleDataset}
              multi={multiDatasets}
              onSingleChange={(v) => { setSingleDataset(v); setDatasetTouched(true); }}
              onMultiChange={(v) => { setMultiDatasets(v); setDatasetTouched(true); }}
              touched={datasetTouched}
            />
          )}

          <Field label="Extra sbatch args (optional)">
            <Input
              placeholder="--exclusive --nice=100"
              value={extraArgs}
              onChange={(e) => setExtraArgs(e.target.value)}
            />
          </Field>
        </CardContent>
      </Card>

      {variant.data && <VariantPreview variant={variant.data} />}

      <div className="flex justify-end">
        <Button onClick={() => submit.mutate()} disabled={!canSubmit}>
          {submit.isPending ? "Submitting…" : `Submit ${phase} → ${cluster}/${partition || "?"}`}
        </Button>
      </div>
        </div>

        <aside className="space-y-4 lg:sticky lg:top-6 lg:self-start">
          {partitions.data && <AvailabilityCard cluster={cluster} partitions={partitions.data} />}
          {mlxp.data && mlxp.data.length > 0 && <MlxpCard nodes={mlxp.data} />}
        </aside>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function DatasetField({
  variant,
  datasets,
  single,
  multi,
  onSingleChange,
  onMultiChange,
  touched,
}: {
  variant: Variant;
  datasets: Dataset[];
  single: string;
  multi: string[];
  onSingleChange: (v: string) => void;
  onMultiChange: (v: string[]) => void;
  touched: boolean;
}) {
  const isMulti = !!variant.arrays.DATASETS;
  const labelSuffix = touched ? (
    <Badge variant="warning" className="ml-2 text-[10px]">override</Badge>
  ) : null;

  if (!isMulti) {
    return (
      <div className="space-y-1.5">
        <Label>Dataset {labelSuffix}</Label>
        <Select value={single} onValueChange={onSingleChange}>
          <SelectTrigger><SelectValue placeholder="select a dataset…" /></SelectTrigger>
          <SelectContent>
            {datasets.map((d) => (
              <SelectItem key={d.name} value={d.name}>
                <span className="font-mono text-xs">{d.name}</span>
                {d.height && d.width && (
                  <span className="ml-2 text-[10px] text-slate-500">{d.height}×{d.width}</span>
                )}
                {d.episodes !== null && (
                  <span className="ml-1 text-[10px] text-slate-500">· {d.episodes} ep</span>
                )}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-slate-500">
          Defaults to <code>{variant.vars.DATASET_NAME ?? "—"}</code> from{" "}
          <code>config.sh</code>. Changing here overrides for this submission only.
        </p>
      </div>
    );
  }

  // multi-task: editable list of "name|cfg|weight"
  const dataConfigDefault = variant.vars.DATA_CONFIG ?? "allex_thetwo_ck40_egostereo";
  const addRow = () => onMultiChange([...multi, `|${dataConfigDefault}|1.0`]);
  const removeRow = (i: number) => onMultiChange(multi.filter((_, j) => j !== i));
  const updateRow = (i: number, name: string, cfg: string, weight: string) => {
    const next = [...multi];
    next[i] = `${name}|${cfg}|${weight}`;
    onMultiChange(next);
  };

  return (
    <div className="space-y-1.5">
      <Label>Datasets {labelSuffix}</Label>
      <div className="space-y-2">
        {multi.map((entry, i) => {
          const [name, cfg, weight] = entry.split("|");
          return (
            <div key={i} className="flex items-center gap-2">
              <Select value={name} onValueChange={(v) => updateRow(i, v, cfg, weight)}>
                <SelectTrigger className="flex-1"><SelectValue placeholder="dataset…" /></SelectTrigger>
                <SelectContent>
                  {datasets.map((d) => (
                    <SelectItem key={d.name} value={d.name}>
                      <span className="font-mono text-xs">{d.name}</span>
                      {d.height && d.width && (
                        <span className="ml-2 text-[10px] text-slate-500">{d.height}×{d.width}</span>
                      )}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Input
                className="w-44 font-mono text-xs"
                value={cfg}
                onChange={(e) => updateRow(i, name, e.target.value, weight)}
                placeholder="data_config"
              />
              <Input
                className="w-20 font-mono text-xs"
                value={weight}
                onChange={(e) => updateRow(i, name, cfg, e.target.value)}
                placeholder="1.0"
              />
              <button
                onClick={() => removeRow(i)}
                className="rounded p-1.5 text-slate-400 hover:bg-slate-100 hover:text-red-600 dark:hover:bg-slate-800"
                title="Remove"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          );
        })}
        <Button variant="outline" size="sm" onClick={addRow} className="gap-1">
          <Plus className="h-3.5 w-3.5" /> add dataset
        </Button>
      </div>
      <p className="text-xs text-slate-500">
        Format: <code>name | data_config | weight</code>. Defaults from{" "}
        <code>config.sh</code>; changes apply to this submission only.
      </p>
    </div>
  );
}

function MlxpCard({ nodes }: { nodes: MlxpNode[] }) {
  const [open, setOpen] = useState(true);
  const idle = nodes.reduce((s, n) => s + n.gpu_free, 0);
  const total = nodes.reduce((s, n) => s + n.gpu_total, 0);
  return (
    <Card>
      <CardHeader className="cursor-pointer select-none" onClick={() => setOpen((o) => !o)}>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            MLXP (Naver, k8s)
          </CardTitle>
          <span className="font-mono text-xs text-slate-500">
            <span className={idle > 0 ? "text-green-600 dark:text-green-400" : ""}>{idle}</span>
            <span className="text-slate-400"> / {total}</span>
          </span>
        </div>
        <CardDescription className="text-xs">
          h200 nodes (8 GPU each) · only{" "}
          <code>h200-03-w-3a18</code> is sanctioned for our team
        </CardDescription>
      </CardHeader>
      {open && (
        <CardContent>
          <div className="space-y-2">
            {nodes.map((n) => (
              <div key={n.name} className="flex items-center justify-between gap-3 text-xs">
                <div className="min-w-0 flex-1 truncate font-mono">
                  {n.name}
                  {n.sanctioned && <Badge variant="default" className="ml-1 text-[10px]">yours</Badge>}
                </div>
                <div className="shrink-0 font-mono">
                  <span className="text-slate-400">free </span>
                  <span className={n.gpu_free > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>
                    {n.gpu_free}
                  </span>
                  <span className="text-slate-400">/{n.gpu_total}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

function AvailabilityCard({ cluster, partitions }: { cluster: string; partitions: Partition[] }) {
  const [open, setOpen] = useState(true);
  const totalIdleGpu = partitions.reduce((s, p) => s + p.gpu_idle, 0);
  const totalGpu = partitions.reduce((s, p) => s + p.gpu_total, 0);
  return (
    <Card>
      <CardHeader className="cursor-pointer select-none" onClick={() => setOpen((o) => !o)}>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            GPU availability
          </CardTitle>
          <span className="font-mono text-xs text-slate-500">
            <span className={totalIdleGpu > 0 ? "text-green-600 dark:text-green-400" : ""}>{totalIdleGpu}</span>
            <span className="text-slate-400"> / {totalGpu}</span>
          </span>
        </div>
        <CardDescription className="text-xs">
          {cluster} · refreshes every 30s
        </CardDescription>
      </CardHeader>
      {open && (
        <CardContent>
          <div className="space-y-2">
            {partitions.map((p) => (
              <div key={p.name} className="flex items-center justify-between gap-3 text-xs">
                <div className="min-w-0 flex-1 truncate font-mono" title={p.name}>
                  {p.name}
                  {p.is_default && <Badge variant="secondary" className="ml-1 text-[10px]">def</Badge>}
                  {p.is_background && <Badge variant="outline" className="ml-1 text-[10px]">bg</Badge>}
                </div>
                <div className="shrink-0 font-mono">
                  <span className="text-slate-400">free </span>
                  <span className={p.gpu_idle > 0 ? "text-green-600 dark:text-green-400" : "text-slate-500"}>
                    {p.gpu_idle}
                  </span>
                  <span className="text-slate-400">/{p.gpu_total}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

function VariantPreview({ variant }: { variant: Variant }) {
  const KEY_ORDER = [
    "MODEL_VERSION", "MAX_STEPS", "SAVE_STEPS", "TRAIN_NUM_GPUS", "TRAIN_BATCH_SIZE",
    "DATA_DIR", "DATASET_NAME", "DATA_CONFIG", "TASK_NAME",
    "N_EPISODES", "N_RUNS", "EXECUTION_HORIZON", "MAX_EPISODE_STEPS", "TRAIN_NOTE",
  ];
  const knownScalars = KEY_ORDER.filter((k) => k in variant.vars);

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle>{variant.name}</CardTitle>
        <CardDescription>{variant.vars.TRAIN_NOTE ?? "—"}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
          {knownScalars.map((k) => (
            <div key={k} className="flex flex-col">
              <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
              <dd className="font-mono">{variant.vars[k]}</dd>
            </div>
          ))}
        </dl>
        {variant.arrays.DATASETS && (
          <div>
            <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">DATASETS</div>
            <ul className="space-y-1 font-mono text-xs">
              {variant.arrays.DATASETS.map((d) => <li key={d}>· {d}</li>)}
            </ul>
          </div>
        )}
        {variant.arrays.TASKS && (
          <div>
            <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">TASKS</div>
            <ul className="space-y-1 font-mono text-xs">
              {variant.arrays.TASKS.map((t) => <li key={t}>· {t}</li>)}
            </ul>
          </div>
        )}
        {variant.arrays.EVAL_SETS && (
          <div className="flex items-center gap-2 text-xs">
            <span className="uppercase tracking-wide text-slate-500">EVAL_SETS:</span>
            {variant.arrays.EVAL_SETS.map((es) => (
              <Badge key={es} variant="secondary">{es}</Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
