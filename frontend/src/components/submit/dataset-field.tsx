"use client";

import { useState } from "react";
import { Plus, Settings, X } from "lucide-react";
import type { Variant, Dataset } from "@/lib/api";
import { MLXP_DATASETS_DIR } from "@/lib/mlxp-config";
import { Badge } from "@/components/ui/badge";
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
import { ImmediateTooltip } from "@/components/immediate-tooltip";

export type DatasetFieldProps = {
  variant: Variant;
  datasets: Dataset[];
  single: string;
  multi: string[];
  onSingleChange: (v: string) => void;
  onMultiChange: (v: string[]) => void;
  touched: boolean;
  cluster: string;
  datasetDir: string;
  onDatasetDirChange: (v: string) => void;
  datasetsError: Error | null;
};

export function DatasetField({
  variant,
  datasets,
  single,
  multi,
  onSingleChange,
  onMultiChange,
  touched,
  cluster,
  datasetDir,
  onDatasetDirChange,
  datasetsError,
}: DatasetFieldProps) {
  // Multi-task variants come in two shapes:
  //   - N1.5: DATASETS=("name|cfg|weight" ...) — three editable fields per row
  //   - N1.6: TRAIN_DATASET_NAMES=("name" ...)  — name-only (cfg/weight live
  //          in the modality config python file)
  const namesArray = variant.arrays.TRAIN_DATASET_NAMES;
  const datasetsArray = variant.arrays.DATASETS;
  const multiKind: "names" | "datasets" | null = namesArray
    ? "names"
    : datasetsArray
      ? "datasets"
      : null;

  const dirSuffix = (
    <DatasetDirControl
      cluster={cluster}
      datasetDir={datasetDir}
      onChange={onDatasetDirChange}
    />
  );
  const labelSuffix = (
    <>
      {touched && (
        <Badge variant="warning" className="ml-2 text-[10px]">
          override
        </Badge>
      )}
      {dirSuffix}
    </>
  );

  const errorBanner = datasetsError ? (
    <p className="text-xs text-red-600 dark:text-red-400">
      Failed to list datasets at <code>{datasetDir}</code>: {datasetsError.message}
    </p>
  ) : null;

  if (multiKind === null) {
    return (
      <>
        <SingleDatasetPicker
          variant={variant}
          datasets={datasets}
          value={single}
          onChange={onSingleChange}
          labelSuffix={labelSuffix}
        />
        {errorBanner}
      </>
    );
  }

  if (multiKind === "names") {
    return (
      <>
        <NamesOnlyPicker
          datasets={datasets}
          values={multi}
          onChange={onMultiChange}
          labelSuffix={labelSuffix}
        />
        {errorBanner}
      </>
    );
  }

  return (
    <>
      <NameCfgWeightPicker
        variant={variant}
        datasets={datasets}
        values={multi}
        onChange={onMultiChange}
        labelSuffix={labelSuffix}
      />
      {errorBanner}
    </>
  );
}

function DatasetDirControl({
  cluster,
  datasetDir,
  onChange,
}: {
  cluster: string;
  datasetDir: string;
  onChange: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(datasetDir);
  return (
    <>
      <ImmediateTooltip content={`Dataset directory: ${datasetDir}`}>
        <button
          type="button"
          onClick={() => {
            setDraft(datasetDir);
            setOpen(true);
          }}
          className="ml-2 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-50"
        >
          <Settings className="h-3 w-3" />
          <code className="font-mono">{datasetDir}</code>
        </button>
      </ImmediateTooltip>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Dataset directory on <code className="font-mono">{cluster}</code>
            </DialogTitle>
            <DialogDescription>
              Absolute path on the cluster (slurm: a path or <code>~/</code>;
              mlxp: under <code>/data/</code>). The submit page lists every
              subdir of this path that contains <code>meta/info.json</code>.
              Saved per-cluster in this browser.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={MLXP_DATASETS_DIR}
            className="font-mono text-xs"
            autoFocus
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                onChange(draft.trim());
                setOpen(false);
              }}
              disabled={!draft.trim() || draft.trim() === datasetDir}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function SingleDatasetPicker({
  variant,
  datasets,
  value,
  onChange,
  labelSuffix,
}: {
  variant: Variant;
  datasets: Dataset[];
  value: string;
  onChange: (v: string) => void;
  labelSuffix: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label>Dataset {labelSuffix}</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger>
          <SelectValue placeholder="select a dataset…" />
        </SelectTrigger>
        <SelectContent>
          {datasets.map((d) => (
            <SelectItem key={d.name} value={d.name}>
              <span className="font-mono text-xs">{d.name}</span>
              {d.height && d.width && (
                <span className="ml-2 text-[10px] text-slate-500">
                  {d.height}×{d.width}
                </span>
              )}
              {d.episodes !== null && (
                <span className="ml-1 text-[10px] text-slate-500">
                  · {d.episodes} ep
                </span>
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

function NamesOnlyPicker({
  datasets,
  values,
  onChange,
  labelSuffix,
}: {
  datasets: Dataset[];
  values: string[];
  onChange: (v: string[]) => void;
  labelSuffix: React.ReactNode;
}) {
  const addRow = () => onChange([...values, ""]);
  const removeRow = (i: number) => onChange(values.filter((_, j) => j !== i));
  const updateRow = (i: number, name: string) => {
    const next = [...values];
    next[i] = name;
    onChange(next);
  };

  return (
    <div className="space-y-1.5">
      <Label>Datasets {labelSuffix}</Label>
      <div className="space-y-2">
        {values.map((name, i) => (
          <div key={i} className="flex items-center gap-2">
            <Select value={name} onValueChange={(v) => updateRow(i, v)}>
              <SelectTrigger className="flex-1">
                <SelectValue placeholder="dataset…" />
              </SelectTrigger>
              <SelectContent>
                {datasets.map((d) => (
                  <SelectItem key={d.name} value={d.name}>
                    <span className="font-mono text-xs">{d.name}</span>
                    {d.height && d.width && (
                      <span className="ml-2 text-[10px] text-slate-500">
                        {d.height}×{d.width}
                      </span>
                    )}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <ImmediateTooltip content="Remove">
              <button
                onClick={() => removeRow(i)}
                className="rounded p-1.5 text-slate-400 hover:bg-slate-100 hover:text-red-600 dark:hover:bg-slate-800"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </ImmediateTooltip>
          </div>
        ))}
        <Button variant="outline" size="sm" onClick={addRow} className="gap-1">
          <Plus className="h-3.5 w-3.5" /> add dataset
        </Button>
      </div>
      <p className="text-xs text-slate-500">
        N1.6 multi-task: list of dataset names. Weights and per-dataset cfg live
        in the modality config python file.
      </p>
    </div>
  );
}

function NameCfgWeightPicker({
  variant,
  datasets,
  values,
  onChange,
  labelSuffix,
}: {
  variant: Variant;
  datasets: Dataset[];
  values: string[];
  onChange: (v: string[]) => void;
  labelSuffix: React.ReactNode;
}) {
  const dataConfigDefault =
    variant.vars.DATA_CONFIG ?? "allex_thetwo_ck40_egostereo";
  const addRow = () => onChange([...values, `|${dataConfigDefault}|1.0`]);
  const removeRow = (i: number) => onChange(values.filter((_, j) => j !== i));
  const updateRow = (i: number, name: string, cfg: string, weight: string) => {
    const next = [...values];
    next[i] = `${name}|${cfg}|${weight}`;
    onChange(next);
  };

  return (
    <div className="space-y-1.5">
      <Label>Datasets {labelSuffix}</Label>
      <div className="space-y-2">
        {values.map((entry, i) => {
          const [name, cfg, weight] = entry.split("|");
          return (
            <div key={i} className="flex items-center gap-2">
              <Select
                value={name}
                onValueChange={(v) => updateRow(i, v, cfg, weight)}
              >
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="dataset…" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map((d) => (
                    <SelectItem key={d.name} value={d.name}>
                      <span className="font-mono text-xs">{d.name}</span>
                      {d.height && d.width && (
                        <span className="ml-2 text-[10px] text-slate-500">
                          {d.height}×{d.width}
                        </span>
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
              <ImmediateTooltip content="Remove">
                <button
                  onClick={() => removeRow(i)}
                  className="rounded p-1.5 text-slate-400 hover:bg-slate-100 hover:text-red-600 dark:hover:bg-slate-800"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </ImmediateTooltip>
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
