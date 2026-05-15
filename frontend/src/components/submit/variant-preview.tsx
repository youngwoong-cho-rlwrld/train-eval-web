"use client";

import type { Variant } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const KEY_ORDER = [
  "MODEL_VERSION",
  "MAX_STEPS",
  "SAVE_STEPS",
  "TRAIN_NUM_GPUS",
  "TRAIN_BATCH_SIZE",
  "DATA_DIR",
  "DATASET_NAME",
  "DATA_CONFIG",
  "TASK_NAME",
  "N_EPISODES",
  "N_RUNS",
  "EXECUTION_HORIZON",
  "MAX_EPISODE_STEPS",
  "TRAIN_NOTE",
];

export function VariantPreview({ variant }: { variant: Variant }) {
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
              <dt className="text-xs uppercase tracking-wide text-slate-500">
                {k}
              </dt>
              <dd className="font-mono">{variant.vars[k]}</dd>
            </div>
          ))}
        </dl>
        {variant.arrays.DATASETS && (
          <ArrayList name="DATASETS" items={variant.arrays.DATASETS} />
        )}
        {variant.arrays.TRAIN_DATASET_NAMES && (
          <ArrayList
            name="TRAIN_DATASET_NAMES"
            items={variant.arrays.TRAIN_DATASET_NAMES}
          />
        )}
        {variant.arrays.TASKS && (
          <ArrayList name="TASKS" items={variant.arrays.TASKS} />
        )}
        {variant.arrays.EVAL_SETS && (
          <div className="flex items-center gap-2 text-xs">
            <span className="uppercase tracking-wide text-slate-500">
              EVAL_SETS:
            </span>
            {variant.arrays.EVAL_SETS.map((es) => (
              <Badge key={es} variant="secondary">
                {es}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ArrayList({ name, items }: { name: string; items: string[] }) {
  return (
    <div>
      <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">
        {name}
      </div>
      <ul className="space-y-1 font-mono text-xs">
        {items.map((d) => (
          <li key={d}>· {d}</li>
        ))}
      </ul>
    </div>
  );
}
