"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { CopyButton } from "@/components/copy-button";

type FlagEntry = { flag: string; value: string };

export function ConfigCard({
  variantName,
  modalityConfigFile,
  flagsUrl,
  queryKey,
  cluster,
  phase,
  className,
}: {
  variantName: string;
  modalityConfigFile?: string | null;
  flagsUrl: string;
  queryKey: unknown[];
  cluster?: string;
  phase?: string;
  className?: string;
}) {
  const flags = useQuery({
    queryKey,
    queryFn: () => api<{ flags: FlagEntry[] }>(flagsUrl),
  });
  const wantsCheckpoint = phase === "eval" && !!cluster && cluster !== "mlxp";
  const selectedCkpt = useQuery({
    queryKey: ["selected-checkpoint", variantName, cluster],
    queryFn: () =>
      api<{ path: string | null; step: number | null }>(
        `/api/variants/${variantName}/selected-checkpoint?cluster=${cluster}`,
      ),
    enabled: wantsCheckpoint,
  });
  const configPath = `configs/experiments/${variantName}/config.sh`;
  const modalityPath = modalityConfigFile
    ? `configs/experiments/${variantName}/${modalityConfigFile}`
    : null;

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>
          Config{" "}
          <span className="text-xs font-normal text-slate-500">
            {variantName}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="divide-y divide-slate-100 dark:divide-slate-900">
          <ConfigPathRow label="config" value={configPath} />
          {modalityPath && (
            <ConfigPathRow label="modality" value={modalityPath} />
          )}
          {wantsCheckpoint && (
            <ConfigPathRow
              label="checkpoint"
              value={
                selectedCkpt.data?.path ??
                (selectedCkpt.isLoading ? "…" : "(none found — eval will fail)")
              }
            />
          )}
        </div>
        {flags.data && (
          <div className="divide-y divide-slate-100 dark:divide-slate-900">
            {flags.data.flags.map((f, i) => (
              <div
                key={`${f.flag}-${i}`}
                className="flex items-baseline gap-4 py-1.5 text-xs"
              >
                <code className="min-w-[220px] font-mono text-slate-600 dark:text-slate-300">
                  {f.flag}
                </code>
                <code className="flex-1 break-all font-mono text-slate-500">
                  {f.value || <span className="text-slate-400">(flag)</span>}
                </code>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ConfigPathRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <div className="min-w-[110px] text-xs uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className="flex-1 truncate font-mono text-xs">{value}</div>
      <CopyButton value={value} />
    </div>
  );
}
