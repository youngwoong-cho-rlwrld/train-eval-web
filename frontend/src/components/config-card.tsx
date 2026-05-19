"use client";

import { useQuery } from "@tanstack/react-query";
import { CircleHelp } from "lucide-react";
import { api } from "@/lib/api";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { CopyButton } from "@/components/copy-button";
import { ImmediateTooltip } from "@/components/immediate-tooltip";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";

type FlagEntry = { flag: string; value: string };

export function ConfigCard({
  variantName,
  modalityConfigFile,
  flagsUrl,
  queryKey,
  cluster,
  phase,
  checkpointOverride,
  checkpointOverrideExists,
  effectiveConfigText,
  effectiveConfigPath,
  effectiveConfigLoading = false,
  effectiveConfigError,
  flagsOverride,
  loading = false,
  error,
  className,
}: {
  variantName: string | null;
  modalityConfigFile?: string | null;
  flagsUrl: string;
  queryKey: unknown[];
  cluster?: string;
  phase?: string;
  checkpointOverride?: string | null;
  // null = unknown / not yet checked; true/false = result from
  // /api/clusters/<c>/path-exists. Owned by the parent so it can also
  // gate the Submit button.
  checkpointOverrideExists?: boolean | null;
  effectiveConfigText?: string | null;
  effectiveConfigPath?: string | null;
  effectiveConfigLoading?: boolean;
  effectiveConfigError?: Error | null;
  flagsOverride?: FlagEntry[] | null;
  loading?: boolean;
  error?: Error | null;
  className?: string;
}) {
  const flags = useQuery({
    queryKey,
    queryFn: () => api<{ flags: FlagEntry[] }>(flagsUrl),
    enabled: !!variantName && !loading && !error && !flagsOverride,
  });
  const wantsCheckpoint =
    !!variantName && phase === "eval" && !!cluster && cluster !== "mlxp";
  const overridePath = checkpointOverride?.trim() || null;
  const overrideMissing = overridePath && checkpointOverrideExists === false;
  const overrideChecking = overridePath && checkpointOverrideExists === null;
  const selectedCkpt = useQuery({
    queryKey: ["selected-checkpoint", variantName, cluster],
    queryFn: () =>
      api<{ path: string | null; step: number | null }>(
        `/api/variants/${variantName}/selected-checkpoint?cluster=${cluster}`,
      ),
    enabled: wantsCheckpoint && !overridePath && !loading && !error,
  });
  const configPath = variantName
    ? `configs/experiments/${variantName}/config.sh`
    : null;
  const shownConfigPath = effectiveConfigPath || configPath;
  const shownFlags = flagsOverride ?? flags.data?.flags;
  const flagsLoading = !flagsOverride && flags.isLoading;
  const flagsError = !flagsOverride ? (flags.error as Error | null) : null;
  const modalityPath =
    variantName && modalityConfigFile
      ? `configs/experiments/${variantName}/${modalityConfigFile}`
      : null;

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>
          Config{" "}
          <span className="text-xs font-normal text-slate-500">
            {loading
              ? "(loading variant...)"
              : variantName ?? "(variant unknown — couldn't parse job_name)"}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading && <LoadingState label="Loading config..." />}
        {!loading && error && <ErrorState message={error.message} />}
        {!loading && !error && !variantName && (
          <EmptyState message="Config unavailable because the variant could not be resolved." />
        )}
        {!loading && !error && variantName && (
          <div className="divide-y divide-slate-100 dark:divide-slate-900">
            {shownConfigPath && (
              <ConfigPathRow
                label={effectiveConfigPath ? "effective config" : "config"}
                value={shownConfigPath}
                labelHelp={
                  effectiveConfigPath
                    ? "The config that will be used for this submission after applying UI overrides."
                    : undefined
                }
                valueTooltip={effectiveConfigPath ? shownConfigPath : undefined}
              />
            )}
            {effectiveConfigPath && configPath && (
              <ConfigPathRow
                label="source config"
                value={configPath}
                labelHelp="The original config.sh in this repo. It is the base file before submit-time overrides are applied."
                valueTooltip={configPath}
              />
            )}
            {modalityPath && (
              <ConfigPathRow
                label="modality"
                value={modalityPath}
                valueTooltip={modalityPath}
              />
            )}
            {wantsCheckpoint && (
              <ConfigPathRow
                label="checkpoint"
                tone={overrideMissing ? "error" : "default"}
                value={
                  overridePath
                    ? overrideChecking
                      ? `${overridePath}  (checking...)`
                      : overrideMissing
                        ? `${overridePath}  (not found on ${cluster})`
                        : overridePath
                    : selectedCkpt.data?.path ??
                      (selectedCkpt.isLoading ? "..." : "(none found - eval will fail)")
                }
              />
            )}
          </div>
        )}
        {!loading && !error && variantName && effectiveConfigLoading && (
          <LoadingState label="Rendering config preview..." rows={3} />
        )}
        {!loading && !error && variantName && effectiveConfigError && (
          <ErrorState message={effectiveConfigError.message} />
        )}
        {!loading && !error && variantName && effectiveConfigText && (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Effective config preview
              </div>
              <CopyButton value={effectiveConfigText} />
            </div>
            <pre className="max-h-80 overflow-auto rounded border border-slate-200 bg-slate-50 p-3 text-xs dark:border-slate-800 dark:bg-slate-950">
              {effectiveConfigText}
            </pre>
          </div>
        )}
        {!loading && !error && variantName && flagsLoading && (
          <LoadingState label="Loading flags..." rows={2} />
        )}
        {!loading && !error && variantName && flagsError && (
          <ErrorState message={flagsError.message} />
        )}
        {!loading && !error && variantName && shownFlags && shownFlags.length === 0 && (
          <EmptyState message="No flags resolved for this job." />
        )}
        {!loading && !error && variantName && shownFlags && shownFlags.length > 0 && (
          <div className="divide-y divide-slate-100 dark:divide-slate-900">
            {shownFlags.map((f, i) => (
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

function ConfigPathRow({
  label,
  value,
  labelHelp,
  valueTooltip,
  tone = "default",
}: {
  label: string;
  value: string;
  labelHelp?: string;
  valueTooltip?: string;
  tone?: "default" | "error";
}) {
  const valueClass =
    tone === "error"
      ? "flex-1 truncate font-mono text-xs text-red-600 dark:text-red-400"
      : "flex-1 truncate font-mono text-xs";
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <div className="flex min-w-[110px] items-center gap-1.5 text-xs uppercase tracking-wide text-slate-500">
        <span>{label}</span>
        {labelHelp && (
          <ImmediateTooltip content={labelHelp}>
            <CircleHelp className="h-3.5 w-3.5 text-slate-400" />
          </ImmediateTooltip>
        )}
      </div>
      <ImmediateTooltip
        content={valueTooltip}
        className="min-w-0 flex-1"
        contentClassName="font-mono"
      >
        <span className={valueClass}>{value}</span>
      </ImmediateTooltip>
      <CopyButton value={value} />
    </div>
  );
}
