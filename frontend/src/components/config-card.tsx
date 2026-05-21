"use client";

import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { CircleHelp } from "lucide-react";
import { api, type DataInterfaceSummary } from "@/lib/api";
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
type FlagRow = { key: string; entry: FlagEntry };
export type FlagEditor =
  | ReactNode
  | {
      content: ReactNode;
      wide?: boolean;
    };
export type ExtraFlagRow = {
  key: string;
  flag: string;
  value: string;
  editor?: FlagEditor;
};

export function ConfigCard({
  variantName,
  flagsUrl,
  queryKey,
  cluster,
  phase,
  checkpointOverride,
  checkpointOverrideExists,
  effectiveConfigText,
  effectiveConfigPath,
  modelLabel,
  modelRepoPath,
  modelRepoError,
  modelRepoMessage,
  modelRepoChecking = false,
  effectiveConfigLoading = false,
  effectiveConfigError,
  flagsOverride,
  flagEditors,
  extraFlagRows = [],
  showCheckpointPathRow = true,
  loading = false,
  error,
  className,
}: {
  variantName: string | null;
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
  modelLabel?: string | null;
  modelRepoPath?: string | null;
  modelRepoError?: string | null;
  modelRepoMessage?: string | null;
  modelRepoChecking?: boolean;
  effectiveConfigLoading?: boolean;
  effectiveConfigError?: Error | null;
  flagsOverride?: FlagEntry[] | null;
  flagEditors?: Record<string, FlagEditor>;
  extraFlagRows?: ExtraFlagRow[];
  showCheckpointPathRow?: boolean;
  loading?: boolean;
  error?: Error | null;
  className?: string;
}) {
  const flags = useQuery({
    queryKey,
    queryFn: () => api<{ flags: FlagEntry[] }>(flagsUrl),
    enabled: !!variantName && !loading && !error,
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
    enabled: showCheckpointPathRow && wantsCheckpoint && !overridePath && !loading && !error,
  });
  const configPath = variantName
    ? `configs/experiments/${variantName}/config.sh`
    : null;
  const shownConfigPath = effectiveConfigPath || configPath;
  const shownFlags = resolveShownFlags({
    override: flagsOverride,
    loaded: flags.data?.flags,
    editors: flagEditors,
  });
  const flagRows = shownFlags ? toFlagRows(shownFlags) : undefined;
  const flagsLoading = !flagsOverride && flags.isLoading;
  const flagsError = !flagsOverride ? (flags.error as Error | null) : null;

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>
          config.sh{" "}
          <span className="text-xs font-normal text-slate-500">
            {loading
              ? "(loading experiment...)"
              : variantName ?? "(experiment unknown - couldn't parse job_name)"}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading && <LoadingState label="Loading config.sh..." />}
        {!loading && error && <ErrorState message={error.message} />}
        {!loading && !error && !variantName && (
          <EmptyState message="config.sh unavailable because the experiment could not be resolved." />
        )}
        {!loading && !error && variantName && (
          <div className="divide-y divide-slate-100 dark:divide-slate-900">
            {shownConfigPath && (
              <ConfigPathRow
                label={effectiveConfigPath ? "effective config.sh" : "config.sh"}
                value={shownConfigPath}
                labelHelp={
                  effectiveConfigPath
                    ? "The config.sh that will be used for this submission after applying UI overrides."
                    : undefined
                }
                valueTooltip={effectiveConfigPath ? shownConfigPath : undefined}
              />
            )}
            {effectiveConfigPath && configPath && (
              <ConfigPathRow
                label="source config.sh"
                value={configPath}
                labelHelp="The original config.sh in this repo. It is the base file before submit-time overrides are applied."
                valueTooltip={configPath}
              />
            )}
            {modelLabel && (
              <ConfigPathRow
                label="model"
                value={modelLabel}
                labelHelp="The model registry entry selected by this experiment."
              />
            )}
            {(modelRepoPath || modelRepoError || modelRepoChecking) && (
              <ConfigPathRow
                label="model repo"
                value={modelRepoPath ?? "(not configured)"}
                labelHelp="The code repository used to run this job. This is different from --base-model-path, which is the pretrained checkpoint argument."
                valueTooltip={modelRepoPath ?? undefined}
                tone={modelRepoError ? "error" : "default"}
                message={
                  modelRepoError ??
                  (modelRepoChecking ? "Checking model repository..." : modelRepoMessage)
                }
                copyValue={modelRepoPath ?? null}
              />
            )}
            {wantsCheckpoint && showCheckpointPathRow && (
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
        {!loading && !error && variantName && flagsLoading && (
          <LoadingState label="Loading flags..." rows={2} />
        )}
        {!loading && !error && variantName && flagsError && (
          <ErrorState message={flagsError.message} />
        )}
        {!loading && !error && variantName && flagRows && flagRows.length === 0 && extraFlagRows.length === 0 && (
          <EmptyState message="No flags resolved for this job." />
        )}
        {!loading && !error && variantName && flagRows && (flagRows.length > 0 || extraFlagRows.length > 0) && (
          <div className="space-y-2">
            <div className="overflow-x-auto pb-1">
              <div className="min-w-[42rem]">
                <div className="grid grid-cols-[minmax(8rem,12rem)_minmax(10rem,1fr)_minmax(12rem,18rem)] gap-3 border-b border-slate-100 pb-1 text-xs font-medium uppercase tracking-wide text-slate-500 dark:border-slate-900">
                  <div>Setting</div>
                  <div>Effective</div>
                  <div>Override</div>
                </div>
                <div className="divide-y divide-slate-100 dark:divide-slate-900">
                  {flagRows.map(({ key, entry }) => (
                    <ConfigFlagRow
                      key={key}
                      flag={entry.flag}
                      value={entry.value}
                      editor={flagEditors?.[entry.flag]}
                    />
                  ))}
                  {extraFlagRows.map((row) => (
                    <ConfigFlagRow
                      key={row.key}
                      flag={row.flag}
                      value={row.value}
                      editor={row.editor}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
        {!loading && !error && variantName && effectiveConfigLoading && (
          <LoadingState label="Rendering config.sh preview..." rows={3} />
        )}
        {!loading && !error && variantName && effectiveConfigError && (
          <ErrorState message={effectiveConfigError.message} />
        )}
        {!loading && !error && variantName && effectiveConfigText && (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Effective config.sh preview
              </div>
              <CopyButton value={effectiveConfigText} />
            </div>
            <pre className="max-h-80 overflow-auto rounded border border-slate-200 bg-slate-50 p-3 text-xs dark:border-slate-800 dark:bg-slate-950">
              {effectiveConfigText}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function resolveShownFlags({
  override,
  loaded,
  editors,
}: {
  override?: FlagEntry[] | null;
  loaded?: FlagEntry[];
  editors?: Record<string, FlagEditor>;
}): FlagEntry[] | undefined {
  if (override) return override;
  if (loaded) return loaded;
  const editableFlags = editors ? Object.keys(editors) : [];
  return editableFlags.length > 0
    ? editableFlags.map((flag) => ({ flag, value: "" }))
    : undefined;
}

function toFlagRows(flags: FlagEntry[]): FlagRow[] {
  const seen = new Map<string, number>();
  return flags.map((entry) => {
    const occurrence = seen.get(entry.flag) ?? 0;
    seen.set(entry.flag, occurrence + 1);
    return {
      key: `${entry.flag}:${occurrence}`,
      entry,
    };
  });
}

function normalizeEditor(editor?: FlagEditor):
  | { content: ReactNode; wide: boolean }
  | null {
  if (!editor) return null;
  if (
    typeof editor === "object" &&
    editor !== null &&
    "content" in editor
  ) {
    return {
      content: editor.content,
      wide: Boolean(editor.wide),
    };
  }
  return { content: editor, wide: false };
}

export function DataInterfaceCard({
  variantName,
  loading = false,
  error,
  className,
}: {
  variantName: string | null;
  loading?: boolean;
  error?: Error | null;
  className?: string;
}) {
  const dataInterface = useQuery({
    queryKey: ["variant-data-interface", variantName],
    queryFn: () => api<DataInterfaceSummary>(`/api/variants/${variantName}/data-interface`),
    enabled: !!variantName && !loading && !error,
  });

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          modality.py{" "}
          <ImmediateTooltip content="GR00T calls this the modality config. It defines the model-facing video/state/action/language schema selected by TRAIN_MODALITY_CONFIG in config.sh.">
            <CircleHelp className="h-4 w-4 text-slate-400" />
          </ImmediateTooltip>
          <span className="text-xs font-normal text-slate-500">
            {loading
              ? "(loading experiment...)"
              : variantName ?? "(experiment unknown - couldn't parse job_name)"}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading && <LoadingState label="Loading modality.py..." />}
        {!loading && error && <ErrorState message={error.message} />}
        {!loading && !error && !variantName && (
          <EmptyState message="modality.py unavailable because the experiment could not be resolved." />
        )}
        {!loading && !error && variantName && (
          <DataInterfaceContent
            loading={dataInterface.isLoading}
            error={dataInterface.error as Error | null}
            summary={dataInterface.data ?? null}
          />
        )}
      </CardContent>
    </Card>
  );
}

function DataInterfaceContent({
  loading,
  error,
  summary,
}: {
  loading: boolean;
  error: Error | null;
  summary: DataInterfaceSummary | null;
}) {
  if (loading) return <LoadingState label="Loading data interface..." rows={3} />;
  if (error) return <ErrorState message={error.message} />;
  if (!summary) return null;
  if (summary.error) {
    return (
      <div className="space-y-2">
        <EmptyState message={summary.error} />
        {summary.text && <ModalityPreview text={summary.text} />}
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="divide-y divide-slate-100 dark:divide-slate-900">
        {summary.path && (
          <ConfigPathRow
            label="modality.py"
            value={summary.path}
            labelHelp="The Python file referenced by TRAIN_MODALITY_CONFIG in config.sh."
            valueTooltip={summary.path}
          />
        )}
        {summary.config_name && (
          <ConfigPathRow
            label="registered schema"
            value={summary.config_name}
            labelHelp="The Python dictionary passed to GR00T's register_modality_config."
          />
        )}
        {summary.embodiment_tag && (
          <ConfigPathRow
            label="embodiment tag"
            value={summary.embodiment_tag}
            labelHelp="The GR00T embodiment tag used when registering this data interface."
          />
        )}
        {summary.action_horizon != null && (
          <ConfigPathRow
            label="action horizon"
            value={String(summary.action_horizon)}
            labelHelp="Parsed from action.delta_indices in the registered modality config."
          />
        )}
      </div>
      {summary.text && <ModalityPreview text={summary.text} />}
    </div>
  );
}

function ModalityPreview({ text }: { text: string }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
          modality.py preview
        </div>
        <CopyButton value={text} />
      </div>
      <pre className="max-h-80 overflow-auto rounded border border-slate-200 bg-slate-50 p-3 text-xs dark:border-slate-800 dark:bg-slate-950">
        {text}
      </pre>
    </div>
  );
}

function ConfigFlagRow({
  flag,
  value,
  editor,
}: {
  flag: string;
  value: string;
  editor?: FlagEditor;
}) {
  const normalizedEditor = normalizeEditor(editor);
  return (
    <div className="grid grid-cols-[minmax(8rem,12rem)_minmax(10rem,1fr)_minmax(12rem,18rem)] items-start gap-3 py-2 text-xs">
      <code className="font-mono text-slate-600 dark:text-slate-300">
        {flag}
      </code>
      <code className="break-all font-mono text-slate-500">
        {value || <span className="text-slate-400">(flag)</span>}
      </code>
      {normalizedEditor && !normalizedEditor.wide ? (
        <div className="min-w-0">{normalizedEditor.content}</div>
      ) : (
        <div className="text-slate-400">
          {normalizedEditor ? "" : "read-only"}
        </div>
      )}
      {normalizedEditor?.wide && (
        <div className="col-span-3 rounded-md border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950">
          {normalizedEditor.content}
        </div>
      )}
    </div>
  );
}

function ConfigPathRow({
  label,
  value,
  labelHelp,
  valueTooltip,
  message,
  copyValue,
  tone = "default",
}: {
  label: string;
  value: string;
  labelHelp?: string;
  valueTooltip?: string;
  message?: string | null;
  copyValue?: string | null;
  tone?: "default" | "error";
}) {
  const valueClass =
    tone === "error"
      ? "flex-1 truncate font-mono text-xs text-red-600 dark:text-red-400"
      : "flex-1 truncate font-mono text-xs";
  const messageClass =
    tone === "error"
      ? "mt-1 pl-[calc(110px+1rem)] text-xs text-red-600 dark:text-red-400"
      : "mt-1 pl-[calc(110px+1rem)] text-xs text-slate-500 dark:text-slate-400";
  const shownCopyValue = copyValue === undefined ? value : copyValue;
  return (
    <div className="py-2">
      <div className="flex items-center justify-between gap-4">
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
        {shownCopyValue ? (
          <CopyButton value={shownCopyValue} />
        ) : (
          <div className="h-8 w-8" />
        )}
      </div>
      {message && <div className={messageClass}>{message}</div>}
    </div>
  );
}
