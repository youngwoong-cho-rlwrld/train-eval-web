"use client";

import { use, useState } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ArrowLeft } from "lucide-react";
import {
  api,
  type ExperimentFiles,
  type SaveExperimentFilesResponse,
} from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";

export default function ExperimentDetail({ params }: { params: Promise<{ name: string }> }) {
  const { name } = use(params);
  const files = useQuery({
    queryKey: ["experiment-files", name],
    queryFn: () => api<ExperimentFiles>(`/api/variants/${encodeURIComponent(name)}/files`),
  });

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <Link
        href="/experiments"
        className="inline-flex items-center gap-1 text-sm text-slate-500 transition-colors hover:text-slate-900 dark:hover:text-slate-50"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to experiments
      </Link>

      <h1 className="mt-4 font-mono text-2xl font-semibold tracking-tight">{name}</h1>
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Experiment files</CardTitle>
          <CardDescription>
            Every experiment has <span className="font-mono">config.sh</span> plus one model-facing schema file.
            N1.5 uses <span className="font-mono">data_config.yaml</span> because <span className="font-mono">gr00t_finetune.py</span> consumes <span className="font-mono">--data-config</span> YAML entries that point to <span className="font-mono">DATA_CONFIG_MAP</span>.
            N1.6 and Physixel use a Python modality config because <span className="font-mono">launch_finetune.py</span> and <span className="font-mono">eval_allex.py</span> consume registered modality schemas.
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="mt-6">
        {files.isLoading && <LoadingState label="Loading experiment files..." rows={8} />}
        {files.error && <ErrorState message={(files.error as Error).message} />}
        {files.data && <ExperimentFilesEditor name={name} files={files.data} />}
      </div>
    </div>
  );
}

function ExperimentFilesEditor({
  name,
  files,
}: {
  name: string;
  files: ExperimentFiles;
}) {
  const qc = useQueryClient();
  const [configTitle, setConfigTitle] = useState(files.config.title);
  const [configContent, setConfigContent] = useState(files.config.content);
  const [secondTitle, setSecondTitle] = useState(files.second_file.title);
  const [secondContent, setSecondContent] = useState(files.second_file.content);

  const save = useMutation({
    mutationFn: () =>
      api<SaveExperimentFilesResponse>(`/api/variants/${encodeURIComponent(name)}/files`, {
        method: "PUT",
        body: JSON.stringify({
          config_title: configTitle,
          config_content: configContent,
          second_title: secondTitle,
          second_content: secondContent,
        }),
      }),
    onSuccess: (res) => {
      setConfigTitle(res.config.title);
      setConfigContent(res.config.content);
      setSecondTitle(res.second_file.title);
      setSecondContent(res.second_file.content);
      qc.setQueryData(["experiment-files", name], res);
      qc.invalidateQueries({ queryKey: ["variant", name] });
      qc.invalidateQueries({ queryKey: ["variant-data-interface", name] });
      toast.success(res.saved_version_path ? "Saved files and archived previous version" : "Files unchanged");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const restore = useMutation({
    mutationFn: (version: string) =>
      api<SaveExperimentFilesResponse>(
        `/api/variants/${name}/files/versions/${encodeURIComponent(version)}/restore`,
        { method: "POST" },
      ),
    onSuccess: (res) => {
      setConfigTitle(res.config.title);
      setConfigContent(res.config.content);
      setSecondTitle(res.second_file.title);
      setSecondContent(res.second_file.content);
      qc.setQueryData(["experiment-files", name], res);
      qc.invalidateQueries({ queryKey: ["variant", name] });
      qc.invalidateQueries({ queryKey: ["variant-data-interface", name] });
      toast.success("Restored previous version");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const dirty =
    configTitle !== files.config.title ||
    configContent !== files.config.content ||
    secondTitle !== files.second_file.title ||
    secondContent !== files.second_file.content;
  const configTitleValid = configTitle.trim() === "config.sh";

  function updateSecondTitle(next: string) {
    setSecondTitle(next);
    if (next.trim()) {
      setConfigContent((current) =>
        rewriteSecondFileRef(current, next.trim(), files.second_file.kind),
      );
    }
  }

  function restoreVersion(version: string) {
    const message = dirty
      ? "Restore this previous version and discard unsaved edits? The current files will be archived first."
      : "Restore this previous version? The current files will be archived first.";
    if (!window.confirm(message)) return;
    restore.mutate(version);
  }

  return (
    <div className="space-y-6">
      <FileCard
        label={files.config.label}
        purpose={files.config.purpose}
        title={configTitle}
        titleInvalid={!configTitleValid}
        path={files.config.path}
        content={configContent}
        onTitleChange={setConfigTitle}
        onContentChange={setConfigContent}
      />
      <FileCard
        label={files.second_file.label}
        purpose={files.second_file.purpose}
        title={secondTitle}
        path={files.second_file.path}
        content={secondContent}
        onTitleChange={updateSecondTitle}
        onContentChange={setSecondContent}
      />
      <div className="flex items-center justify-between gap-4">
        <VersionList
          versions={files.versions}
          restoringVersion={restore.isPending ? restore.variables : undefined}
          onRestore={restoreVersion}
        />
        <Button
          onClick={() => save.mutate()}
          disabled={!dirty || !configTitleValid || save.isPending || restore.isPending}
        >
          {save.isPending ? "Saving..." : "Save files"}
        </Button>
      </div>
    </div>
  );
}

function FileCard({
  label,
  purpose,
  title,
  titleInvalid = false,
  path,
  content,
  onTitleChange,
  onContentChange,
}: {
  label: string;
  purpose: string;
  title: string;
  titleInvalid?: boolean;
  path: string;
  content: string;
  onTitleChange: (value: string) => void;
  onContentChange: (value: string) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{label}</CardTitle>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          {purpose}
        </p>
        <CardDescription className="font-mono text-xs">{path}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <Label>Title</Label>
          <Input
            value={title}
            onChange={(e) => onTitleChange(e.target.value)}
            className={
              titleInvalid
                ? "font-mono text-xs border-red-500 focus-visible:ring-red-500"
                : "font-mono text-xs"
            }
          />
        </div>
        <div className="space-y-1.5">
          <Label>Content</Label>
          <textarea
            value={content}
            onChange={(e) => onContentChange(e.target.value)}
            spellCheck={false}
            className="min-h-[28rem] w-full resize-y rounded-md border border-slate-200 bg-white p-3 font-mono text-xs leading-relaxed outline-none ring-offset-white focus-visible:ring-2 focus-visible:ring-slate-950 focus-visible:ring-offset-2 dark:border-slate-800 dark:bg-slate-950 dark:ring-offset-slate-950 dark:focus-visible:ring-slate-300"
          />
        </div>
      </CardContent>
    </Card>
  );
}

function VersionList({
  versions,
  restoringVersion,
  onRestore,
}: {
  versions: ExperimentFiles["versions"];
  restoringVersion?: string;
  onRestore: (version: string) => void;
}) {
  if (versions.length === 0) {
    return <EmptyState message="No previous file versions." />;
  }
  return (
    <div className="min-w-0 text-xs text-slate-500">
      <div className="mb-1 font-medium uppercase tracking-wide">Previous versions</div>
      <ul className="space-y-1">
        {versions.slice(0, 5).map((version) => (
          <li
            key={version.path}
            className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1"
          >
            <span className="font-mono">{version.created_at}</span>
            <span className="min-w-0 font-mono text-slate-400">
              {version.files.join(", ")}
            </span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => onRestore(version.created_at)}
              disabled={restoringVersion === version.created_at}
            >
              {restoringVersion === version.created_at ? "Restoring..." : "Restore"}
            </Button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function rewriteSecondFileRef(config: string, title: string, kind: string) {
  const key = kind === "data_config_yaml" ? "TRAIN_DATA_CONFIG" : "TRAIN_MODALITY_CONFIG";
  const line = `${key}=${title}`;
  const pattern = new RegExp(`^(?:export\\s+)?${key}=.*$`, "m");
  if (pattern.test(config)) return config.replace(pattern, line);
  const suffix = config.endsWith("\n") ? "" : "\n";
  return `${config}${suffix}\n${line}\n`;
}
