"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ExternalLink } from "lucide-react";
import { api, type ClusterEnvSettings, type WandbStatus } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { CopyButton } from "@/components/copy-button";
import {
  fieldsForClusterEnv,
  normalizeEnvDraft,
  parseEnvText,
  renderEnvText,
  sameEnvValues,
} from "@/lib/cluster-env";

export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
      <ClusterSettingsCard />
      <WandbCard />
    </div>
  );
}

function ClusterSettingsCard() {
  const qc = useQueryClient();
  const [edits, setEdits] = useState<Record<string, Record<string, string>>>({});

  const settings = useQuery({
    queryKey: ["cluster-settings"],
    queryFn: () => api<ClusterEnvSettings[]>("/api/cluster-settings"),
  });

  const save = useMutation({
    mutationFn: ({ name, envText }: { name: string; envText: string }) =>
      api<ClusterEnvSettings>(`/api/cluster-settings/${encodeURIComponent(name)}`, {
        method: "PUT",
        body: JSON.stringify({ env_text: envText }),
      }),
    onSuccess: (res) => {
      toast.success(`${res.name} settings saved`);
      setEdits((prev) => {
        const next = { ...prev };
        delete next[res.name];
        return next;
      });
      qc.setQueryData<ClusterEnvSettings[]>(["cluster-settings"], (old) =>
        old?.map((item) => item.name === res.name ? res : item) ?? [res],
      );
      qc.invalidateQueries({ queryKey: ["cluster-settings"] });
      qc.invalidateQueries({ queryKey: ["clusters"] });
      qc.invalidateQueries({ queryKey: ["partitions"] });
      qc.invalidateQueries({ queryKey: ["mlxp-settings"] });
      qc.invalidateQueries({ queryKey: ["mlxp-gpus"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle>Cluster env</CardTitle>
        <p className="text-sm text-slate-500">
          User-specific cluster paths and MLXP settings. Saved outside git under{" "}
          <code className="font-mono">~/.train-eval-web/clusters</code>.
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {settings.isLoading && <p className="text-sm text-slate-500">Loading cluster settings...</p>}
        {settings.error && (
          <p className="text-sm text-red-600 dark:text-red-400">
            {(settings.error as Error).message}
          </p>
        )}
        {settings.data?.map((item) => {
          const savedValues = parseEnvText(item.env_text);
          const draft = normalizeEnvDraft(edits[item.name], savedValues);
          const fields = fieldsForClusterEnv(item.name, savedValues, draft);
          const dirty = !sameEnvValues(savedValues, draft, fields);
          const pending = save.isPending && save.variables?.name === item.name;
          const envText = renderEnvText(fields, draft);
          return (
            <div key={item.name} className="space-y-3">
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <Label className="font-mono">{item.name}.env</Label>
                  {item.path && (
                    <div className="flex min-w-0 items-center gap-1 text-xs text-slate-500">
                      <code className="truncate font-mono">{item.path}</code>
                      <CopyButton value={item.path} title={`Copy ${item.name}.env path`} />
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    onClick={() =>
                      setEdits((prev) => {
                        const next = { ...prev };
                        delete next[item.name];
                        return next;
                      })
                    }
                    disabled={!dirty || pending}
                  >
                    Reset
                  </Button>
                  <Button
                    onClick={() => save.mutate({ name: item.name, envText })}
                    disabled={!dirty || pending}
                  >
                    {pending ? "Saving..." : "Save"}
                  </Button>
                </div>
              </div>
              <div className="divide-y divide-slate-100 rounded-md border border-slate-200 dark:divide-slate-900 dark:border-slate-800">
                {fields.map((field) => (
                  <div
                    key={field.key}
                    className="grid gap-3 px-3 py-3 md:grid-cols-[240px_minmax(0,1fr)]"
                  >
                    <div className="min-w-0">
                      <div className="truncate font-mono text-xs font-semibold">{field.key}</div>
                      <p className="mt-1 text-xs text-slate-500">{field.description}</p>
                    </div>
                    <Input
                      value={draft[field.key] ?? ""}
                      onChange={(e) =>
                        setEdits((prev) => ({
                          ...prev,
                          [item.name]: {
                            ...normalizeEnvDraft(prev[item.name], draft),
                            [field.key]: e.target.value,
                          },
                        }))
                      }
                      placeholder={savedValues[field.key] ?? ""}
                      className="font-mono text-xs"
                      autoComplete="off"
                    />
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function WandbCard() {
  const qc = useQueryClient();
  const [key, setKey] = useState("");
  const [projectDraft, setProjectDraft] = useState<string | null>(null);

  const status = useQuery({
    queryKey: ["wandb-status"],
    queryFn: () => api<WandbStatus>("/api/wandb/status"),
  });

  const login = useMutation({
    mutationFn: () =>
      api<WandbStatus>("/api/wandb/login", {
        method: "POST",
        body: JSON.stringify({ key }),
      }),
    onSuccess: (res) => {
      if (res.logged_in) {
        toast.success(`Signed in to wandb as ${res.entity ?? "(unknown)"}`);
        setKey("");
        qc.invalidateQueries({ queryKey: ["wandb-status"] });
      } else {
        toast.error(res.error ?? "Login failed");
      }
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const savedProject = status.data?.project ?? "";
  const project = projectDraft ?? savedProject;
  const projectDirty = project.trim() !== "" && project.trim() !== savedProject;

  const saveProject = useMutation({
    mutationFn: () =>
      api<WandbStatus>("/api/wandb/project", {
        method: "POST",
        body: JSON.stringify({ project }),
      }),
    onSuccess: (res) => {
      toast.success(`Project set to ${res.project}`);
      setProjectDraft(null);
      qc.setQueryData(["wandb-status"], res);
      qc.invalidateQueries({ queryKey: ["wandb-status"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Weights &amp; Biases
          {status.data?.logged_in ? (
            <Badge variant="success" className="text-[10px]">
              {status.data.entity ?? "connected"}
            </Badge>
          ) : (
            <Badge variant="warning" className="text-[10px]">
              not connected
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label className="flex items-center justify-between">
            <span>API key</span>
            {!status.data?.logged_in && (
              <a
                href="https://wandb.ai/authorize"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs text-blue-600 hover:underline"
              >
                get one <ExternalLink className="h-3 w-3" />
              </a>
            )}
          </Label>
          {status.data?.logged_in ? (
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Connected as{" "}
              <span className="font-mono">{status.data.entity ?? "(unknown)"}</span>.
            </p>
          ) : (
            <div className="flex gap-2">
              <Input
                type="password"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder="wandb api key"
                className="flex-1 font-mono text-xs"
                autoComplete="off"
              />
              <Button
                onClick={() => login.mutate()}
                disabled={!key.trim() || login.isPending}
              >
                {login.isPending ? "Saving…" : "Save"}
              </Button>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <Label>Project</Label>
          <div className="flex gap-2">
            <Input
              value={project}
              onChange={(e) => setProjectDraft(e.target.value)}
              placeholder="my project"
              className="flex-1 font-mono text-xs"
              autoComplete="off"
            />
            <Button
              onClick={() => saveProject.mutate()}
              disabled={!projectDirty || saveProject.isPending}
            >
              {saveProject.isPending ? "Saving…" : "Save"}
            </Button>
          </div>
          <p className="text-xs text-slate-500">
            Used for wandb API lookup and as the default project for training
            jobs.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
