"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ExternalLink } from "lucide-react";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type WandbStatus = {
  logged_in: boolean;
  entity: string | null;
  project: string;
  error: string | null;
};

export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
      <WandbCard />
    </div>
  );
}

function WandbCard() {
  const qc = useQueryClient();
  const [key, setKey] = useState("");
  const [project, setProject] = useState("");

  const status = useQuery({
    queryKey: ["wandb-status"],
    queryFn: () => api<WandbStatus>("/api/wandb/status"),
  });

  // Keep the project input in sync with the saved value, but don't clobber
  // a user edit in progress.
  useEffect(() => {
    if (status.data && project === "") setProject(status.data.project);
  }, [status.data, project]);

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

  const saveProject = useMutation({
    mutationFn: () =>
      api<WandbStatus>("/api/wandb/project", {
        method: "POST",
        body: JSON.stringify({ project }),
      }),
    onSuccess: (res) => {
      toast.success(`Project set to ${res.project}`);
      qc.invalidateQueries({ queryKey: ["wandb-status"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const savedProject = status.data?.project ?? "";
  const projectDirty = project.trim() !== "" && project.trim() !== savedProject;

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
              onChange={(e) => setProject(e.target.value)}
              placeholder="finetune-gr00t-n1d6"
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
            Used for both the wandb API lookup and the{" "}
            <code>--wandb-project</code> flag passed to MLXP jobs.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
