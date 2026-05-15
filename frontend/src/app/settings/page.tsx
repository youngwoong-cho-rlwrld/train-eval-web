"use client";

import { useState } from "react";
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
      <CardContent className="space-y-3">
        <div className="space-y-2">
          <Label className="flex items-center justify-between">
            <span>API key</span>
            <a
              href="https://wandb.ai/authorize"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs text-blue-600 hover:underline"
            >
              get one <ExternalLink className="h-3 w-3" />
            </a>
          </Label>
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
        </div>
      </CardContent>
    </Card>
  );
}
