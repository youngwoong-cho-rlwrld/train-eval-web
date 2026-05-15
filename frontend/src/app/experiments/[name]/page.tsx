"use client";

import { use } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type Variant } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function ExperimentDetail({ params }: { params: Promise<{ name: string }> }) {
  const { name } = use(params);
  const variant = useQuery({
    queryKey: ["variant", name],
    queryFn: () => api<Variant>(`/api/variants/${name}`),
  });

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="font-mono text-2xl font-semibold tracking-tight">{name}</h1>

      {variant.isLoading && <p className="mt-4 text-sm text-slate-500">Loading…</p>}
      {variant.error && <p className="mt-4 text-sm text-red-600">{(variant.error as Error).message}</p>}

      {variant.data && (
        <>
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Variables</CardTitle>
              <CardDescription>{variant.data.vars.TRAIN_NOTE ?? "—"}</CardDescription>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
                {Object.entries(variant.data.vars).map(([k, v]) => (
                  <div key={k} className="flex flex-col">
                    <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
                    <dd className="font-mono text-xs break-all">{v}</dd>
                  </div>
                ))}
              </dl>
            </CardContent>
          </Card>

          {Object.entries(variant.data.arrays).map(([k, items]) => (
            <Card key={k} className="mt-6">
              <CardHeader>
                <CardTitle>{k}</CardTitle>
              </CardHeader>
              <CardContent>
                {items.length === 0 ? (
                  <p className="text-sm text-slate-500">(empty)</p>
                ) : k === "EVAL_SETS" ? (
                  <div className="flex flex-wrap gap-2">
                    {items.map((it) => <Badge key={it} variant="secondary">{it}</Badge>)}
                  </div>
                ) : (
                  <ul className="space-y-1 font-mono text-xs">
                    {items.map((it, idx) => <li key={`${it}-${idx}`}>· {it}</li>)}
                  </ul>
                )}
              </CardContent>
            </Card>
          ))}

          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Raw config.sh</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="overflow-x-auto rounded-md bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-100">
                {variant.data.raw}
              </pre>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
