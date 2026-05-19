"use client";

import { use } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type Variant } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";

export default function ExperimentDetail({ params }: { params: Promise<{ name: string }> }) {
  const { name } = use(params);
  const variant = useQuery({
    queryKey: ["variant", name],
    queryFn: () => api<Variant>(`/api/variants/${name}`),
  });

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="font-mono text-2xl font-semibold tracking-tight">{name}</h1>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Variables</CardTitle>
          <CardDescription>{variant.data?.vars.TRAIN_NOTE ?? "config.sh scalars"}</CardDescription>
        </CardHeader>
        <CardContent>
          {variant.isLoading && <LoadingState label="Loading variables..." rows={5} />}
          {variant.error && <ErrorState message={(variant.error as Error).message} />}
          {variant.data && (
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
              {Object.entries(variant.data.vars).map(([k, v]) => (
                <div key={k} className="flex flex-col">
                  <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
                  <dd className="font-mono text-xs break-all">{v}</dd>
                </div>
              ))}
            </dl>
          )}
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Arrays</CardTitle>
        </CardHeader>
        <CardContent>
          {variant.isLoading && <LoadingState label="Loading arrays..." rows={3} />}
          {variant.error && <ErrorState message={(variant.error as Error).message} />}
          {variant.data && Object.keys(variant.data.arrays).length === 0 && (
            <EmptyState message="No arrays in this config." />
          )}
          {variant.data && Object.keys(variant.data.arrays).length > 0 && (
            <div className="space-y-4">
              {Object.entries(variant.data.arrays).map(([k, items]) => (
                <section key={k}>
                  <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">
                    {k}
                  </div>
                  {items.length === 0 ? (
                    <EmptyState message="(empty)" />
                  ) : k === "EVAL_SETS" ? (
                    <div className="flex flex-wrap gap-2">
                      {items.map((it) => <Badge key={it} variant="secondary">{it}</Badge>)}
                    </div>
                  ) : (
                    <ul className="space-y-1 font-mono text-xs">
                      {items.map((it, idx) => <li key={`${it}-${idx}`}>· {it}</li>)}
                    </ul>
                  )}
                </section>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Raw config.sh</CardTitle>
        </CardHeader>
        <CardContent>
          {variant.isLoading && <LoadingState label="Loading raw config..." rows={6} />}
          {variant.error && <ErrorState message={(variant.error as Error).message} />}
          {variant.data && (
            <pre className="overflow-x-auto rounded-md bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-100">
              {variant.data.raw}
            </pre>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
