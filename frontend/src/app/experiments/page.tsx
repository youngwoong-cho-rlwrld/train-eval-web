"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState, ErrorState, LoadingState } from "@/components/loading-state";

export default function ExperimentsPage() {
  const variants = useQuery({
    queryKey: ["variants"],
    queryFn: () => api<{ variants: string[] }>("/api/variants").then((d) => d.variants),
  });

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Experiments</h1>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>All experiments</CardTitle>
        </CardHeader>
        <CardContent>
          {variants.isLoading && <LoadingState label="Loading experiments..." rows={5} />}
          {variants.error && <ErrorState message={(variants.error as Error).message} />}
          {!variants.isLoading && !variants.error && variants.data?.length === 0 && (
            <EmptyState message="No experiments found." />
          )}
          {variants.data && (
            <ul className="divide-y divide-slate-100 dark:divide-slate-900">
              {variants.data.map((v) => (
                <li key={v} className="py-2">
                  <Link
                    href={`/experiments/${encodeURIComponent(v)}`}
                    className="font-mono text-sm text-blue-600 hover:underline"
                  >
                    {v}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
