"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function ExperimentsPage() {
  const variants = useQuery({
    queryKey: ["variants"],
    queryFn: () => api<{ variants: string[] }>("/api/variants").then((d) => d.variants),
  });

  return (
    <div className="mx-auto max-w-7xl px-8 py-12">
      <h1 className="text-2xl font-semibold tracking-tight">Experiments</h1>
      <p className="mt-2 text-slate-600 dark:text-slate-400">
        Variants under <code className="text-xs">configs/experiments/</code> in this repo. Click to view.
      </p>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>All variants</CardTitle>
        </CardHeader>
        <CardContent>
          {variants.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
          {variants.data && (
            <ul className="divide-y divide-slate-100 dark:divide-slate-900">
              {variants.data.map((v) => (
                <li key={v} className="py-2">
                  <Link
                    href={`/experiments/${v}`}
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
