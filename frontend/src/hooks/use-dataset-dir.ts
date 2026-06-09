"use client";

import { useEffect, useState } from "react";

const PREFIX = "datasetDir";
const EVENT = "dataset-dir-change";

const SLURM_DEFAULT = "~/datasets";

function keyFor(cluster: string) {
  return `${PREFIX}:${cluster}`;
}

function defaultFor(defaultDir?: string) {
  return defaultDir || SLURM_DEFAULT;
}

function storedFor(cluster: string, defaultDir?: string) {
  if (typeof window === "undefined") return defaultFor(defaultDir);
  return localStorage.getItem(keyFor(cluster)) || defaultFor(defaultDir);
}

/** localStorage-backed value for "where should the dataset picker look on
 *  this cluster?", reactive within a single tab via a custom event so
 *  multiple components on the page update together when one writes. Stored
 *  per-cluster under `datasetDir:<cluster>`. */
export function useDatasetDir(cluster: string, defaultDir?: string): [string, (v: string) => void] {
  const [local, setLocal] = useState<Record<string, string>>({});
  const dir = local[cluster] ?? storedFor(cluster, defaultDir);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onCustom = (e: Event) => {
      const d = (e as CustomEvent<{ cluster: string; value: string }>).detail;
      if (d.cluster === cluster) {
        setLocal((prev) => ({ ...prev, [cluster]: d.value }));
      }
    };
    const onStorage = (e: StorageEvent) => {
      if (e.key === keyFor(cluster) && e.newValue) {
        setLocal((prev) => ({ ...prev, [cluster]: e.newValue ?? defaultFor(defaultDir) }));
      }
    };
    window.addEventListener(EVENT, onCustom);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(EVENT, onCustom);
      window.removeEventListener("storage", onStorage);
    };
  }, [cluster, defaultDir]);

  const set = (v: string) => {
    setLocal((prev) => ({ ...prev, [cluster]: v }));
    if (typeof window !== "undefined") {
      localStorage.setItem(keyFor(cluster), v);
      window.dispatchEvent(
        new CustomEvent(EVENT, { detail: { cluster, value: v } }),
      );
    }
  };

  return [dir, set];
}
