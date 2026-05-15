"use client";

import { useEffect, useState } from "react";

const PREFIX = "datasetDir";
const EVENT = "dataset-dir-change";

const DEFAULTS: Record<string, string> = {
  mlxp: "/data/youngwoong/datasets",
};
const SLURM_DEFAULT = "~/datasets";

function keyFor(cluster: string) {
  return `${PREFIX}:${cluster}`;
}

function defaultFor(cluster: string) {
  return DEFAULTS[cluster] ?? SLURM_DEFAULT;
}

/** localStorage-backed value for "where should the dataset picker look on
 *  this cluster?", reactive within a single tab via a custom event so
 *  multiple components on the page update together when one writes. Stored
 *  per-cluster under `datasetDir:<cluster>`. */
export function useDatasetDir(cluster: string): [string, (v: string) => void] {
  const [dir, setLocal] = useState<string>(() => {
    if (typeof window === "undefined") return defaultFor(cluster);
    return localStorage.getItem(keyFor(cluster)) || defaultFor(cluster);
  });

  useEffect(() => {
    if (typeof window === "undefined") return;
    setLocal(localStorage.getItem(keyFor(cluster)) || defaultFor(cluster));
    const onCustom = (e: Event) => {
      const d = (e as CustomEvent<{ cluster: string; value: string }>).detail;
      if (d.cluster === cluster) setLocal(d.value);
    };
    const onStorage = (e: StorageEvent) => {
      if (e.key === keyFor(cluster) && e.newValue) setLocal(e.newValue);
    };
    window.addEventListener(EVENT, onCustom);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(EVENT, onCustom);
      window.removeEventListener("storage", onStorage);
    };
  }, [cluster]);

  const set = (v: string) => {
    setLocal(v);
    if (typeof window !== "undefined") {
      localStorage.setItem(keyFor(cluster), v);
      window.dispatchEvent(
        new CustomEvent(EVENT, { detail: { cluster, value: v } }),
      );
    }
  };

  return [dir, set];
}
