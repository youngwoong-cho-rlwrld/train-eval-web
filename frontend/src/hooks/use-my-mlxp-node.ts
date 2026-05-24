"use client";

import { useEffect, useState } from "react";

const KEY = "mlxpNode";
const EVENT = "mlxp-node-change";

/** localStorage-backed value for "my MLXP node", reactive within a single
 *  tab via a custom event so multiple components on the page update together
 *  when one of them writes a new value. */
export function useMyMlxpNode(defaultNode = ""): [string, (v: string) => void] {
  const [node, setLocal] = useState<string>(() => {
    if (typeof window === "undefined") return defaultNode;
    return localStorage.getItem(KEY) || defaultNode;
  });

  useEffect(() => {
    if (!defaultNode || typeof window === "undefined") return;
    if (!localStorage.getItem(KEY)) setLocal(defaultNode);
  }, [defaultNode]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onCustom = (e: Event) => setLocal((e as CustomEvent<string>).detail);
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY && e.newValue) setLocal(e.newValue);
    };
    window.addEventListener(EVENT, onCustom);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(EVENT, onCustom);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  const set = (v: string) => {
    setLocal(v);
    if (typeof window !== "undefined") {
      localStorage.setItem(KEY, v);
      window.dispatchEvent(new CustomEvent(EVENT, { detail: v }));
    }
  };

  return [node, set];
}
