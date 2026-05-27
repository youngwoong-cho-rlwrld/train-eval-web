"use client";

import { useCallback, useSyncExternalStore } from "react";

const KEY = "mlxpNode";
const EVENT = "mlxp-node-change";

function getSnapshot() {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(KEY);
}

function getServerSnapshot() {
  return null;
}

function subscribe(onStoreChange: () => void) {
  if (typeof window === "undefined") return () => {};
  const onCustom = () => onStoreChange();
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) onStoreChange();
  };
  window.addEventListener(EVENT, onCustom);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(EVENT, onCustom);
    window.removeEventListener("storage", onStorage);
  };
}

/** localStorage-backed value for "my MLXP node", reactive within a single
 *  tab via a custom event so multiple components on the page update together
 *  when one of them writes a new value. */
export function useMyMlxpNode(defaultNode = ""): [string, (v: string) => void] {
  const localNode = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  );
  const node = localNode ?? defaultNode;

  const set = useCallback((v: string) => {
    if (typeof window !== "undefined") {
      localStorage.setItem(KEY, v);
      window.dispatchEvent(new CustomEvent(EVENT, { detail: v }));
    }
  }, []);

  return [node, set];
}
