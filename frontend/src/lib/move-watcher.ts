"use client";

import { toast } from "sonner";
import { api } from "@/lib/api";

const STORAGE_KEY = "move-checkpoint.active";

export type MoveJobStatus = {
  move_id: string;
  status: "running" | "done" | "error";
  error: string | null;
  moves_total: number;
  moves_done: number;
  current_source: string | null;
  current_dest: string | null;
  src_size_bytes: number | null;
  dest_size_bytes: number | null;
  started_at: number;
  finished_at: number | null;
};

type ActiveMove = {
  moveId: string;
  verb: "Move" | "Copy";
  destCluster: string;
};

// Some browsers fire multiple events for the same tab; dedupe in-process so
// we don't end up with N toasts for the same move on hot-reload or remount.
const watched = new Set<string>();

function formatBytes(b: number | null): string {
  if (b == null) return "—";
  if (b < 1024) return `${b}B`;
  const u = ["KB", "MB", "GB", "TB"];
  let v = b / 1024;
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(1)}${u[i]}`;
}

function readActive(): ActiveMove[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function writeActive(list: ActiveMove[]) {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

function addActive(m: ActiveMove) {
  const list = readActive().filter((x) => x.moveId !== m.moveId);
  list.push(m);
  writeActive(list);
}

function removeActive(moveId: string) {
  writeActive(readActive().filter((x) => x.moveId !== moveId));
}

export function startMoveWatcher(
  moveId: string,
  verb: "Move" | "Copy",
  destCluster: string,
) {
  addActive({ moveId, verb, destCluster });
  void watch({ moveId, verb, destCluster });
}

/** Resume any moves that were in flight when the page was loaded/refreshed. */
export function resumeActiveMoves() {
  for (const m of readActive()) {
    if (watched.has(m.moveId)) continue;
    void watch(m);
  }
}

async function watch({ moveId, verb, destCluster }: ActiveMove) {
  if (watched.has(moveId)) return;
  watched.add(moveId);

  const past = verb === "Move" ? "Moved" : "Copied";
  const present = verb === "Move" ? "Moving" : "Copying";

  let cancelled = false;
  const cancelAction = {
    label: "Cancel",
    onClick: () => {
      cancelled = true;
      void api(`/api/move-jobs/${moveId}/cancel`, { method: "POST" }).catch(() => {});
    },
  };

  const toastId = toast.loading(`${present} checkpoint…`, {
    duration: Infinity,
    action: cancelAction,
  });
  try {
    while (true) {
      let s: MoveJobStatus;
      try {
        s = await api<MoveJobStatus>(`/api/move-jobs/${moveId}`);
      } catch (e) {
        // 404 means the backend forgot about this move (process restart).
        // Drop it silently — the toast clears below.
        if (/^404/.test((e as Error).message)) {
          toast.dismiss(toastId);
          removeActive(moveId);
          return;
        }
        throw e;
      }
      if (s.status === "done") {
        toast.success(
          `${past} ${s.moves_done} checkpoint${s.moves_done === 1 ? "" : "s"} to ${destCluster}`,
          { id: toastId, duration: 6000 },
        );
        removeActive(moveId);
        return;
      }
      if (s.status === "error") {
        if (cancelled || s.error === "cancelled") {
          toast.info(`${verb} cancelled`, { id: toastId, duration: 4000 });
        } else {
          toast.error(s.error ?? `${verb} failed`, {
            id: toastId,
            duration: 10_000,
          });
        }
        removeActive(moveId);
        return;
      }
      const src = s.src_size_bytes;
      const dst = s.dest_size_bytes;
      const pct =
        src && src > 0 && dst != null
          ? Math.min(100, Math.round((dst / src) * 100))
          : null;
      const summary = `${s.moves_done + 1}/${s.moves_total}`;
      const name = s.current_source
        ? s.current_source.split("/").filter(Boolean).pop()
        : null;
      const prefix = name
        ? `${present} ${name} (${summary})`
        : `${present} ${summary}`;
      toast.loading(
        pct != null
          ? `${prefix} — ${pct}% (${formatBytes(dst)} / ${formatBytes(src)})`
          : `${prefix}…`,
        { id: toastId, duration: Infinity, action: cancelAction },
      );
      await new Promise((r) => setTimeout(r, 2000));
    }
  } catch (e) {
    toast.error(
      `Lost connection while ${present.toLowerCase()}: ${(e as Error).message}`,
      { id: toastId, duration: 10_000 },
    );
    // Keep the entry — the user can refresh and we'll try again.
  } finally {
    watched.delete(moveId);
  }
}
