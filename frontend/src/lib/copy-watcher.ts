"use client";

import { toast } from "sonner";
import { api, type CopyJobStatus } from "@/lib/api";
import { basename } from "@/lib/format";

const STORAGE_KEY = "copy-checkpoint.active";
const POLL_MS = 2000;
const STATUS_RECONNECT_MS = 60_000;

type ActiveCopy = {
  copyId: string;
  destCluster: string;
};

// Some browsers fire multiple events for the same tab; dedupe in-process so
// we don't end up with N toasts for the same copy on hot-reload or remount.
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

function readActive(): ActiveCopy[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function writeActive(list: ActiveCopy[]) {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

function addActive(m: ActiveCopy) {
  const list = readActive().filter((x) => x.copyId !== m.copyId);
  list.push(m);
  writeActive(list);
}

function removeActive(copyId: string) {
  writeActive(readActive().filter((x) => x.copyId !== copyId));
}

export function startCopyWatcher(copyId: string, destCluster: string) {
  addActive({ copyId, destCluster });
  void watch({ copyId, destCluster });
}

/** Resume any copies that were in flight when the page was loaded/refreshed. */
export function resumeActiveCopies() {
  for (const m of readActive()) {
    if (watched.has(m.copyId)) continue;
    void watch(m);
  }
}

async function watch({ copyId, destCluster }: ActiveCopy) {
  if (watched.has(copyId)) return;
  watched.add(copyId);

  let cancelled = false;
  const cancelAction = {
    label: "Cancel",
    onClick: () => {
      cancelled = true;
      void api(`/api/copy-jobs/${copyId}/cancel`, { method: "POST" }).catch(() => {});
    },
  };

  const toastId = `copy-checkpoint:${copyId}`;
  toast.loading("Copying checkpoint...", {
    id: toastId,
    duration: Infinity,
    action: cancelAction,
  });
  let lastStatusAt = Date.now();
  try {
    while (true) {
      let s: CopyJobStatus;
      try {
        s = await api<CopyJobStatus>(`/api/copy-jobs/${copyId}`);
        lastStatusAt = Date.now();
      } catch (e) {
        const message = (e as Error).message;
        if (/^404\b/.test(message)) {
          toast.dismiss(toastId);
          removeActive(copyId);
          return;
        }
        const elapsed = Date.now() - lastStatusAt;
        if (elapsed >= STATUS_RECONNECT_MS) {
          toast.dismiss(toastId);
          toast.error(
            `Copy status unavailable: ${message}`,
            { duration: 10_000 },
          );
          removeActive(copyId);
          return;
        }
        toast.loading(
          `Copying checkpoint... reconnecting (${Math.ceil((STATUS_RECONNECT_MS - elapsed) / 1000)}s)`,
          { id: toastId, duration: Infinity, action: cancelAction },
        );
        await new Promise((r) => setTimeout(r, POLL_MS));
        continue;
      }
      if (s.status === "done") {
        toast.success(
          `Copied ${s.copies_done} checkpoint${s.copies_done === 1 ? "" : "s"} to ${destCluster}`,
          { id: toastId, duration: 6000 },
        );
        removeActive(copyId);
        return;
      }
      if (s.status === "error") {
        // Dismiss the loading toast first — sonner with `richColors`
        // sometimes renders just the icon (no text) when a `loading` toast
        // is replaced in-place by `error` / `info`.
        toast.dismiss(toastId);
        if (cancelled || s.error === "cancelled") {
          toast("Copy cancelled", { duration: 4000 });
        } else {
          const msg = (s.error && s.error.trim()) || "Copy failed";
          toast.error(msg, { duration: 10_000 });
        }
        removeActive(copyId);
        return;
      }
      const src = s.src_size_bytes;
      const dst = s.dest_size_bytes;
      const shownDst =
        src && src > 0 && dst != null
          ? Math.min(dst, src)
          : dst;
      const pct =
        src && src > 0 && shownDst != null
          ? Math.min(100, Math.round((shownDst / src) * 100))
          : null;
      const summary = `${s.copies_done + 1}/${s.copies_total}`;
      const name = s.current_source ? basename(s.current_source) : null;
      const prefix = name
        ? `Copying ${name} (${summary})`
        : `Copying ${summary}`;
      const phase = s.phase ? `${s.phase}: ` : "";
      toast.loading(
        pct != null
          ? `${prefix} — ${phase}${pct}% (${formatBytes(shownDst)} / ${formatBytes(src)})`
          : `${prefix}…`,
        { id: toastId, duration: Infinity, action: cancelAction },
      );
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
  } catch (e) {
    toast.error(
      `Lost connection while copying: ${(e as Error).message}`,
      { id: toastId, duration: 10_000 },
    );
    // Keep the entry — the user can refresh and we'll try again.
  } finally {
    watched.delete(copyId);
  }
}
