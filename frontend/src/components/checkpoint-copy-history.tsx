"use client";

import type { CheckpointCopyRecord } from "@/lib/api";
import { CopyButton } from "@/components/copy-button";

export function CheckpointCopyList({
  records,
  className = "",
  itemClassName = "",
  showTime = true,
}: {
  records: CheckpointCopyRecord[];
  className?: string;
  itemClassName?: string;
  showTime?: boolean;
}) {
  return (
    <div className={className}>
      {records.map((item) => (
        <div
          key={`${item.copy_id}:${item.source_path}:${item.dest_path}`}
          className={itemClassName}
        >
          <div className="grid min-w-0 grid-cols-[76px_minmax(0,1fr)_auto] items-center gap-2">
            <span className="uppercase tracking-wide text-slate-500">copied to</span>
            <CheckpointPath path={item.dest_path} exists={item.dest_exists} />
            <CopyButton value={item.dest_path} />
          </div>
          {showTime && (
            <div className="mt-1 pl-[84px] text-[10px] text-slate-500">
              {formatCopyTime(item.copied_at)}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function CheckpointPath({
  path,
  exists,
}: {
  path: string;
  exists: boolean | null;
}) {
  return (
    <span
      className={`truncate font-mono ${exists === false ? "line-through" : ""}`}
      title={path}
    >
      {path}
    </span>
  );
}

function formatCopyTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "";
  return new Date(seconds * 1000).toLocaleString();
}
