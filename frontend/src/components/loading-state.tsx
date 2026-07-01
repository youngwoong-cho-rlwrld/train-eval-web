import { Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";

export function LoadingState({
  label = "Loading...",
  rows = 3,
  className,
}: {
  label?: string;
  rows?: number;
  className?: string;
}) {
  return (
    <div className={cn("space-y-3", className)} aria-busy="true">
      <p className="text-sm text-slate-500 dark:text-slate-400">{label}</p>
      <div className="space-y-2">
        {Array.from({ length: rows }).map((_, i) => (
          <div
            key={i}
            className={cn(
              "h-3 animate-pulse rounded bg-slate-100 dark:bg-slate-800",
              i % 3 === 0 && "w-full",
              i % 3 === 1 && "w-5/6",
              i % 3 === 2 && "w-2/3",
            )}
          />
        ))}
      </div>
    </div>
  );
}

export function InlineLoading({
  label = "Loading...",
  className,
}: {
  label?: string;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400",
        className,
      )}
      aria-busy="true"
    >
      <Loader2 aria-hidden="true" className="h-3.5 w-3.5 shrink-0 animate-spin" />
      {label}
    </span>
  );
}

export function ErrorState({
  message,
  className,
}: {
  message: string;
  className?: string;
}) {
  return (
    <p className={cn("text-sm text-red-600 dark:text-red-400", className)}>
      {message}
    </p>
  );
}

export function EmptyState({
  message,
  className,
}: {
  message: string;
  className?: string;
}) {
  return (
    <p className={cn("text-sm text-slate-500 dark:text-slate-400", className)}>
      {message}
    </p>
  );
}
