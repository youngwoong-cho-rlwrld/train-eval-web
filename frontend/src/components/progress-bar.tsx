import { cn } from "@/lib/utils";

// Shared progress track + fill. Clamps the fill width to 0-100%. The default
// 1.5 height matches the jobs-table cell; pass height="h-2" for the taller
// detail-page bars (N154).
export function ProgressBar({
  percent,
  height = "h-1.5",
  className,
}: {
  percent: number;
  height?: string;
  className?: string;
}) {
  const width = Math.max(0, Math.min(100, percent));
  return (
    <div
      className={cn(
        height,
        "w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800",
        className,
      )}
    >
      <div
        className="h-full rounded-full bg-slate-900 transition-all dark:bg-slate-50"
        style={{ width: `${width}%` }}
      />
    </div>
  );
}
