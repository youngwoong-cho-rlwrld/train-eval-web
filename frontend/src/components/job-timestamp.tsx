import { formatJobTimestamp } from "@/lib/job-time";
import { ImmediateTooltip } from "@/components/immediate-tooltip";

// Shared rendering for a job timestamp: short form in the cell, full form in an
// immediate tooltip, em-dash placeholder when the value is missing/unknown.
// Used by the jobs list and the job-detail sacct table (N153).
export function JobTimestamp({ iso }: { iso?: string | null }) {
  const formatted = formatJobTimestamp(iso);
  if (!formatted) return <span className="text-slate-400">—</span>;
  return (
    <ImmediateTooltip content={formatted.full}>
      <span>{formatted.short}</span>
    </ImmediateTooltip>
  );
}
