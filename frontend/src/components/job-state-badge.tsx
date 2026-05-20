import { Badge } from "@/components/ui/badge";
import {
  isActiveJobState,
  isCompletedJobState,
  isTerminalJobState,
} from "@/lib/job-status";

export function JobStateBadge({ state }: { state: string }) {
  const upper = state.toUpperCase();
  const variant =
    upper === "RUNNING" || upper === "COMPLETING" ? "success"
    : isActiveJobState(state) ? "warning"
    : isTerminalJobState(state) && !isCompletedJobState(state) ? "danger"
    : isCompletedJobState(state) ? "secondary"
    : "outline";

  return <Badge variant={variant}>{state}</Badge>;
}
