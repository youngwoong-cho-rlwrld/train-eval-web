import { Badge } from "@/components/ui/badge";

export function JobStateBadge({ state }: { state: string }) {
  const upper = state.toUpperCase();
  const variant =
    upper === "RUNNING" || upper === "COMPLETING" ? "success"
    : upper === "PENDING" || upper === "CONFIGURING" || upper === "SUSPENDED" ? "warning"
    : upper.startsWith("FAIL") ||
      upper.startsWith("TIMEOUT") ||
      upper.startsWith("CANCEL") ||
      upper.startsWith("OUT_OF_MEMORY") ||
      upper.startsWith("NODE_FAIL") ||
      upper.startsWith("PREEMPT") ? "danger"
    : upper.startsWith("COMPLET") ? "secondary"
    : "outline";

  return <Badge variant={variant}>{state}</Badge>;
}
