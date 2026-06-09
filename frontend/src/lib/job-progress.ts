import { formatDuration, parseSlurmDuration } from "@/lib/duration";

/**
 * Linear ETA from elapsed × steps-remaining / current-step. Shared by the
 * /jobs table (ActiveProgressCell) and the job detail page (ProgressCard).
 * Returns null when the inputs can't support an estimate.
 */
export function stepEta(
  elapsed: string | null | undefined,
  currentStep: number | null | undefined,
  maxStep: number | null | undefined,
  phase: string | null | undefined,
): { etaLabel: string; etaTitle: string } | null {
  if (!currentStep || !maxStep || currentStep >= maxStep || !elapsed) return null;
  const elapsedSec = parseSlurmDuration(elapsed);
  if (elapsedSec <= 0) return null;
  const etaSec = (elapsedSec * (maxStep - currentStep)) / currentStep;
  const unit = phase === "eval" ? "episode" : "step";
  return {
    etaLabel: formatDuration(etaSec),
    etaTitle: `Estimated from aggregate ${unit} throughput`,
  };
}
