import { formatDuration, parseSlurmDuration } from "@/lib/duration";
import type { Progress } from "@/lib/api";

/**
 * Human-readable progress label shared by the /jobs table
 * (ActiveProgressCell) and the job detail page (ProgressCard). The backend
 * already builds a complete `current_label`, so it is rendered verbatim; the
 * eval `current_step`/`max_steps` pair provides an "episodes" fallback when
 * the label is absent. Returns null when there is nothing meaningful to show.
 */
export function activeProgressLabel(
  phase: string | null | undefined,
  progress: Progress | undefined,
): string | null {
  if (
    phase === "eval" &&
    progress?.current_step != null &&
    progress.max_steps != null
  ) {
    return `${progress.current_step}/${progress.max_steps} episodes`;
  }
  return progress?.current_label ?? null;
}

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
