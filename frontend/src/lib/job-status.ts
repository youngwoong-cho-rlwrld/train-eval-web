export type JobPhase = "train" | "resume" | "eval" | "other";

const ACTIVE_STATES = new Set([
  "RUNNING",
  "PENDING",
  "COMPLETING",
  "CONFIGURING",
  "SUSPENDED",
]);

export function jobPhase(jobName?: string | null): JobPhase {
  const match = jobName?.match(/(?:^|[-_])(train|resume|eval)[-_]/);
  return (match?.[1] as JobPhase | undefined) ?? "other";
}

export function normalizeJobPhase(phase?: string | null): JobPhase | null {
  if (phase === "train" || phase === "resume" || phase === "eval") return phase;
  return null;
}

export function isTrainJobPhase(phase?: string | null): boolean {
  return phase === "train" || phase === "resume";
}

export function isActiveJobState(state?: string | null): boolean {
  return ACTIVE_STATES.has((state ?? "").toUpperCase());
}

export function isCompletedJobState(state?: string | null): boolean {
  return (state ?? "").toUpperCase().startsWith("COMPLET");
}

export function isTimeoutJobState(state?: string | null): boolean {
  return (state ?? "").toUpperCase().startsWith("TIMEOUT");
}

export function isFailedJobState(state?: string | null): boolean {
  return /^(FAIL|OUT_OF_MEMORY|NODE_FAIL|PREEMPT)/i.test(state ?? "");
}

export function isTerminalJobState(state?: string | null): boolean {
  return /^(COMPLET|FAIL|CANCEL|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPT)/i.test(
    state ?? "",
  );
}
