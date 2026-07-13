import { isSlurmCluster } from "./cluster-env";

export type JobPhase = "train" | "resume" | "eval" | "unknown" | "other";

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

export function isRunningOrCompletingJobState(state?: string | null): boolean {
  return /^(RUNNING|COMPLETING)$/i.test(state ?? "");
}

// First whitespace-delimited token, upper-cased — strips trailing reasons such
// as "CANCELLED by 123" down to "CANCELLED" so states aggregate consistently.
export function primaryJobState(state?: string | null): string {
  return (state ?? "").split(" ")[0].toUpperCase();
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

// Human-readable lead-in for the "resubmitted from …" annotation shown next to
// resume/retry jobs. The optional sourceState lets callers that know the origin
// job's state (the detail page) refine the wording when resubmit_action is
// absent; callers that only know the action (the jobs table) omit it.
export function resubmitSourceLabel(
  action?: string | null,
  sourceState?: string | null,
): string {
  if (action === "retry" || isFailedJobState(sourceState)) {
    return "restarted from a failed job:";
  }
  if (action === "resume" || isTimeoutJobState(sourceState)) {
    return "resumed from a timeout job:";
  }
  return "resubmitted from job:";
}

export function canResumeJob({
  cluster,
  state,
}: {
  cluster: string;
  state?: string | null;
}): boolean {
  // Resume/retry rebuild from the Slurm sidecar meta; MLXP has no equivalent.
  return isSlurmCluster(cluster) && isTimeoutJobState(state);
}

export function canRetryJob({
  cluster,
  state,
}: {
  cluster: string;
  state?: string | null;
}): boolean {
  return isSlurmCluster(cluster) && isFailedJobState(state);
}

export function canCopyCheckpoint({
  state,
  phase,
}: {
  state?: string | null;
  phase?: string | null;
}): boolean {
  return isTrainJobPhase(phase) && isCompletedJobState(state);
}
