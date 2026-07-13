/**
 * Model + harness resolution shared by the submit flow. The variant config
 * exposes raw env vars (`MODEL_ID`, `MODEL_VERSION`, `EVAL_HARNESS`); these
 * helpers centralize the precedence rules so the UI reads them consistently.
 */

type EvalHarness = "isaac" | "dexjoco";

/**
 * Resolve the variant's model identity. `modelFamily` (the backend registry's
 * answer, served on /api/variants/{name}) wins when present — the string
 * heuristic below can't know that e.g. dexjoco-* ids are n1.6-family.
 * `modelId` falls back through MODEL_ID -> MODEL_VERSION -> "n1.5"; without a
 * registry family, `model` normalizes the legacy "physixel" id to "n1.6"
 * unless MODEL_VERSION pins it.
 */
export function resolveModel(
  vars: Record<string, string> | undefined,
  modelFamily?: string | null,
): { modelId: string; model: string } {
  const modelId = vars?.MODEL_ID ?? vars?.MODEL_VERSION ?? "n1.5";
  const model =
    modelFamily ?? vars?.MODEL_VERSION ?? (modelId === "physixel" ? "n1.6" : modelId);
  return { modelId, model };
}

/** The eval harness for a variant: "dexjoco" when pinned, else "isaac". */
export function evalHarness(
  vars: Record<string, string> | undefined,
): EvalHarness {
  return vars?.EVAL_HARNESS === "dexjoco" ? "dexjoco" : "isaac";
}
