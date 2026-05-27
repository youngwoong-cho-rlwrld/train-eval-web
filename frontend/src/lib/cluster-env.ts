export type EnvField = {
  key: string;
  description: string;
};

const SLURM_ENV_FIELDS: EnvField[] = [
  { key: "CLUSTER", description: "Cluster id used by jobs and metadata." },
  { key: "SSH_ALIAS", description: "Local SSH host alias for backend commands." },
  { key: "PARTITION", description: "Default Slurm partition when submit does not override it." },
  { key: "REPO_ROOT", description: "Legacy remote script root; submit preserves its staged root at runtime." },
  { key: "GROOT_DIR", description: "Remote GR00T N1.5 repository directory." },
  { key: "GROOT_N16_DIR", description: "Remote GR00T N1.6 repository directory." },
  { key: "PHYSIXEL_DIR", description: "Remote PhysiXel repository directory." },
  { key: "ISAAC_DIR", description: "Remote Isaac Sim / environment repository directory." },
  { key: "DATA_DIR", description: "Remote dataset root used by experiment configs." },
  { key: "LOG_DIR", description: "Remote Slurm stdout/stderr log directory." },
  { key: "SBATCH_EXCLUDE", description: "Optional comma-separated Slurm nodes to exclude." },
  { key: "SLURM_EXCLUDE_NODES", description: "Fallback node exclusion list for older scripts." },
  { key: "LD_LIBRARY_PATH", description: "Extra runtime library path exported before train/eval starts." },
];

const MLXP_ENV_FIELDS: EnvField[] = [
  { key: "TRAIN_EVAL_MLXP_USER", description: "MLXP user; derives /data/<user> defaults." },
  { key: "TRAIN_EVAL_MLXP_NAMESPACE", description: "Kubernetes namespace." },
  { key: "TRAIN_EVAL_MLXP_OWNER", description: "Owner label used to find jobs and pods." },
  { key: "TRAIN_EVAL_MLXP_TOOL_LABEL", description: "Tool label used to find jobs and pods." },
  { key: "TRAIN_EVAL_MLXP_NODE", description: "Default GPU node for MLXP jobs." },
  { key: "TRAIN_EVAL_MLXP_GPU_NODE_PREFIX", description: "GPU node name prefix shown in monitor." },
  { key: "TRAIN_EVAL_MLXP_GPU_TYPE", description: "GPU type label shown in monitor." },
  { key: "TRAIN_EVAL_MLXP_GPUS_PER_NODE", description: "Total GPU count per MLXP node." },
  { key: "TRAIN_EVAL_MLXP_DDN_MOUNT", description: "Shared DDN mount path." },
  { key: "TRAIN_EVAL_MLXP_HOME", description: "User home directory on DDN." },
  { key: "TRAIN_EVAL_MLXP_DATASETS_DIR", description: "Default MLXP dataset root." },
  { key: "TRAIN_EVAL_MLXP_EXPERIMENTS_DIR", description: "MLXP experiment and checkpoint root." },
  { key: "TRAIN_EVAL_MLXP_HF_HOME", description: "Hugging Face cache directory." },
  { key: "TRAIN_EVAL_MLXP_WORKSPACE_DIR", description: "MLXP workspace root containing repositories." },
  { key: "TRAIN_EVAL_MLXP_ISAAC_DIR", description: "MLXP Isaac Sim / environment repository directory." },
  { key: "TRAIN_EVAL_MLXP_DATA_POD", description: "Long-lived data pod used for listing and copying files." },
  { key: "TRAIN_EVAL_MLXP_DDN_PVC", description: "PVC mounted into MLXP jobs." },
  { key: "TRAIN_EVAL_MLXP_IMAGE", description: "Container image for MLXP train/eval jobs." },
  { key: "TRAIN_EVAL_MLXP_IMAGE_PULL_SECRET", description: "Kubernetes image pull secret." },
  { key: "TRAIN_EVAL_MLXP_ZONE", description: "MLXP node zone label." },
  { key: "TRAIN_EVAL_MLXP_WANDB_SECRET", description: "Kubernetes secret containing the W&B key." },
];

export function fieldsForClusterEnv(
  clusterName: string,
  saved: Record<string, string>,
  draft: Record<string, string>,
): EnvField[] {
  const base = clusterName === "mlxp" ? MLXP_ENV_FIELDS : SLURM_ENV_FIELDS;
  const known = new Set(base.map((field) => field.key));
  const extras = Array.from(new Set([...Object.keys(saved), ...Object.keys(draft)]))
    .filter((key) => !known.has(key))
    .sort()
    .map((key) => ({ key, description: "Additional environment value." }));
  return [...base, ...extras];
}

export function parseEnvText(text: string): Record<string, string> {
  const values: Record<string, string> = {};
  for (const raw of text.split("\n")) {
    let line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    if (line.startsWith("export ")) line = line.slice("export ".length).trim();
    const eq = line.indexOf("=");
    const key = line.slice(0, eq).trim();
    const value = stripEnvQuotes(line.slice(eq + 1).trim());
    if (key) values[key] = value;
  }
  return values;
}

export function renderEnvText(
  fields: EnvField[],
  values: Record<string, string>,
): string {
  return fields
    .map((field) => `export ${field.key}=${formatEnvValue(values[field.key] ?? "")}`)
    .join("\n") + "\n";
}

export function sameEnvValues(
  a: Record<string, string>,
  b: Record<string, string>,
  fields: EnvField[],
): boolean {
  return fields.every((field) => (a[field.key] ?? "") === (b[field.key] ?? ""));
}

export function normalizeEnvDraft(
  value: unknown,
  fallback: Record<string, string>,
): Record<string, string> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fallback;
  const out: Record<string, string> = {};
  for (const [key, raw] of Object.entries(value)) {
    if (typeof raw === "string") out[key] = raw;
  }
  return out;
}

function stripEnvQuotes(value: string): string {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === `"` && last === `"`) || (first === `'` && last === `'`)) {
      return value.slice(1, -1);
    }
  }
  return value;
}

function formatEnvValue(value: string): string {
  if (value === "") return "";
  return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/`/g, "\\`")}"`;
}
