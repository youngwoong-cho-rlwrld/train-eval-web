const mlxpUser = process.env.NEXT_PUBLIC_MLXP_USER ?? "youngwoong";

export const MLXP_DEFAULT_NODE =
  process.env.NEXT_PUBLIC_MLXP_DEFAULT_NODE ?? "h200-03-w-3c55";

export const MLXP_DATASETS_DIR =
  process.env.NEXT_PUBLIC_MLXP_DATASETS_DIR ?? `/data/${mlxpUser}/datasets`;
