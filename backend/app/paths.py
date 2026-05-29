"""Path constants. Repo root is two levels up from this file."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
CLUSTERS_DIR = CONFIGS_DIR / "clusters"
EXPERIMENTS_DIR = CONFIGS_DIR / "experiments"
MODELS_DIR = CONFIGS_DIR / "models"
LIB_DIR = REPO_ROOT / "lib"

# Cluster-side staging dir, relative to the user's $HOME on each cluster.
CLUSTER_STAGING_REL = ".train-eval-web"

# Cluster-side copy-history dir. Slurm stores this below $HOME; MLXP stores it
# below the configured DDN experiments root.
CHECKPOINT_COPY_HISTORY_REL = f"{CLUSTER_STAGING_REL}/checkpoint-copies"
