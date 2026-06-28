"""Path constants. Repo root is two levels up from this file."""
from __future__ import annotations


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


# ── Output path-layout builders ──
# Pure string builders for the per-submission output layout shared by the slurm
# (submit.py) and MLXP (mlxp_submit.py) submitters. `exp_dir` is the experiment
# root (slurm uses `$HOME/<staging>/experiments/<variant>`; MLXP uses the DDN
# experiments root), `namespace` is the per-submission output namespace.

def checkpoint_dir(exp_dir: str, namespace: str) -> str:
    return f"{exp_dir}/checkpoints/{namespace}"


def eval_dir(exp_dir: str, namespace: str) -> str:
    return f"{exp_dir}/eval_results/{namespace}"


def results_path(eval_dir: str) -> str:
    return f"{eval_dir}/results.json"


def job_log_dir(exp_dir: str, namespace: str) -> str:
    return f"{exp_dir}/logs/{namespace}"


def config_path(exp_dir: str, suffix: str) -> str:
    return f"{exp_dir}/config_{suffix}.sh"


def meta_path(exp_dir: str, suffix: str) -> str:
    return f"{exp_dir}/config_{suffix}.meta.json"
