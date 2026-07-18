#!/usr/bin/env bash
# Run by sbatch via the top-level submit wrapper (family "gam").
# Reads $REPO_ROOT, $CLUSTER, $VARIANT from the environment (set by submit --export).
#
# GAM does not use gr00t's launch_finetune.py. This body reproduces the shared
# train harness (env/cluster/config sourcing, repo pinning, checkpoint layout,
# logging, resume/skip detection) but hands the actual launch to the GAM fork's
# wrapper: `bash dexjoco/train_dexjoco.sh` run from $TRAIN_REPO_DIR, driven
# entirely by GAM_* environment variables (see the integration contract). The
# wrapper writes a single-file checkpoint-final.pt (+ action_stats/ + config.yaml)
# directly into $GAM_RESULTS_DIR.
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

# REPO_ROOT, CLUSTER, VARIANT come from sbatch --export (see submit wrapper).
: "${REPO_ROOT:?REPO_ROOT must be set by submit wrapper}"
: "${CLUSTER:?CLUSTER must be set by submit wrapper}"
: "${VARIANT:?VARIANT must be set by submit wrapper}"
# Cluster envs still export legacy REPO_ROOT; keep the submitted staging root.
SUBMIT_REPO_ROOT="$REPO_ROOT"
source "$REPO_ROOT/clusters/${CLUSTER}.env"
REPO_ROOT="$SUBMIT_REPO_ROOT"
source "$REPO_ROOT/lib/_common.sh"

resolve_exp_and_config

# GAM's cluster repo path comes from SLURM_REPO_VAR=GAM_DIR (exported as
# SUBMIT_TRAIN_REPO_DIR by submit); TRAIN_REPO_DIR/GAM_DIR are fallbacks for
# ad-hoc runs.
TRAIN_REPO_DIR="${SUBMIT_TRAIN_REPO_DIR:-${TRAIN_REPO_DIR:-${GAM_DIR:-}}}"
: "${TRAIN_REPO_DIR:?TRAIN_REPO_DIR/GAM_DIR not set (cluster env GAM_DIR or SUBMIT_TRAIN_REPO_DIR)}"

# Submit-time overrides (backend/app/submit.py). gr00t-specific knobs
# (action horizon, extra torchrun args) do not apply to GAM and are ignored.
TRAIN_NUM_GPUS="${SUBMIT_TRAIN_NUM_GPUS:-$TRAIN_NUM_GPUS}"
MAX_STEPS="${SUBMIT_TRAIN_MAX_STEPS:-$MAX_STEPS}"
SAVE_STEPS="${SUBMIT_TRAIN_SAVE_STEPS:-$SAVE_STEPS}"
TRAIN_NUM_WORKERS="${SUBMIT_TRAIN_NUM_WORKERS:-${TRAIN_NUM_WORKERS:-16}}"
# GAM consumes a global batch size directly (like n1.6). Prefer the submit
# override, then the config's TRAIN_GLOBAL_BATCH_SIZE, then a per-device fallback.
GLOBAL_BATCH_SIZE="${SUBMIT_TRAIN_GLOBAL_BATCH_SIZE:-${TRAIN_GLOBAL_BATCH_SIZE:-$((TRAIN_NUM_GPUS * ${TRAIN_BATCH_SIZE:-1}))}}"

GPU_INSTANCE="$(detect_gpu_instance)"
# EXP_NAME mirrors the slurm job name when launched via submit; fallback for ad-hoc runs.
EXP_NAME="${SLURM_JOB_NAME:-${VARIANT}_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)}"
OUTPUT_NAMESPACE="${SUBMIT_OUTPUT_NAMESPACE:-$EXP_NAME}"

# Checkpoint layout matches the shared convention: $OUT_DIR/checkpoints/$OUTPUT_NAMESPACE.
CKPT_DIR="$OUT_DIR/checkpoints"
RUN_CKPT_DIR="$CKPT_DIR/$OUTPUT_NAMESPACE"
# Org policy: the training output path is exported as MODEL_OUTPUT_DIR and
# consumed by the launch argument below (GAM_RESULTS_DIR).
export MODEL_OUTPUT_DIR="$RUN_CKPT_DIR"
mkdir -p "$OUT_DIR/logs" "$LOG_DIR" "$RUN_CKPT_DIR"
LOG_FILE="$OUT_DIR/logs/train.log"

# Capture the main checkout before pin_training_repo_dir swaps TRAIN_REPO_DIR
# for a per-job git worktree. GAM's untracked assets (the DA3 backbone ckpt and
# the pretrained init ckpt) live ONLY in the main repo, never in the worktree.
GAM_MAIN_REPO_DIR="$TRAIN_REPO_DIR"
SUBMIT_GIT_COMMIT="${SUBMIT_GIT_COMMIT:-${TRAIN_GIT_COMMIT:-}}"
pin_training_repo_dir "$TRAIN_REPO_DIR" "$SUBMIT_GIT_COMMIT" "${SLURM_JOB_ID:-$OUTPUT_NAMESPACE}"

# GAM training config (the variant's "second file"): OmegaConf YAML.
: "${TRAIN_MODALITY_CONFIG:?TRAIN_MODALITY_CONFIG not set in config.sh}"
GAM_CONFIG_YAML="$EXP_DIR/$TRAIN_MODALITY_CONFIG"
[ -f "$GAM_CONFIG_YAML" ] || { echo "ERROR: GAM config not found: $GAM_CONFIG_YAML"; exit 1; }

: "${DATA_DIR:?DATA_DIR not set in config.sh}"
[ -x "$TRAIN_REPO_DIR/.venv/bin/python" ] || log "WARNING: $TRAIN_REPO_DIR/.venv/bin/python not found; GAM wrapper must provision its own venv"

log "============================================="
log "$EXP_NAME"
log "  cluster=$CLUSTER  partition=${SUBMIT_PARTITION:-$PARTITION}  gpu=$GPU_INSTANCE  model=${MODEL_ID:-dexjoco-gam}"
log "  train repo=$TRAIN_REPO_DIR"
log "  variant note: ${TRAIN_NOTE:-}"
log "  run namespace=$OUTPUT_NAMESPACE"
log "  output=$RUN_CKPT_DIR"
log "  GAM config=$GAM_CONFIG_YAML"
log "  data root=$DATA_DIR"
log "  global batch=$GLOBAL_BATCH_SIZE  gpus=$TRAIN_NUM_GPUS  workers=$TRAIN_NUM_WORKERS"
log "  max steps=$MAX_STEPS  save steps=$SAVE_STEPS"
log "============================================="

# Completion marker is the single-file checkpoint the wrapper guarantees.
if [ -f "$RUN_CKPT_DIR/checkpoint-final.pt" ]; then
    log "Final checkpoint already exists at $RUN_CKPT_DIR/checkpoint-final.pt — skipping training."
    exit 0
fi
if [[ "${RESUME_EXPECTED:-0}" == "1" ]]; then
    if compgen -G "$RUN_CKPT_DIR/checkpoint-*" > /dev/null 2>&1; then
        log "Resume requested; existing checkpoint state found under $RUN_CKPT_DIR"
    else
        log "ERROR: resume requested but no checkpoint state in $RUN_CKPT_DIR"
        exit 1
    fi
fi

export WANDB_PROJECT="${SUBMIT_WANDB_PROJECT:-${WANDB_PROJECT:-my project}}"
export WANDB_DIR="$EXP_DIR"
export WANDB_RESUME=allow

cd "$TRAIN_REPO_DIR"
log "Launching GAM training wrapper: dexjoco/train_dexjoco.sh"

# The wrapper defaults DA3_ROOT/GAM_INIT_CKPT to the (worktree) repo dir, where
# these untracked assets are absent. Pin them to the main checkout so stage_1
# finds checkpoints/track4world_da3.pth and training resumes the pretrained init.
export DA3_ROOT="$GAM_MAIN_REPO_DIR"
export GAM_INIT_CKPT="$GAM_MAIN_REPO_DIR/checkpoints/pretrained-gam.pt"
log "  DA3_ROOT=$DA3_ROOT  GAM_INIT_CKPT=$GAM_INIT_CKPT"

# GAM_WANDB_RUN_ID == the slurm job/exp name: the wrapper enables --wandb with
# id==job_name (project pinned to dexjoco) so the backend resolves the run link.
GAM_CONFIG_YAML="$GAM_CONFIG_YAML" \
GAM_DATA_ROOT="$DATA_DIR" \
GAM_RESULTS_DIR="$MODEL_OUTPUT_DIR" \
GAM_NUM_GPUS="$TRAIN_NUM_GPUS" \
GAM_GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
GAM_MAX_STEPS="$MAX_STEPS" \
GAM_SAVE_STEPS="$SAVE_STEPS" \
GAM_NUM_WORKERS="${TRAIN_NUM_WORKERS:-}" \
GAM_WANDB_RUN_ID="$EXP_NAME" \
    bash dexjoco/train_dexjoco.sh

log "Training completed."
if [ ! -f "$RUN_CKPT_DIR/checkpoint-final.pt" ]; then
    log "ERROR: GAM wrapper exited 0 but $RUN_CKPT_DIR/checkpoint-final.pt is missing"
    exit 1
fi
log "Checkpoint: $RUN_CKPT_DIR/checkpoint-final.pt"
