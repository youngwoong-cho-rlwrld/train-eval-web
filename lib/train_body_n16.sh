#!/usr/bin/env bash
# Run by sbatch via the top-level submit wrapper when MODEL_VERSION=n1.6.
# Reads $REPO_ROOT, $CLUSTER, $VARIANT from the environment (set by submit --export).
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

: "${REPO_ROOT:?REPO_ROOT must be set by submit wrapper}"
: "${CLUSTER:?CLUSTER must be set by submit wrapper}"
: "${VARIANT:?VARIANT must be set by submit wrapper}"
source "$REPO_ROOT/clusters/${CLUSTER}.env"
source "$REPO_ROOT/lib/_common.sh"

EXP_DIR="$REPO_ROOT/experiments/$VARIANT"
[ -d "$EXP_DIR" ] || { echo "ERROR: experiment dir not found: $EXP_DIR"; exit 1; }
source "$EXP_DIR/config.sh"

GPU_INSTANCE="$(detect_gpu_instance)"
# EXP_NAME mirrors the slurm job name when launched via submit; fallback for ad-hoc runs.
EXP_NAME="${SLURM_JOB_NAME:-${VARIANT}_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)}"

CKPT_DIR="$EXP_DIR/checkpoints"
mkdir -p "$EXP_DIR/logs" "$LOG_DIR" "$CKPT_DIR"
LOG_FILE="$EXP_DIR/logs/train.log"

log "============================================="
log "$EXP_NAME"
log "  cluster=$CLUSTER  partition=$PARTITION  gpu=$GPU_INSTANCE  model=n1.6"
log "  variant note: $TRAIN_NOTE"
log "============================================="

# ── Build dataset path list (multi-dataset support, backward compat) ──
# Precedence: TRAIN_DATASET_NAMES (array) > DATASET_NAME (single)
DATASET_PATHS=()
if declare -p TRAIN_DATASET_NAMES 2>/dev/null | grep -q "declare -a"; then
    for n in "${TRAIN_DATASET_NAMES[@]}"; do DATASET_PATHS+=("$DATA_DIR/$n"); done
else
    DATASET_PATHS=("$DATA_DIR/$DATASET_NAME")
fi
log "Datasets (${#DATASET_PATHS[@]}):"
for p in "${DATASET_PATHS[@]}"; do log "  - $p"; done

# ── Per-variant modality config (Python file, copied into experiment dir) ──
: "${TRAIN_MODALITY_CONFIG:?TRAIN_MODALITY_CONFIG not set in config.sh}"
MODALITY_CONFIG_FILE="$EXP_DIR/$TRAIN_MODALITY_CONFIG"
[ -f "$MODALITY_CONFIG_FILE" ] || { echo "ERROR: modality config not found: $MODALITY_CONFIG_FILE"; exit 1; }
log "Modality config: $MODALITY_CONFIG_FILE"

# ── Per-device → global batch size (decision: keep TRAIN_BATCH_SIZE per-device) ──
GLOBAL_BATCH_SIZE=$((TRAIN_NUM_GPUS * TRAIN_BATCH_SIZE))
log "Global batch: $TRAIN_NUM_GPUS GPUs × $TRAIN_BATCH_SIZE per-device = $GLOBAL_BATCH_SIZE"

if [ -d "$CKPT_DIR/checkpoint-${MAX_STEPS}" ]; then
    log "Final checkpoint already exists at $CKPT_DIR/checkpoint-${MAX_STEPS} — skipping training."
    exit 0
fi

# uv may not be on PATH in non-login shells (it lives at $HOME/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=gr00t
export WANDB_DIR="$EXP_DIR"

cd "$GROOT_N16_DIR"

uv run torchrun --nproc_per_node="$TRAIN_NUM_GPUS" gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path "${DATASET_PATHS[@]}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG_FILE" \
    --num-gpus "$TRAIN_NUM_GPUS" \
    --output-dir "$CKPT_DIR" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --learning-rate 1e-4 \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit 5 \
    --dataloader-num-workers 8 \
    --experiment-name "$EXP_NAME" \
    --use-wandb \
    --color-jitter-params brightness 0.2 contrast 0.2 saturation 0.2 hue 0.1 \
    "${TRAIN_EXTRA_ARGS[@]}"

log "Training completed."
