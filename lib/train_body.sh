#!/usr/bin/env bash
# Run by sbatch via the top-level submit wrapper.
# Reads $VARIANT and $CLUSTER from the environment (set by submit --export).
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

# REPO_ROOT, CLUSTER, VARIANT come from sbatch --export (see submit wrapper).
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

log "========================================="
log "$EXP_NAME"
log "  cluster=$CLUSTER  partition=$PARTITION  gpu=$GPU_INSTANCE"
log "  variant note: $TRAIN_NOTE"
log "========================================="

# Render data_config.yaml from cluster + variant config.
# Two modes:
#   (a) DATASETS=("name|data_config|weight" ...) — multi-dataset co-training.
#   (b) DATASET_NAME=<name> + DATA_CONFIG=<cfg>  — legacy single-dataset (weight 1.0).
# Each <name> is joined with $DATA_DIR to form the dataset path on disk.
DATA_CONFIG_YAML="$EXP_DIR/data_config.yaml"
{
    echo "train:"
    echo "  datasets:"
    if [[ "${DATASETS+set}" == set ]] && [ "${#DATASETS[@]}" -gt 0 ]; then
        for entry in "${DATASETS[@]}"; do
            IFS='|' read -r dname dcfg dweight <<<"$entry"
            echo "    - path: $DATA_DIR/$dname"
            echo "      embodiment_tag: new_embodiment"
            echo "      data_config: $dcfg"
            echo "      weight: $dweight"
        done
    else
        echo "    - path: $DATA_DIR/$DATASET_NAME"
        echo "      embodiment_tag: new_embodiment"
        echo "      data_config: $DATA_CONFIG"
        echo "      weight: 1.0"
    fi
} > "$DATA_CONFIG_YAML"

if [[ "${DATASETS+set}" == set ]] && [ "${#DATASETS[@]}" -gt 0 ]; then
    log "Datasets (${#DATASETS[@]}):"
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r dname dcfg dweight <<<"$entry"
        log "  - $DATA_DIR/$dname  (data_config=$dcfg, weight=$dweight)"
    done
else
    log "Dataset:        $DATA_DIR/$DATASET_NAME"
    log "Data config:    $DATA_CONFIG"
fi
log "Output:         $CKPT_DIR"
log "Max steps:      $MAX_STEPS"


# Detect existing intermediate checkpoint → auto-resume
RESUME_FLAG=False
if compgen -G "$CKPT_DIR/checkpoint-*" > /dev/null; then
    LATEST_CKPT=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    log "Existing checkpoint found: $LATEST_CKPT — will resume"
    RESUME_FLAG=True
fi
if [[ "${RESUME_EXPECTED:-0}" == "1" && "$RESUME_FLAG" == "False" ]]; then
    log "ERROR: resume requested but no checkpoint in $CKPT_DIR"
    exit 1
fi

if [ -d "$CKPT_DIR/checkpoint-${MAX_STEPS}" ]; then
    log "Final checkpoint already exists at $CKPT_DIR/checkpoint-${MAX_STEPS} — skipping training."
    exit 0
fi

cd "$GROOT_DIR"
source "$GROOT_DIR/.venv/bin/activate"
export WANDB_PROJECT=gr00t
export WANDB_DIR="$EXP_DIR"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    export WANDB_RUN_ID="slurm_${SLURM_JOB_ID}"
    export WANDB_RESUME=allow
fi

python scripts/gr00t_finetune.py \
    --num-gpus "$TRAIN_NUM_GPUS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --learning_rate 1e-4 \
    --output-dir "$CKPT_DIR" \
    --data-config "$DATA_CONFIG_YAML" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --dataloader_num_workers 16 \
    --dataloader-prefetch-factor 10 \
    --video-backend torchcodec \
$([[ "$RESUME_FLAG" == "True" ]] && echo "--resume ") \
    --report-to wandb \
    --pin_memory \
    --run_name "$EXP_NAME" \
    --seed 42 \
    "${TRAIN_EXTRA_ARGS[@]}"

log "Training completed."
