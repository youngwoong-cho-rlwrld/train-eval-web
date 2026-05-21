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
# Cluster envs still export legacy REPO_ROOT; keep the submitted staging root.
SUBMIT_REPO_ROOT="$REPO_ROOT"
source "$REPO_ROOT/clusters/${CLUSTER}.env"
REPO_ROOT="$SUBMIT_REPO_ROOT"
source "$REPO_ROOT/lib/_common.sh"

EXP_DIR="$REPO_ROOT/experiments/$VARIANT"
[ -d "$EXP_DIR" ] || { echo "ERROR: experiment dir not found: $EXP_DIR"; exit 1; }
source "$EXP_DIR/config.sh"
TRAIN_REPO_DIR="${SUBMIT_TRAIN_REPO_DIR:-${TRAIN_REPO_DIR:-$GROOT_DIR}}"
TRAIN_NUM_GPUS="${SUBMIT_TRAIN_NUM_GPUS:-$TRAIN_NUM_GPUS}"
MAX_STEPS="${SUBMIT_TRAIN_MAX_STEPS:-$MAX_STEPS}"
SAVE_STEPS="${SUBMIT_TRAIN_SAVE_STEPS:-$SAVE_STEPS}"
if [ -n "${SUBMIT_TRAIN_GLOBAL_BATCH_SIZE:-}" ]; then
    if (( SUBMIT_TRAIN_GLOBAL_BATCH_SIZE % TRAIN_NUM_GPUS != 0 )); then
        echo "ERROR: SUBMIT_TRAIN_GLOBAL_BATCH_SIZE must be divisible by TRAIN_NUM_GPUS for n1.5 training"
        exit 1
    fi
    TRAIN_BATCH_SIZE=$((SUBMIT_TRAIN_GLOBAL_BATCH_SIZE / TRAIN_NUM_GPUS))
fi

GPU_INSTANCE="$(detect_gpu_instance)"
# EXP_NAME mirrors the slurm job name when launched via submit; fallback for ad-hoc runs.
EXP_NAME="${SLURM_JOB_NAME:-${VARIANT}_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)}"

CKPT_DIR="$EXP_DIR/checkpoints"
mkdir -p "$EXP_DIR/logs" "$LOG_DIR" "$CKPT_DIR"
LOG_FILE="$EXP_DIR/logs/train.log"

log "========================================="
log "$EXP_NAME"
log "  cluster=$CLUSTER  partition=${SUBMIT_PARTITION:-$PARTITION}  gpu=$GPU_INSTANCE"
log "  variant note: $TRAIN_NOTE"
log "========================================="

# N1.5 consumes a YAML data config. Prefer the editable per-experiment YAML
# when present; otherwise render the legacy config.sh-derived YAML.
DATA_CONFIG_SOURCE="$EXP_DIR/${TRAIN_DATA_CONFIG:-data_config.yaml}"
DATA_CONFIG_YAML="$EXP_DIR/.resolved_data_config.yaml"
USING_DATA_CONFIG_SOURCE=0
if [ -f "$DATA_CONFIG_SOURCE" ]; then
    USING_DATA_CONFIG_SOURCE=1
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line//\$\{DATA_DIR\}/$DATA_DIR}"
        line="${line//\$DATA_DIR/$DATA_DIR}"
        printf '%s\n' "$line"
    done < "$DATA_CONFIG_SOURCE" > "$DATA_CONFIG_YAML"
else
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
fi

if [[ "${DATASETS+set}" == set ]] && [ "${#DATASETS[@]}" -gt 0 ]; then
    log "Datasets (${#DATASETS[@]}):"
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r dname dcfg dweight <<<"$entry"
        log "  - $DATA_DIR/$dname  (data_config=$dcfg, weight=$dweight)"
    done
else
    if [ "${USING_DATA_CONFIG_SOURCE}" = "1" ]; then
        log "Datasets:       from $DATA_CONFIG_SOURCE"
    else
        log "Dataset:        $DATA_DIR/$DATASET_NAME"
        log "Data config:    $DATA_CONFIG"
    fi
fi
log "Data config YAML: $DATA_CONFIG_YAML"
log "Output:         $CKPT_DIR"
log "Max steps:      $MAX_STEPS"
log "Save steps:     $SAVE_STEPS"
log "Train GPUs:     $TRAIN_NUM_GPUS"
log "Global batch:   $((TRAIN_NUM_GPUS * TRAIN_BATCH_SIZE)) ($TRAIN_BATCH_SIZE per GPU)"


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

cd "$TRAIN_REPO_DIR"
source "$TRAIN_REPO_DIR/.venv/bin/activate"
export WANDB_PROJECT="${SUBMIT_WANDB_PROJECT:-${WANDB_PROJECT:-my project}}"
export WANDB_DIR="$EXP_DIR"

export WANDB_RESUME=allow

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
