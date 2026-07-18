#!/usr/bin/env bash
# Run by sbatch via the top-level submit wrapper.
# Reads $REPO_ROOT, $CLUSTER, $VARIANT from the environment (set by submit --export).
#
# Unified across model families: the family is taken from the sourced config.sh
# as MODEL_FAMILY (preferred), then MODEL_VERSION, defaulting to n1.5. n1.5 runs
# scripts/gr00t_finetune.py against a rendered YAML data config; n1.6 runs
# gr00t/experiment/launch_finetune.py via torchrun. The shared prologue (env,
# cluster/config sourcing, GPU/exp-name resolution, trainer-state cleanup helper
# from lib/_common.sh) is common to both.
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

# SUBMIT_EXP_DIR is only ever exported for eval jobs, so honoring it here is a
# no-op for train — kept uniform via the shared helper.
resolve_exp_and_config

# Resolve the model family: MODEL_FAMILY > MODEL_VERSION > n1.5 default.
MODEL_FAMILY="${MODEL_FAMILY:-${MODEL_VERSION:-n1.5}}"

if [ "$MODEL_FAMILY" = "n1.5" ]; then
    TRAIN_REPO_DIR="${SUBMIT_TRAIN_REPO_DIR:-${TRAIN_REPO_DIR:-$GROOT_DIR}}"
    TRAIN_NUM_GPUS="${SUBMIT_TRAIN_NUM_GPUS:-$TRAIN_NUM_GPUS}"
    MAX_STEPS="${SUBMIT_TRAIN_MAX_STEPS:-$MAX_STEPS}"
    SAVE_STEPS="${SUBMIT_TRAIN_SAVE_STEPS:-$SAVE_STEPS}"
    TRAIN_NUM_WORKERS="${SUBMIT_TRAIN_NUM_WORKERS:-${TRAIN_NUM_WORKERS:-16}}"
    if [ -n "${SUBMIT_TRAIN_GLOBAL_BATCH_SIZE:-}" ]; then
        if (( SUBMIT_TRAIN_GLOBAL_BATCH_SIZE % TRAIN_NUM_GPUS != 0 )); then
            echo "ERROR: SUBMIT_TRAIN_GLOBAL_BATCH_SIZE must be divisible by TRAIN_NUM_GPUS for n1.5 training"
            exit 1
        fi
        TRAIN_BATCH_SIZE=$((SUBMIT_TRAIN_GLOBAL_BATCH_SIZE / TRAIN_NUM_GPUS))
    fi
else
    # n1.6 and n1.7 share this prologue; only the training repo default and the
    # final launcher differ. Webapp submissions always set SUBMIT_TRAIN_REPO_DIR
    # (resolved from the model's SLURM_REPO_VAR), so the fallback below only
    # matters for ad-hoc runs.
    if [ "$MODEL_FAMILY" = "n1.7" ]; then
        TRAIN_REPO_DIR="${SUBMIT_TRAIN_REPO_DIR:-${TRAIN_REPO_DIR:-${GROOT_N17_DIR:-}}}"
    else
        TRAIN_REPO_DIR="${SUBMIT_TRAIN_REPO_DIR:-${TRAIN_REPO_DIR:-$GROOT_N16_DIR}}"
    fi
    TRAIN_NUM_GPUS="${SUBMIT_TRAIN_NUM_GPUS:-$TRAIN_NUM_GPUS}"
    MAX_STEPS="${SUBMIT_TRAIN_MAX_STEPS:-$MAX_STEPS}"
    SAVE_STEPS="${SUBMIT_TRAIN_SAVE_STEPS:-$SAVE_STEPS}"
    TRAIN_NUM_WORKERS="${SUBMIT_TRAIN_NUM_WORKERS:-${TRAIN_NUM_WORKERS:-16}}"
    append_submit_extra_train_args
fi

GPU_INSTANCE="$(detect_gpu_instance)"
# EXP_NAME mirrors the slurm job name when launched via submit; fallback for ad-hoc runs.
EXP_NAME="${SLURM_JOB_NAME:-${VARIANT}_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)}"
OUTPUT_NAMESPACE="${SUBMIT_OUTPUT_NAMESPACE:-$EXP_NAME}"

if [ "$MODEL_FAMILY" = "n1.5" ]; then
    CKPT_DIR="$OUT_DIR/checkpoints/$OUTPUT_NAMESPACE"
    # Org policy: the training output path is exported as MODEL_OUTPUT_DIR and
    # consumed by the python --output-dir argument below.
    export MODEL_OUTPUT_DIR="$CKPT_DIR"
    mkdir -p "$OUT_DIR/logs" "$LOG_DIR" "$CKPT_DIR"
    LOG_FILE="$OUT_DIR/logs/train.log"
    SUBMIT_GIT_COMMIT="${SUBMIT_GIT_COMMIT:-${TRAIN_GIT_COMMIT:-}}"
    pin_training_repo_dir "$TRAIN_REPO_DIR" "$SUBMIT_GIT_COMMIT" "${SLURM_JOB_ID:-$OUTPUT_NAMESPACE}"

    log "========================================="
    log "$EXP_NAME"
    log "  cluster=$CLUSTER  partition=${SUBMIT_PARTITION:-$PARTITION}  gpu=$GPU_INSTANCE"
    log "  train repo=$TRAIN_REPO_DIR"
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
    log "Run namespace:  $OUTPUT_NAMESPACE"
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
        cleanup_trainer_state "$CKPT_DIR" "$MAX_STEPS"
        exit 0
    fi

    cd "$TRAIN_REPO_DIR"
    source "$TRAIN_REPO_DIR/.venv/bin/activate"
    export WANDB_PROJECT="${SUBMIT_WANDB_PROJECT:-${WANDB_PROJECT:-my project}}"
    export WANDB_DIR="$EXP_DIR"

    export WANDB_RESUME=allow
    append_submit_extra_train_args

    python scripts/gr00t_finetune.py \
        --num-gpus "$TRAIN_NUM_GPUS" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --learning_rate 1e-4 \
        --output-dir "$MODEL_OUTPUT_DIR" \
        --data-config "$DATA_CONFIG_YAML" \
        --max-steps "$MAX_STEPS" \
        --save-steps "$SAVE_STEPS" \
        --dataloader_num_workers "$TRAIN_NUM_WORKERS" \
        --dataloader-prefetch-factor 10 \
        --video-backend torchcodec \
    $([[ "$RESUME_FLAG" == "True" ]] && echo "--resume ") \
        --report-to wandb \
        --pin_memory \
        --run_name "$EXP_NAME" \
        --seed 42 \
        "${TRAIN_EXTRA_ARGS[@]}"

    log "Training completed."
    cleanup_trainer_state "$CKPT_DIR" "$MAX_STEPS"
else
    CKPT_DIR="$OUT_DIR/checkpoints"
    RUN_CKPT_DIR="$CKPT_DIR/$OUTPUT_NAMESPACE"
    # Org policy: the training output path is exported as MODEL_OUTPUT_DIR and
    # consumed by the python --output-dir argument below. For n1.6/n1.7 the
    # launcher takes the parent "checkpoints" dir plus --experiment-name, so
    # MODEL_OUTPUT_DIR is that parent (the per-run dir is $RUN_CKPT_DIR).
    export MODEL_OUTPUT_DIR="$CKPT_DIR"
    mkdir -p "$OUT_DIR/logs" "$LOG_DIR" "$CKPT_DIR"
    LOG_FILE="$OUT_DIR/logs/train.log"
    SUBMIT_GIT_COMMIT="${SUBMIT_GIT_COMMIT:-${TRAIN_GIT_COMMIT:-}}"
    pin_training_repo_dir "$TRAIN_REPO_DIR" "$SUBMIT_GIT_COMMIT" "${SLURM_JOB_ID:-$OUTPUT_NAMESPACE}"

    log "============================================="
    log "$EXP_NAME"
    log "  cluster=$CLUSTER  partition=${SUBMIT_PARTITION:-$PARTITION}  gpu=$GPU_INSTANCE  model=${MODEL_ID:-n1.6}"
    log "  train repo=$TRAIN_REPO_DIR"
    log "  variant note: $TRAIN_NOTE"
    log "  run namespace=$OUTPUT_NAMESPACE"
    log "  output=$RUN_CKPT_DIR"
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

    # Optional per-dataset embodiment tags (enum NAMES, parallel to TRAIN_DATASET_NAMES).
    # Enables mixed-embodiment training; falls back to the single NEW_EMBODIMENT tag.
    EMB_TAG_ARGS=()
    if declare -p TRAIN_DATASET_EMBODIMENT_TAGS 2>/dev/null | grep -q "declare -a"; then
        if [ "${#TRAIN_DATASET_EMBODIMENT_TAGS[@]}" -ne "${#DATASET_PATHS[@]}" ]; then
            echo "ERROR: TRAIN_DATASET_EMBODIMENT_TAGS length (${#TRAIN_DATASET_EMBODIMENT_TAGS[@]}) != dataset count (${#DATASET_PATHS[@]})"
            exit 1
        fi
        EMB_TAG_ARGS=(--embodiment-tags "${TRAIN_DATASET_EMBODIMENT_TAGS[@]}")
        log "Embodiment tags: ${TRAIN_DATASET_EMBODIMENT_TAGS[*]}"
    fi

    # ── Per-variant modality config (Python file, copied into experiment dir) ──
    # n1.7 multi-dataset (launch_finetune_multi.py) references its modality
    # configs through the data YAML's per-row `data_config` names, so it does
    # not need a standalone modality .py. Every other path (n1.6, and n1.7
    # single via launch_finetune.py) requires one.
    N17_DATA_YAML="${TRAIN_DATA_YAML:-}"
    if [ "$MODEL_FAMILY" = "n1.7" ] && [ -n "$N17_DATA_YAML" ]; then
        MODALITY_CONFIG_FILE=""
        log "Multi-dataset n1.7: modality configs resolved via data YAML rows"
    else
        : "${TRAIN_MODALITY_CONFIG:?TRAIN_MODALITY_CONFIG not set in config.sh}"
        MODALITY_CONFIG_FILE="$EXP_DIR/$TRAIN_MODALITY_CONFIG"
        [ -f "$MODALITY_CONFIG_FILE" ] || { echo "ERROR: modality config not found: $MODALITY_CONFIG_FILE"; exit 1; }
        log "Modality config: $MODALITY_CONFIG_FILE"
    fi

    # ── Per-device → global batch size (default: keep TRAIN_BATCH_SIZE per-device) ──
    GLOBAL_BATCH_SIZE="${SUBMIT_TRAIN_GLOBAL_BATCH_SIZE:-$((TRAIN_NUM_GPUS * TRAIN_BATCH_SIZE))}"
    TRAIN_ACTION_HORIZON="${SUBMIT_TRAIN_ACTION_HORIZON:-${TRAIN_ACTION_HORIZON:-}}"
    ACTION_HORIZON_MODE="${SUBMIT_ACTION_HORIZON_MODE:-${ACTION_HORIZON_MODE:-modality}}"
    ACTION_HORIZON_ARGS=()
    if [[ -n "$TRAIN_ACTION_HORIZON" && ( "$ACTION_HORIZON_MODE" == "cli" || "$ACTION_HORIZON_MODE" == "modality_and_cli" ) ]]; then
        ACTION_HORIZON_ARGS=(--action-horizon "$TRAIN_ACTION_HORIZON")
    fi
    log "Global batch: $GLOBAL_BATCH_SIZE"
    log "Train GPUs: $TRAIN_NUM_GPUS"
    log "Save steps: $SAVE_STEPS"
    log "Action horizon: ${TRAIN_ACTION_HORIZON:-default} ($ACTION_HORIZON_MODE)"

    if [[ "${RESUME_EXPECTED:-0}" == "1" ]]; then
        if compgen -G "$RUN_CKPT_DIR/checkpoint-*" > /dev/null; then
            LATEST_CKPT=$(ls -d "$RUN_CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
            log "Resume requested; existing checkpoint found: $LATEST_CKPT"
        else
            log "ERROR: resume requested but no checkpoint in $RUN_CKPT_DIR"
            exit 1
        fi
    fi

    if [ -d "$RUN_CKPT_DIR/checkpoint-${MAX_STEPS}" ]; then
        log "Final checkpoint already exists at $RUN_CKPT_DIR/checkpoint-${MAX_STEPS} — skipping training."
        cleanup_trainer_state "$RUN_CKPT_DIR" "$MAX_STEPS"
        exit 0
    fi

    # uv may not be on PATH in non-login shells (it lives at $HOME/.local/bin)
    export PATH="$HOME/.local/bin:$PATH"
    export WANDB_PROJECT="${SUBMIT_WANDB_PROJECT:-${WANDB_PROJECT:-my project}}"
    export WANDB_DIR="$EXP_DIR"
    export WANDB_RESUME=allow

    cd "$TRAIN_REPO_DIR"

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        MASTER_PORT="${MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 40000)))}"
    else
        MASTER_PORT="${MASTER_PORT:-29500}"
    fi
    log "Torchrun master port: $MASTER_PORT"

    if [ "$MODEL_FAMILY" = "n1.7" ]; then
        # GR00T N1.7. The base model is passed explicitly for both launchers
        # (launch_finetune.py requires it; launch_finetune_multi.py would
        # otherwise default to the 2B checkpoint). --output-dir is the parent
        # "checkpoints" dir; --experiment-name makes the trainer write into
        # $CKPT_DIR/$OUTPUT_NAMESPACE, matching the n1.6 layout.
        if [ -n "$N17_DATA_YAML" ]; then
            # Multi-dataset via launch_finetune_multi.py (--data-yaml). Only this
            # launcher accepts --action-horizon.
            N17_AH_ARGS=()
            [ -n "$TRAIN_ACTION_HORIZON" ] && N17_AH_ARGS=(--action-horizon "$TRAIN_ACTION_HORIZON")
            uv run torchrun --nproc_per_node="$TRAIN_NUM_GPUS" --master-port "$MASTER_PORT" gr00t/experiment/launch_finetune_multi.py \
                --base-model-path nvidia/GR00T-N1.7-3B \
                --data-yaml "$EXP_DIR/$N17_DATA_YAML" \
                --num-gpus "$TRAIN_NUM_GPUS" \
                --output-dir "$MODEL_OUTPUT_DIR" \
                --global-batch-size "$GLOBAL_BATCH_SIZE" \
                --learning-rate 1e-4 \
                --max-steps "$MAX_STEPS" \
                --save-steps "$SAVE_STEPS" \
                --save-total-limit 5 \
                --dataloader-num-workers "$TRAIN_NUM_WORKERS" \
                --experiment-name "$OUTPUT_NAMESPACE" \
                --use-wandb \
                --wandb-project "$WANDB_PROJECT" \
                --color-jitter-params brightness 0.2 contrast 0.2 saturation 0.2 hue 0.1 \
                ${N17_AH_ARGS[@]+"${N17_AH_ARGS[@]}"} \
                "${TRAIN_EXTRA_ARGS[@]}"
        else
            # Single-dataset via launch_finetune.py (--dataset-path singular +
            # --modality-config-path; no --action-horizon on this launcher).
            uv run torchrun --nproc_per_node="$TRAIN_NUM_GPUS" --master-port "$MASTER_PORT" gr00t/experiment/launch_finetune.py \
                --base-model-path nvidia/GR00T-N1.7-3B \
                --dataset-path "${DATASET_PATHS[0]}" \
                --embodiment-tag NEW_EMBODIMENT \
                --modality-config-path "$MODALITY_CONFIG_FILE" \
                --num-gpus "$TRAIN_NUM_GPUS" \
                --output-dir "$MODEL_OUTPUT_DIR" \
                --global-batch-size "$GLOBAL_BATCH_SIZE" \
                --learning-rate 1e-4 \
                --max-steps "$MAX_STEPS" \
                --save-steps "$SAVE_STEPS" \
                --save-total-limit 5 \
                --dataloader-num-workers "$TRAIN_NUM_WORKERS" \
                --experiment-name "$OUTPUT_NAMESPACE" \
                --use-wandb \
                --wandb-project "$WANDB_PROJECT" \
                --color-jitter-params brightness 0.2 contrast 0.2 saturation 0.2 hue 0.1 \
                "${TRAIN_EXTRA_ARGS[@]}"
        fi
    else
        uv run torchrun --nproc_per_node="$TRAIN_NUM_GPUS" --master-port "$MASTER_PORT" gr00t/experiment/launch_finetune.py \
            --base-model-path nvidia/GR00T-N1.6-3B \
            --dataset-path "${DATASET_PATHS[@]}" \
            --embodiment-tag NEW_EMBODIMENT \
            ${EMB_TAG_ARGS[@]+"${EMB_TAG_ARGS[@]}"} \
            --modality-config-path "$MODALITY_CONFIG_FILE" \
            --num-gpus "$TRAIN_NUM_GPUS" \
            --output-dir "$MODEL_OUTPUT_DIR" \
            --global-batch-size "$GLOBAL_BATCH_SIZE" \
            --learning-rate 1e-4 \
            --max-steps "$MAX_STEPS" \
            --save-steps "$SAVE_STEPS" \
            --save-total-limit 5 \
            --dataloader-num-workers "$TRAIN_NUM_WORKERS" \
            --experiment-name "$OUTPUT_NAMESPACE" \
            --use-wandb \
            --wandb-project "$WANDB_PROJECT" \
            --color-jitter-params brightness 0.2 contrast 0.2 saturation 0.2 hue 0.1 \
            "${ACTION_HORIZON_ARGS[@]}" \
            "${TRAIN_EXTRA_ARGS[@]}"
    fi

    log "Training completed."
    cleanup_trainer_state "$RUN_CKPT_DIR" "$MAX_STEPS"
fi
