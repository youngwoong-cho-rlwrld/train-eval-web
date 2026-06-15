#!/usr/bin/env bash
# Run by sbatch via the top-level submit wrapper for MODEL_VERSION=n1.6 + PHASE=eval.
# Reads $REPO_ROOT, $CLUSTER, $VARIANT from the environment (set by submit --export).
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=Y
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

: "${REPO_ROOT:?REPO_ROOT must be set by submit wrapper}"
: "${CLUSTER:?CLUSTER must be set by submit wrapper}"
: "${VARIANT:?VARIANT must be set by submit wrapper}"
# Cluster envs still export legacy REPO_ROOT; keep the submitted staging root.
SUBMIT_REPO_ROOT="$REPO_ROOT"
source "$REPO_ROOT/clusters/${CLUSTER}.env"
REPO_ROOT="$SUBMIT_REPO_ROOT"
source "$REPO_ROOT/lib/_common.sh"

EXP_DIR="${SUBMIT_EXP_DIR:-$REPO_ROOT/experiments/$VARIANT}"
[ -d "$EXP_DIR" ] || { echo "ERROR: experiment dir not found: $EXP_DIR"; exit 1; }
CONFIG_FILE="${SUBMIT_CONFIG_FILE:-$EXP_DIR/config.sh}"
[ -f "$CONFIG_FILE" ] || { echo "ERROR: config not found: $CONFIG_FILE"; exit 1; }
source "$CONFIG_FILE"
if [ -n "${SUBMIT_DATA_DIR:-}" ]; then
    DATA_DIR="$SUBMIT_DATA_DIR"
fi
TRAIN_NUM_GPUS="${SUBMIT_TRAIN_NUM_GPUS:-${TRAIN_NUM_GPUS:-1}}"
TRAIN_REPO_DIR="${SUBMIT_TRAIN_REPO_DIR:-${TRAIN_REPO_DIR:-$GROOT_N16_DIR}}"

GPU_INSTANCE="$(detect_gpu_instance)"
EXP_NAME="${SLURM_JOB_NAME:-${VARIANT}_eval_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)}"
OUTPUT_NAMESPACE="${SUBMIT_OUTPUT_NAMESPACE:-}"

CKPT_DIR="$EXP_DIR/checkpoints"
if [ -n "${SUBMIT_EVAL_DIR:-}" ]; then
    EVAL_DIR="$SUBMIT_EVAL_DIR"
elif [ -n "$OUTPUT_NAMESPACE" ]; then
    EVAL_DIR="$EXP_DIR/eval_results/$OUTPUT_NAMESPACE"
else
    EVAL_DIR="$EXP_DIR/eval_results"
fi
RESULTS_PATH="${SUBMIT_RESULTS_PATH:-$EVAL_DIR/results.json}"
JOB_LOG_DIR="$EXP_DIR/logs/${OUTPUT_NAMESPACE:-${SLURM_JOB_ID:-$EXP_NAME}}"
mkdir -p "$JOB_LOG_DIR" "$LOG_DIR" "$EVAL_DIR"
LOG_FILE="$JOB_LOG_DIR/eval.log"
SUBMIT_GIT_COMMIT="${SUBMIT_GIT_COMMIT:-${TRAIN_GIT_COMMIT:-}}"
pin_training_repo_dir "$TRAIN_REPO_DIR" "$SUBMIT_GIT_COMMIT" "${SLURM_JOB_ID:-$OUTPUT_NAMESPACE}"

log "========================================================"
log "$EXP_NAME"
log "  cluster=$CLUSTER  partition=${SUBMIT_PARTITION:-$PARTITION}  gpu=$GPU_INSTANCE  model=${MODEL_ID:-n1.6}"
log "  train repo=$TRAIN_REPO_DIR"
log "  run namespace=${OUTPUT_NAMESPACE:-legacy}"
log "  eval results=$EVAL_DIR"
log "========================================================"

if [ -z "${EVAL_CHECKPOINT:-}" ]; then
    log "ERROR: EVAL_CHECKPOINT is required"
    exit 1
fi
LAST_CKPT="$EVAL_CHECKPOINT"
if [ -z "$LAST_CKPT" ] || [ ! -d "$LAST_CKPT" ]; then
    log "ERROR: no checkpoint at '$LAST_CKPT'"
    exit 1
fi
log "Checkpoint: $LAST_CKPT"

# Per-variant Python modality config (relative to experiment dir, copied at variant-creation time)
: "${TRAIN_MODALITY_CONFIG:?TRAIN_MODALITY_CONFIG not set in config.sh}"
MODALITY_CONFIG_FILE="$EXP_DIR/$TRAIN_MODALITY_CONFIG"
[ -f "$MODALITY_CONFIG_FILE" ] || { echo "ERROR: modality config not found: $MODALITY_CONFIG_FILE"; exit 1; }
log "Modality config: $MODALITY_CONFIG_FILE"

# Task mode:
#   (a) TASKS=("short|task_name|instruction" ...) - multi-task eval matrix.
#   (b) TASK_NAME=<task> + INSTRUCTION=<text>    - single-task eval.
# Synthesize a one-element list for single-task so the loop below handles both.
if [[ "${TASKS+set}" == set ]] && [ "${#TASKS[@]}" -gt 0 ]; then
    MULTI_TASK=1
    log "Mode: multi-task over ${#TASKS[@]} tasks"
    for entry in "${TASKS[@]}"; do
        IFS='|' read -r tshort tname _tinstr <<<"$entry"
        log "  - $tshort  ($tname)"
    done
else
    MULTI_TASK=0
    TASKS=("__single__|${TASK_NAME}|${INSTRUCTION}")
    log "Mode: single-task ($TASK_NAME)"
fi

###############################################################################
# Phase 2: Evaluation (Isaac Sim server + N1.6 eval client)
###############################################################################

if [ -n "${SUBMIT_EVAL_N_EPISODES:-}" ]; then
    N_EPISODES="$SUBMIT_EVAL_N_EPISODES"
fi
if [ -n "${SUBMIT_EVAL_N_RUNS:-}" ]; then
    N_RUNS="$SUBMIT_EVAL_N_RUNS"
fi
if [ -n "${SUBMIT_EVAL_SETS:-}" ]; then
    read -r -a EVAL_SETS <<< "$SUBMIT_EVAL_SETS"
fi
if ! [[ "${N_EPISODES:-}" =~ ^[0-9]+$ ]] || [ "$N_EPISODES" -lt 1 ]; then
    log "ERROR: N_EPISODES must be a positive integer, got '${N_EPISODES:-}'"
    exit 1
fi
if ! [[ "${N_RUNS:-}" =~ ^[0-9]+$ ]] || [ "$N_RUNS" -lt 1 ]; then
    log "ERROR: N_RUNS must be a positive integer, got '${N_RUNS:-}'"
    exit 1
fi
if [[ "${EVAL_SETS+set}" != set ]] || [ "${#EVAL_SETS[@]}" -eq 0 ]; then
    log "ERROR: EVAL_SETS must contain at least one eval set"
    exit 1
fi
log "Eval shape: ${N_RUNS} runs x ${N_EPISODES} episodes; eval_sets=${EVAL_SETS[*]}"

EVAL_OVERWRITE_RESULTS="${SUBMIT_EVAL_OVERWRITE_RESULTS:-${EVAL_OVERWRITE_RESULTS:-0}}"
if [ "$EVAL_OVERWRITE_RESULTS" != "0" ] && [ "$EVAL_OVERWRITE_RESULTS" != "1" ]; then
    log "ERROR: EVAL_OVERWRITE_RESULTS must be 0 or 1, got '$EVAL_OVERWRITE_RESULTS'"
    exit 1
fi
if [ "$EVAL_OVERWRITE_RESULTS" = "1" ]; then
    log "Overwrite existing evaluation results: enabled"
    rm -f "$RESULTS_PATH"
fi

EVAL_BASE_SEED="${EVAL_BASE_SEED:-42}"
if ! [[ "$EVAL_BASE_SEED" =~ ^[0-9]+$ ]]; then
    log "ERROR: EVAL_BASE_SEED must be a non-negative integer, got '$EVAL_BASE_SEED'"
    exit 1
fi
EVAL_SEED_RUN_STRIDE="${EVAL_SEED_RUN_STRIDE:-$((N_EPISODES + 1))}"
EVAL_SEED_SET_STRIDE="${EVAL_SEED_SET_STRIDE:-$((N_RUNS * EVAL_SEED_RUN_STRIDE + 1))}"
EVAL_SEED_TASK_STRIDE="${EVAL_SEED_TASK_STRIDE:-$((${#EVAL_SETS[@]} * EVAL_SEED_SET_STRIDE + 1))}"
for seed_var in EVAL_SEED_RUN_STRIDE EVAL_SEED_SET_STRIDE EVAL_SEED_TASK_STRIDE; do
    seed_val="${!seed_var}"
    if ! [[ "$seed_val" =~ ^[0-9]+$ ]] || [ "$seed_val" -lt 1 ]; then
        log "ERROR: $seed_var must be a positive integer, got '$seed_val'"
        exit 1
    fi
done
log "Eval seed ranges: base=$EVAL_BASE_SEED run_stride=$EVAL_SEED_RUN_STRIDE eval_set_stride=$EVAL_SEED_SET_STRIDE task_stride=$EVAL_SEED_TASK_STRIDE"

EVAL_GPU_COUNT="${TRAIN_NUM_GPUS:-1}"
if ! [[ "$EVAL_GPU_COUNT" =~ ^[0-9]+$ ]] || [ "$EVAL_GPU_COUNT" -lt 1 ]; then
    EVAL_GPU_COUNT=1
fi

if [ -n "${SUBMIT_EVAL_NUM_ENVS_PER_GPU:-}" ]; then
    EVAL_NUM_ENVS_PER_GPU="$SUBMIT_EVAL_NUM_ENVS_PER_GPU"
elif [ -n "${SUBMIT_EVAL_PARALLEL_SIMS_PER_GPU:-}" ]; then
    EVAL_NUM_ENVS_PER_GPU="$SUBMIT_EVAL_PARALLEL_SIMS_PER_GPU"
fi
EVAL_NUM_ENVS_PER_GPU="${EVAL_NUM_ENVS_PER_GPU:-${EVAL_NATIVE_NUM_ENVS_PER_SERVER:-${EVAL_NUM_ENVS_PER_SERVER:-${EVAL_PARALLEL_SIMS_PER_GPU:-${EVAL_PARALLEL_SIMS:-1}}}}}"
if ! [[ "$EVAL_NUM_ENVS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$EVAL_NUM_ENVS_PER_GPU" -lt 1 ]; then
    log "ERROR: EVAL_NUM_ENVS_PER_GPU must be a positive integer, got '$EVAL_NUM_ENVS_PER_GPU'"
    exit 1
fi
EVAL_REQUESTED_NUM_ENVS_PER_GPU="$EVAL_NUM_ENVS_PER_GPU"
if [ "$EVAL_NUM_ENVS_PER_GPU" -gt 1 ]; then
    log "EVAL_NUM_ENVS_PER_GPU=$EVAL_NUM_ENVS_PER_GPU requested; using 1 because ALLEX target reset is not vector-env safe"
    EVAL_NUM_ENVS_PER_GPU=1
fi
if [[ "${N_EPISODES:-}" =~ ^[0-9]+$ ]] && [ "$N_EPISODES" -gt 0 ] && [ "$EVAL_NUM_ENVS_PER_GPU" -gt "$N_EPISODES" ]; then
    log "Requested EVAL_NUM_ENVS_PER_GPU=$EVAL_NUM_ENVS_PER_GPU exceeds N_EPISODES=$N_EPISODES; using $N_EPISODES"
    EVAL_NUM_ENVS_PER_GPU="$N_EPISODES"
fi
EVAL_PARALLEL_WORKERS="$EVAL_GPU_COUNT"
EVAL_TOTAL_NUM_ENVS=$((EVAL_NUM_ENVS_PER_GPU * EVAL_PARALLEL_WORKERS))
log "Isaac native envs: $EVAL_TOTAL_NUM_ENVS total (${EVAL_NUM_ENVS_PER_GPU} per GPU x $EVAL_GPU_COUNT GPUs)"
log "Isaac server workers: $EVAL_PARALLEL_WORKERS x ${EVAL_NUM_ENVS_PER_GPU} native envs"
EVAL_PIN_CUDA_DEVICES="${EVAL_PIN_CUDA_DEVICES:-1}"
if [ "$EVAL_PIN_CUDA_DEVICES" != "0" ] && [ "$EVAL_PIN_CUDA_DEVICES" != "1" ]; then
    log "ERROR: EVAL_PIN_CUDA_DEVICES must be 0 or 1, got '$EVAL_PIN_CUDA_DEVICES'"
    exit 1
fi
EVAL_PIN_CLIENT_CUDA_DEVICES="${EVAL_PIN_CLIENT_CUDA_DEVICES:-1}"
if [ "$EVAL_PIN_CLIENT_CUDA_DEVICES" != "0" ] && [ "$EVAL_PIN_CLIENT_CUDA_DEVICES" != "1" ]; then
    log "ERROR: EVAL_PIN_CLIENT_CUDA_DEVICES must be 0 or 1, got '$EVAL_PIN_CLIENT_CUDA_DEVICES'"
    exit 1
fi
EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER="${SUBMIT_EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER:-${EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER:-0}}"
if [ "$EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER" != "0" ] && [ "$EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER" != "1" ]; then
    log "ERROR: EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER must be 0 or 1, got '$EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER'"
    exit 1
fi

EVAL_SIM_START_STAGGER_SECONDS="${EVAL_SIM_START_STAGGER_SECONDS:-10}"
if ! [[ "$EVAL_SIM_START_STAGGER_SECONDS" =~ ^[0-9]+$ ]]; then
    log "ERROR: EVAL_SIM_START_STAGGER_SECONDS must be a non-negative integer, got '$EVAL_SIM_START_STAGGER_SECONDS'"
    exit 1
fi
if [ -z "${EVAL_SERVER_READY_TIMEOUT_SECONDS:-}" ]; then
    # Cold Isaac kit/shader caches on freshly provisioned nodes (skt dy-*)
    # routinely push server startup past 280s, especially when several jobs
    # cold-start against the shared filesystem at once. Crashed servers are
    # detected separately, so a generous deadline only delays true hangs.
    EVAL_SERVER_READY_TIMEOUT_SECONDS=$((900 + EVAL_PARALLEL_WORKERS * EVAL_SIM_START_STAGGER_SECONDS))
fi
if ! [[ "$EVAL_SERVER_READY_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || [ "$EVAL_SERVER_READY_TIMEOUT_SECONDS" -lt 1 ]; then
    log "ERROR: EVAL_SERVER_READY_TIMEOUT_SECONDS must be a positive integer, got '$EVAL_SERVER_READY_TIMEOUT_SECONDS'"
    exit 1
fi

PIDS=()
PORTS=()
FAILED=0
LAUNCH_IDX=0
EVAL_LAUNCHED=0

cleanup_all() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_all EXIT
trap 'cleanup_all; exit 130' INT TERM

refresh_running_pids() {
    local status=0
    local pid
    local running_pid
    local is_running
    local running_pids=()
    local active_pids=()

    mapfile -t running_pids < <(jobs -pr)
    for pid in "${PIDS[@]}"; do
        is_running=0
        for running_pid in "${running_pids[@]}"; do
            if [ "$pid" = "$running_pid" ]; then
                is_running=1
                break
            fi
        done
        if [ "$is_running" -eq 1 ]; then
            active_pids+=("$pid")
        elif wait "$pid"; then
            :
        else
            local worker_status=$?
            log "ERROR: eval worker pid $pid exited with status $worker_status"
            FAILED=1
            status=1
        fi
    done
    PIDS=("${active_pids[@]}")
    return "$status"
}

wait_for_slot() {
    while true; do
        if ! refresh_running_pids; then
            return 1
        fi
        if [ "${#PIDS[@]}" -lt "$EVAL_PARALLEL_WORKERS" ]; then
            return 0
        fi
        sleep 2
    done
}

wait_for_all() {
    local status=0
    while true; do
        if ! refresh_running_pids; then
            status=1
        fi
        if [ "${#PIDS[@]}" -eq 0 ]; then
            break
        fi
        sleep 2
    done
    PIDS=()
    return "$status"
}

find_eval_port() {
    local port
    local existing
    local used
    while true; do
        port="$(find_available_port)"
        used=0
        for existing in "${PORTS[@]}"; do
            if [ "$existing" = "$port" ]; then
                used=1
                break
            fi
        done
        if [ "$used" -eq 0 ]; then
            PORTS+=("$port")
            echo "$port"
            return 0
        fi
    done
}

select_cuda_device() {
    local slot="$1"
    local visible="${CUDA_VISIBLE_DEVICES:-}"
    if [ -n "$visible" ]; then
        IFS=',' read -r -a devices <<< "$visible"
        local count="${#devices[@]}"
        if [ "$count" -gt 0 ]; then
            echo "${devices[$((slot % count))]}"
            return 0
        fi
    fi
    local gpu_count="${TRAIN_NUM_GPUS:-}"
    if [[ "$gpu_count" =~ ^[0-9]+$ ]] && [ "$gpu_count" -gt 0 ]; then
        echo "$((slot % gpu_count))"
    else
        echo "$slot"
    fi
}

run_eval_one() (
    set -euo pipefail
    local TASK_SHORT_LOOP="$1"
    local TASK_NAME_LOOP="$2"
    local INSTRUCTION_LOOP="$3"
    local TASK_EVAL_DIR="$4"
    local SERVER_LOG_TAG="$5"
    local EVAL_SET="$6"
    local RUN_IDX="$7"
    local RUN_SEED="$8"
    local GPU_SLOT="$9"
    local PORT="${10}"
    local START_SLOT="${11}"
    local RUN_DIR="${TASK_EVAL_DIR}/${EVAL_SET}/run_${RUN_IDX}"
    local SERVER_PID=""
    local SERVER_LOG="${JOB_LOG_DIR}/server_${SERVER_LOG_TAG}${EVAL_SET}_run${RUN_IDX}.log"

    cleanup() {
        [ -n "$SERVER_PID" ] && kill -9 -"$SERVER_PID" 2>/dev/null || true
        [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
        [ -n "$PORT" ] && pkill -9 -f "server_v2.py.*--port $PORT" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'cleanup; exit 130' INT TERM

    kill_server() {
        log "Killing server (PID=$SERVER_PID, PORT=$PORT)..."
        if [ -n "$SERVER_PID" ]; then
            kill -9 -"$SERVER_PID" 2>/dev/null || true
            kill -9 "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
        fi
        if [ -n "$PORT" ]; then
            pkill -9 -f "server_v2.py.*--port $PORT" 2>/dev/null || true
        fi
        for attempt in $(seq 1 10); do
            if ! ss -tuln | grep -q ":$PORT "; then break; fi
            sleep 2
        done
        SERVER_PID=""
        log "Server stopped."
    }

    dump_server_tail() {
        if [ -f "$SERVER_LOG" ]; then
            tail -n 80 "$SERVER_LOG" | sed 's/^/[server] /' | tee -a "$LOG_FILE" || true
        fi
    }

    server_crashed() {
        grep -Eq "GPU crash is detected|VkResult: ERROR_DEVICE_LOST|\\[Fatal\\].*\\[crash\\]|A GPU crash occurred" "$SERVER_LOG" 2>/dev/null
    }

    wait_for_server_ready() {
        local elapsed=0
        while [ "$elapsed" -le "$EVAL_SERVER_READY_TIMEOUT_SECONDS" ]; do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                log "ERROR: Isaac Sim server exited before ready (port=$PORT)"
                dump_server_tail
                return 1
            fi
            if grep -q "Allex env server is ready to accept requests" "$SERVER_LOG" 2>/dev/null; then
                log "  Isaac Sim server ready on port $PORT"
                return 0
            fi
            if server_crashed; then
                log "ERROR: Isaac Sim server crashed during startup (port=$PORT)"
                dump_server_tail
                return 1
            fi
            sleep 2
            elapsed=$((elapsed + 2))
        done
        log "ERROR: Isaac Sim server was not ready after ${EVAL_SERVER_READY_TIMEOUT_SECONDS}s (port=$PORT)"
        dump_server_tail
        return 1
    }

    log ""
    log "  eval_set: ${EVAL_SET} / Run ${RUN_IDX}/${N_RUNS} / seed ${RUN_SEED}"

    if [ "$EVAL_OVERWRITE_RESULTS" = "1" ] && [ -e "$RUN_DIR" ]; then
        log "  OVERWRITE: removing existing results at ${RUN_DIR}"
        rm -rf -- "$RUN_DIR"
    fi

    # Idempotent: skip if this (set, run) already produced a results.json.
    if [ -f "${RUN_DIR}/results.json" ]; then
        log "  SKIP (results.json already exists): ${RUN_DIR}"
        exit 0
    fi

    local start_delay=$((START_SLOT * EVAL_SIM_START_STAGGER_SECONDS))
    if [ "$start_delay" -gt 0 ]; then
        log "  Staggering server startup by ${start_delay}s"
        sleep "$start_delay"
    fi

    local worker_cuda_device
    worker_cuda_device="$(select_cuda_device "$GPU_SLOT")"
    local cuda_devices="${CUDA_VISIBLE_DEVICES:-<unset>}"
    if [ "$EVAL_PIN_CUDA_DEVICES" = "1" ] || [ "$EVAL_PIN_CLIENT_CUDA_DEVICES" = "1" ]; then
        export CUDA_VISIBLE_DEVICES="$worker_cuda_device"
        cuda_devices="$CUDA_VISIBLE_DEVICES"
    fi
    local server_cuda_devices="$cuda_devices"
    if [ "$EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER" = "1" ]; then
        server_cuda_devices="<unset>"
    fi
    log "  Isaac Sim server starting on port $PORT with CUDA_VISIBLE_DEVICES=${server_cuda_devices}, num_envs=${EVAL_NUM_ENVS_PER_GPU}"

    # Server: run from rlwrld_isaac venv (Python 3.11 + isaac-sim).
    setsid bash -c "
        if [ '${EVAL_UNSET_CUDA_VISIBLE_DEVICES_FOR_SERVER}' = '1' ]; then
            unset CUDA_VISIBLE_DEVICES
        fi
        source '${ISAAC_DIR}/.venv/bin/activate'
        cd '${ISAAC_DIR}'
        exec python '${REPO_ROOT}/lib/isaac_server_runner.py' scripts/environments/server_v2.py \
            --task 'Isaac-UniPickPlace-ALLEX-JointAction-VisualStereo-Abs-v0' \
            --task_name '${TASK_NAME_LOOP}' \
            --max-episode-steps ${MAX_EPISODE_STEPS} \
            --image_crop_ratio 1.0 \
            --image_resize_height 480 \
            --image_resize_width 640 \
            --port $PORT \
            --num_envs ${EVAL_NUM_ENVS_PER_GPU} \
            --device cpu \
            --eval_set "$EVAL_SET" \
            --app_launcher.headless
    " > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    log "  Waiting for server readiness..."
    wait_for_server_ready

    # Client: gr00t-n16 venv (uv run handles env activation; PATH carries uv).
    export PATH="$HOME/.local/bin:$PATH"
    cd "$TRAIN_REPO_DIR"

    log "  Running eval on CUDA_VISIBLE_DEVICES=${cuda_devices} -> ${RUN_DIR}"
    uv run --extra allex python scripts/eval_allex.py \
        --model-path "$LAST_CKPT" \
        --modality-config "$MODALITY_CONFIG_FILE" \
        --embodiment-tag NEW_EMBODIMENT \
        --server-host localhost \
        --server-port $PORT \
        --output-dir "$RUN_DIR" \
        --instruction "${INSTRUCTION_LOOP}" \
        --n-episodes ${N_EPISODES} \
        --execution-horizon ${EXECUTION_HORIZON} \
        --seed ${RUN_SEED} &
    local CLIENT_PID=$!
    while kill -0 "$CLIENT_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: Isaac Sim server exited during eval (port=$PORT)"
            dump_server_tail
            kill "$CLIENT_PID" 2>/dev/null || true
            wait "$CLIENT_PID" 2>/dev/null || true
            return 1
        fi
        if server_crashed; then
            log "ERROR: Isaac Sim server crashed during eval (port=$PORT)"
            dump_server_tail
            kill "$CLIENT_PID" 2>/dev/null || true
            wait "$CLIENT_PID" 2>/dev/null || true
            return 1
        fi
        sleep 2
    done
    if ! wait "$CLIENT_PID"; then
        log "ERROR: eval client failed"
        return 1
    fi
    if [ ! -f "${RUN_DIR}/results.json" ]; then
        log "ERROR: eval client exited successfully but did not write ${RUN_DIR}/results.json"
        kill_server || true
        return 1
    fi

    kill_server || cleanup || true
    trap - EXIT
    log "  Completed eval_set: ${EVAL_SET} / Run ${RUN_IDX}/${N_RUNS}"
    sleep 5
    return 0
)

TASK_IDX=0
for task_entry in "${TASKS[@]}"; do
    IFS='|' read -r TASK_SHORT_LOOP TASK_NAME_LOOP INSTRUCTION_LOOP <<<"$task_entry"
    if [ "$MULTI_TASK" -eq 1 ]; then
        TASK_EVAL_DIR="${EVAL_DIR}/${TASK_SHORT_LOOP}"
        log ""
        log "========== Task: $TASK_SHORT_LOOP ($TASK_NAME_LOOP) =========="
        SERVER_LOG_TAG="${TASK_SHORT_LOOP}_"
    else
        TASK_EVAL_DIR="${EVAL_DIR}"
        SERVER_LOG_TAG=""
    fi
    mkdir -p "$TASK_EVAL_DIR"

    EVAL_SET_IDX=0
    for EVAL_SET in "${EVAL_SETS[@]}"; do
        for i in $(seq 1 ${N_RUNS}); do
            RUN_DIR="${TASK_EVAL_DIR}/${EVAL_SET}/run_${i}"
            RUN_SEED=$((EVAL_BASE_SEED + TASK_IDX * EVAL_SEED_TASK_STRIDE + EVAL_SET_IDX * EVAL_SEED_SET_STRIDE + (i - 1) * EVAL_SEED_RUN_STRIDE))
            if [ "$EVAL_OVERWRITE_RESULTS" != "1" ] && [ -f "${RUN_DIR}/results.json" ]; then
                log ""
                log "  eval_set: ${EVAL_SET} / Run ${i}/${N_RUNS} / seed ${RUN_SEED}"
                log "  SKIP (results.json already exists): ${RUN_DIR}"
                continue
            fi
            wait_for_slot
            GPU_SLOT="$LAUNCH_IDX"
            START_SLOT="$((LAUNCH_IDX % EVAL_PARALLEL_WORKERS))"
            PORT="$(find_eval_port)"
            LAUNCH_IDX=$((LAUNCH_IDX + 1))
            run_eval_one \
                "$TASK_SHORT_LOOP" \
                "$TASK_NAME_LOOP" \
                "$INSTRUCTION_LOOP" \
                "$TASK_EVAL_DIR" \
                "$SERVER_LOG_TAG" \
                "$EVAL_SET" \
                "$i" \
                "$RUN_SEED" \
                "$GPU_SLOT" \
                "$PORT" \
                "$START_SLOT" &
            PIDS+=("$!")
            EVAL_LAUNCHED=$((EVAL_LAUNCHED + 1))
        done
        EVAL_SET_IDX=$((EVAL_SET_IDX + 1))
    done
    TASK_IDX=$((TASK_IDX + 1))
done

if ! wait_for_all; then
    FAILED=1
fi
finish_eval_launch_phase "$EVAL_LAUNCHED" "$FAILED" "$RESULTS_PATH"

###############################################################################
# Phase 3: Aggregate
###############################################################################

log "Aggregating results..."

# Dump TASKS as JSON so Python can iterate without bash quoting issues.
TASKS_JSON="$EVAL_DIR/.eval_tasks.json"
python3 - "$TASKS_JSON" "${TASKS[@]}" <<'PYDUMP'
import json, sys
out_path = sys.argv[1]
tasks = []
for entry in sys.argv[2:]:
    parts = entry.split('|', 2)
    tasks.append({'short': parts[0], 'task_name': parts[1], 'instruction': parts[2]})
with open(out_path, 'w') as f:
    json.dump(tasks, f)
PYDUMP

EVAL_SETS_STR=$(printf "'%s', " "${EVAL_SETS[@]}")
EVAL_SETS_STR="[${EVAL_SETS_STR%, }]"

python3 - <<PYEOF
import json
from pathlib import Path
import statistics

base = Path('${EVAL_DIR}')
eval_sets = ${EVAL_SETS_STR}
n_runs = ${N_RUNS}
multi_task = bool(${MULTI_TASK})

with open('${TASKS_JSON}') as f:
    tasks = json.load(f)

def aggregate_task(task_eval_dir):
    all_results = {}
    for es in eval_sets:
        rates = []
        for i in range(1, n_runs + 1):
            p = task_eval_dir / es / f'run_{i}' / 'results.json'
            if p.exists():
                with open(p) as f:
                    rates.append(json.load(f)['summary']['success_rate'])
            else:
                print(f'WARNING: {p} not found')
        if rates:
            mean = statistics.mean(rates)
            std = statistics.pstdev(rates) if len(rates) > 1 else 0.0
            all_results[es] = {
                'per_run_success_rate': rates,
                'mean_success_rate': mean,
                'std_success_rate': std,
            }
            print(f'  {es}: {mean:.4f} +/- {std:.4f}  {rates}')
    return all_results

agg = {
    'experiment': '${EXP_NAME}',
    'output_namespace': '${OUTPUT_NAMESPACE}',
    'cluster': '${CLUSTER}',
    'gpu': '${GPU_INSTANCE}',
    'model_version': '${MODEL_ID:-n1.6}',
    'note': '${TRAIN_NOTE}',
    'checkpoint': '${LAST_CKPT}',
    'modality_config': '${MODALITY_CONFIG_FILE}',
    'n_episodes': ${N_EPISODES},
    'execution_horizon': ${EXECUTION_HORIZON},
    'max_steps': ${MAX_STEPS},
    'n_runs': n_runs,
    'server_workers': ${EVAL_PARALLEL_WORKERS},
    'requested_num_envs_per_gpu': ${EVAL_REQUESTED_NUM_ENVS_PER_GPU},
    'num_envs_per_gpu': ${EVAL_NUM_ENVS_PER_GPU},
    'total_num_envs': ${EVAL_TOTAL_NUM_ENVS},
    'eval_base_seed': ${EVAL_BASE_SEED},
    'eval_seed_run_stride': ${EVAL_SEED_RUN_STRIDE},
    'eval_seed_set_stride': ${EVAL_SEED_SET_STRIDE},
    'eval_seed_task_stride': ${EVAL_SEED_TASK_STRIDE},
}

if multi_task:
    tasks_out = {}
    for t in tasks:
        ts = t['short']
        print(f'=== {ts} ({t["task_name"]}) ===')
        tasks_out[ts] = {
            'task_name': t['task_name'],
            'instruction': t['instruction'],
            'eval_sets': aggregate_task(base / ts),
        }
    agg['tasks'] = tasks_out
else:
    agg['task_name'] = tasks[0]['task_name']
    agg['eval_sets'] = aggregate_task(base)

out = Path('${RESULTS_PATH}')
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(agg, f, indent=2)
print(f'Saved to {out}')
PYEOF

log "DONE  $RESULTS_PATH"
