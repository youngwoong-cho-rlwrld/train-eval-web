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

EXP_DIR="$REPO_ROOT/experiments/$VARIANT"
[ -d "$EXP_DIR" ] || { echo "ERROR: experiment dir not found: $EXP_DIR"; exit 1; }
source "$EXP_DIR/config.sh"

GPU_INSTANCE="$(detect_gpu_instance)"
EXP_NAME="${SLURM_JOB_NAME:-${VARIANT}_eval_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)}"

CKPT_DIR="$EXP_DIR/checkpoints"
EVAL_DIR="$EXP_DIR/eval_results"
mkdir -p "$EXP_DIR/logs" "$LOG_DIR" "$EVAL_DIR"
LOG_FILE="$EXP_DIR/logs/eval.log"

log "========================================================"
log "$EXP_NAME"
log "  cluster=$CLUSTER  partition=$PARTITION  gpu=$GPU_INSTANCE  model=n1.6"
log "========================================================"

# Honor explicit `--export EVAL_CHECKPOINT=<path>` from the submit wrapper.
# Otherwise auto-pick: N1.6 nests under <experiment_name>/checkpoint-N (from
# launch_finetune.py's --experiment-name); fall back to the flat layout.
if [ -n "${EVAL_CHECKPOINT:-}" ]; then
    LAST_CKPT="$EVAL_CHECKPOINT"
else
    LAST_CKPT="$(ls -d "$CKPT_DIR"/*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
    if [ -z "$LAST_CKPT" ]; then
        LAST_CKPT="$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
    fi
fi
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

###############################################################################
# Phase 2: Evaluation (Isaac Sim server + N1.6 eval client)
###############################################################################

SERVER_PID=""
PORT=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
    [ -n "$PORT" ] && pkill -9 -f "server_v2.py.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

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

for EVAL_SET in "${EVAL_SETS[@]}"; do
    for i in $(seq 1 ${N_RUNS}); do
        log ""
        log "  eval_set: ${EVAL_SET} / Run ${i}/${N_RUNS}"

        # Idempotent: skip if this (set, run) already produced a results.json.
        RUN_DIR="${EVAL_DIR}/${EVAL_SET}/run_${i}"
        if [ -f "${RUN_DIR}/results.json" ]; then
            log "  SKIP (results.json already exists): ${RUN_DIR}"
            continue
        fi

        PORT=$(find_available_port)
        log "  Isaac Sim server starting on port $PORT"

        # Server: run from rlwrld_isaac venv (Python 3.11 + isaac-sim).
        setsid bash -c "
            source '${ISAAC_DIR}/.venv/bin/activate'
            cd '${ISAAC_DIR}'
            exec python scripts/environments/server_v2.py \
                --task 'Isaac-UniPickPlace-ALLEX-JointAction-VisualStereo-Abs-v0' \
                --task_name '${TASK_NAME}' \
                --max-episode-steps ${MAX_EPISODE_STEPS} \
                --image_crop_ratio 1.0 \
                --image_resize_height 480 \
                --image_resize_width 640 \
                --port $PORT \
                --device cpu \
                --eval_set $EVAL_SET \
                --app_launcher.headless
        " > "${EXP_DIR}/logs/server_${EVAL_SET}_run${i}.log" 2>&1 &
        SERVER_PID=$!

        log "  Waiting 30s for server startup..."
        sleep 30

        # Client: gr00t-n16 venv (uv run handles env activation; PATH carries uv).
        export PATH="$HOME/.local/bin:$PATH"
        cd "$GROOT_N16_DIR"

        log "  Running eval -> ${RUN_DIR}"
        uv run python scripts/eval_allex.py \
            --model-path "$LAST_CKPT" \
            --modality-config "$MODALITY_CONFIG_FILE" \
            --embodiment-tag NEW_EMBODIMENT \
            --server-host localhost \
            --server-port $PORT \
            --output-dir "$RUN_DIR" \
            --instruction "${INSTRUCTION}" \
            --n-episodes ${N_EPISODES} \
            --execution-horizon ${EXECUTION_HORIZON} \
            --seed 42

        kill_server
        sleep 5
    done
done

###############################################################################
# Phase 3: Aggregate
###############################################################################

log "Aggregating results..."

EVAL_SETS_STR=$(printf "'%s', " "${EVAL_SETS[@]}")
EVAL_SETS_STR="[${EVAL_SETS_STR%, }]"

python3 - <<PYEOF
import json
from pathlib import Path
import statistics

base = Path('${EVAL_DIR}')
eval_sets = ${EVAL_SETS_STR}
n_runs = ${N_RUNS}
all_results = {}

for es in eval_sets:
    rates = []
    for i in range(1, n_runs + 1):
        p = base / es / f'run_{i}' / 'results.json'
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
        print(f'{es}: {mean:.4f} +/- {std:.4f}  {rates}')

agg = {
    'experiment': '${EXP_NAME}',
    'cluster': '${CLUSTER}',
    'gpu': '${GPU_INSTANCE}',
    'model_version': 'n1.6',
    'note': '${TRAIN_NOTE}',
    'checkpoint': '${LAST_CKPT}',
    'modality_config': '${MODALITY_CONFIG_FILE}',
    'task_name': '${TASK_NAME}',
    'n_episodes': ${N_EPISODES},
    'execution_horizon': ${EXECUTION_HORIZON},
    'max_steps': ${MAX_STEPS},
    'n_runs': n_runs,
    'eval_sets': all_results,
}
out = Path('${EXP_DIR}') / 'results.json'
with open(out, 'w') as f:
    json.dump(agg, f, indent=2)
print(f'Saved to {out}')
PYEOF

log "DONE  $EXP_DIR/results.json"
