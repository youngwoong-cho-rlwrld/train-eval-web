#!/usr/bin/env bash
# Run by sbatch via the top-level submit wrapper.
# Reads $VARIANT and $CLUSTER from the environment.
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

GPU_INSTANCE="$(detect_gpu_instance)"
EXP_NAME="${VARIANT}_eval_${GPU_INSTANCE}_$(date +%Y%m%d%H%M%S)"

CKPT_DIR="$EXP_DIR/checkpoints"
EVAL_DIR="$EXP_DIR/eval_results"
mkdir -p "$EXP_DIR/logs" "$LOG_DIR" "$EVAL_DIR"
LOG_FILE="$EXP_DIR/logs/eval.log"

log "========================================================"
log "$EXP_NAME"
log "  cluster=$CLUSTER  partition=$PARTITION  gpu=$GPU_INSTANCE"
log "========================================================"

if [ -n "${EVAL_CHECKPOINT:-}" ]; then
    LAST_CKPT="$EVAL_CHECKPOINT"
else
    LAST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
if [ -z "$LAST_CKPT" ] || [ ! -d "$LAST_CKPT" ]; then
    log "ERROR: no checkpoint at '$LAST_CKPT'"
    exit 1
fi
log "Checkpoint: $LAST_CKPT"

# Task mode:
#   (a) TASKS=("short|task_name|instruction" ...) — multi-task eval matrix.
#   (b) TASK_NAME=<task> + INSTRUCTION=<text>    — legacy single-task.
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

# DATASET_NAME may be absent in multi-task variants (they use DATASETS at train time).
if [[ "${DATASET_NAME+set}" == set ]]; then
    DATA_PATH="$DATA_DIR/$DATASET_NAME"
else
    DATA_PATH="(multi-dataset; see $EXP_DIR/data_config.yaml)"
fi

# Auto-detect input resolution from training dataset's meta/info.json.
# gr00t bakes this into the checkpoint's modality config and asserts equality
# at eval time (VideoToTensor.check_input). Mismatched sim -> ckpt = hard fail.
if [[ "${DATASET_NAME+set}" == set ]]; then
    FIRST_DS_PATH="$DATA_DIR/$DATASET_NAME"
elif [[ "${DATASETS+set}" == set && ${#DATASETS[@]} -gt 0 ]]; then
    FIRST_DS_PATH="$DATA_DIR/${DATASETS[0]%%|*}"
else
    FIRST_DS_PATH=""
fi
if [[ -n "$FIRST_DS_PATH" && -f "$FIRST_DS_PATH/meta/info.json" ]]; then
    read EVAL_IMG_H EVAL_IMG_W < <(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
shape = next(v['shape'] for v in d['features'].values() if v.get('dtype') == 'video')
print(shape[1], shape[2])
" "$FIRST_DS_PATH/meta/info.json")
    log "Auto-detected input resolution: ${EVAL_IMG_H}x${EVAL_IMG_W} (from $FIRST_DS_PATH/meta/info.json)"
else
    EVAL_IMG_H=224
    EVAL_IMG_W=224
    log "Could not locate info.json; defaulting input resolution to ${EVAL_IMG_H}x${EVAL_IMG_W}"
fi

###############################################################################
# Phase 2: Evaluation
###############################################################################

SERVER_PID=""
PORT=""

cleanup() {
    [ -n "$SERVER_PID" ] && kill -9 $SERVER_PID 2>/dev/null || true
    [ -n "$PORT" ] && pkill -9 -f "server_v2.py.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

kill_server() {
    log "Killing server (PID=$SERVER_PID, PORT=$PORT)..."
    if [ -n "$SERVER_PID" ]; then
        kill -9 -$SERVER_PID 2>/dev/null || true
        kill -9 $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
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

source "${GROOT_DIR}/.venv/bin/activate"
cd "${GROOT_DIR}"

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

    for EVAL_SET in "${EVAL_SETS[@]}"; do
        for i in $(seq 1 ${N_RUNS}); do
            log ""
            log "  eval_set: ${EVAL_SET} / Run ${i}/${N_RUNS}"

            # Idempotent: skip if this (set, run) already produced a results.json.
            # Useful when re-running after a partial completion (e.g. server hang).
            RUN_DIR="${TASK_EVAL_DIR}/${EVAL_SET}/run_${i}"
            if [ -f "${RUN_DIR}/results.json" ]; then
                log "  SKIP (results.json already exists): ${RUN_DIR}"
                continue
            fi

            PORT=$(find_available_port)
            log "  Isaac Sim server starting on port $PORT"

            setsid bash -c "
                source '${ISAAC_DIR}/.venv/bin/activate'
                cd '${ISAAC_DIR}'
                exec python scripts/environments/server_v2.py \
                    --task 'Isaac-UniPickPlace-ALLEX-JointAction-VisualStereo-Abs-v0' \
                    --task_name '${TASK_NAME_LOOP}' \
                    --max-episode-steps ${MAX_EPISODE_STEPS} \
                    --image_crop_ratio 1.0 \
                    --image_resize_height $EVAL_IMG_H \
                    --image_resize_width $EVAL_IMG_W \
                    --port $PORT \
                    --device cpu \
                    --eval_set $EVAL_SET \
                    --app_launcher.headless
            " > "${EXP_DIR}/logs/server_${SERVER_LOG_TAG}${EVAL_SET}_run${i}.log" 2>&1 &
            SERVER_PID=$!

            log "  Waiting 30s for server startup..."
            sleep 30

            source "${GROOT_DIR}/.venv/bin/activate"

            log "  Running eval -> ${RUN_DIR}"

            python scripts/eval_allex.py \
                --model-path "$LAST_CKPT" \
                --server-port $PORT \
                --output-dir "$RUN_DIR" \
                --instruction "${INSTRUCTION_LOOP}" \
                --n-episodes ${N_EPISODES} \
                --execution_horizon ${EXECUTION_HORIZON} \
                --data_config "${DATA_CONFIG}" \
                --action_type joint_action

            kill_server
            sleep 5
        done
    done
done

###############################################################################
# Phase 3: Aggregate
###############################################################################

log "Aggregating results..."

# Dump the TASKS list as JSON so the python block can iterate it cleanly
# (avoids quoting/escaping instructions through a bash-built python literal).
TASKS_JSON="$EXP_DIR/.eval_tasks.json"
python - "$TASKS_JSON" "${TASKS[@]}" <<'PYDUMP'
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

python - <<PYEOF
import json, numpy as np
from pathlib import Path

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
                with open(p) as fh:
                    rates.append(json.load(fh)['summary']['success_rate'])
            else:
                print(f'WARNING: {p} not found')
        if rates:
            rates = np.array(rates)
            all_results[es] = {
                'per_run_success_rate': rates.tolist(),
                'mean_success_rate': float(np.mean(rates)),
                'std_success_rate': float(np.std(rates)),
            }
            print(f'  {es}: {np.mean(rates):.4f} +/- {np.std(rates):.4f}  {rates}')
    return all_results

agg = {
    'experiment': '${EXP_NAME}',
    'cluster': '${CLUSTER}',
    'gpu': '${GPU_INSTANCE}',
    'note': '${TRAIN_NOTE}',
    'checkpoint': '${LAST_CKPT}',
    'data_config': '${DATA_CONFIG}',
    'dataset': '${DATA_PATH}',
    'n_episodes': ${N_EPISODES},
    'execution_horizon': ${EXECUTION_HORIZON},
    'max_steps': ${MAX_STEPS},
    'n_runs': n_runs,
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

out = Path('${EXP_DIR}') / 'results.json'
with open(out, 'w') as f:
    json.dump(agg, f, indent=2)
print(f'Saved to {out}')
PYEOF

log "DONE  $EXP_DIR/results.json"
