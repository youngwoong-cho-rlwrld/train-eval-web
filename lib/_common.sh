# Helpers shared by train_body.sh and eval_body.sh.
# Sourced — does not run on its own.

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE:-/dev/null}"
}

detect_gpu_instance() {
    if [ -n "${SUBMIT_GPU_INSTANCE:-}" ]; then
        echo "$SUBMIT_GPU_INSTANCE"
        return 0
    fi

    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        local job_info node_info slurm_gpu
        job_info="$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null || true)"
        slurm_gpu="$(printf '%s\n' "$job_info" | grep -oE 'gres/gpu:[A-Za-z0-9_.-]+' | head -1 | cut -d: -f2)"
        if [ -n "$slurm_gpu" ]; then
            echo "$slurm_gpu"
            return 0
        fi
        if [ -n "${SLURMD_NODENAME:-}" ]; then
            node_info="$(scontrol show node "$SLURMD_NODENAME" -o 2>/dev/null || true)"
            slurm_gpu="$(printf '%s\n' "$node_info" | grep -oE 'Gres=[^ ]*gpu:[A-Za-z0-9_.-]+' | head -1 | sed -E 's/.*gpu:([^:, ]+).*/\1/')"
            if [ -n "$slurm_gpu" ]; then
                echo "$slurm_gpu"
                return 0
            fi
        fi
    fi

    local n
    n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    case "$n" in
        *A100*) echo a100 ;;
        *H100*) echo h100 ;;
        *H200*) echo h200 ;;
        *L40S*) echo l40s ;;
        *V100*) echo v100 ;;
        *)      echo unknown ;;
    esac
}

find_available_port() {
    local port
    while true; do
        port=$((RANDOM % 64511 + 1024))
        if ! ss -tuln | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
}

append_submit_extra_train_args() {
    if ! declare -p TRAIN_EXTRA_ARGS >/dev/null 2>&1; then
        TRAIN_EXTRA_ARGS=()
    fi
    if [[ -n "${SUBMIT_EXTRA_ARGS_FILE:-}" ]]; then
        source "$SUBMIT_EXTRA_ARGS_FILE"
        TRAIN_EXTRA_ARGS+=("${SUBMIT_EXTRA_ARGS[@]}")
    fi
}

# Once training has fully completed (checkpoint-<max_steps> exists), strip
# resume-only trainer state from every checkpoint-step dir, leaving the same
# deployable "core" files the checkpoint-copy feature keeps: model shards +
# index, config.json, experiment_cfg/, processor/statistics/embodiment files.
# Incomplete runs are left untouched — --resume needs this state to continue.
cleanup_trainer_state() {
    local run_dir="$1"
    local max_steps="$2"
    if [ ! -d "$run_dir/checkpoint-$max_steps" ]; then
        log "Trainer-state cleanup skipped: $run_dir/checkpoint-$max_steps not found"
        return 0
    fi
    log "Training complete — removing resume-only trainer state under $run_dir"
    local step_dir
    for step_dir in "$run_dir"/checkpoint-*/; do
        [ -d "$step_dir" ] || continue
        rm -rf "$step_dir"global_step* \
               "$step_dir"optimizer* \
               "$step_dir"scheduler.pt \
               "$step_dir"rng_state_*.pth \
               "$step_dir"trainer_state.json \
               "$step_dir"latest \
               "$step_dir"zero_to_fp32.py || true
    done
    log "Trainer-state cleanup done."
}

pin_training_repo_dir() {
    local repo_src="$1"
    local commit="${2:-}"
    local namespace="${3:-job}"

    if [ -z "$commit" ]; then
        TRAIN_REPO_DIR="$repo_src"
        return 0
    fi

    local safe_namespace worktree_parent worktree current expected
    safe_namespace="$(printf '%s' "$namespace" | sed -E 's/[^A-Za-z0-9_.-]+/_/g' | sed -E 's/^_+|_+$//g')"
    safe_namespace="${safe_namespace:-job}"
    worktree_parent="$REPO_ROOT/.worktrees"
    worktree="$worktree_parent/$safe_namespace"

    mkdir -p "$worktree_parent"
    git -c safe.directory="$repo_src" -C "$repo_src" worktree prune || true

    if [ ! -e "$worktree/.git" ]; then
        if [ -e "$worktree" ]; then
            log "ERROR: refusing to use non-git worktree path: $worktree"
            exit 1
        fi
        for attempt in 1 2 3 4 5; do
            if git -c safe.directory="$repo_src" -C "$repo_src" worktree add --detach "$worktree" "$commit"; then
                break
            fi
            rc=$?
            if [ "$attempt" = "5" ]; then
                exit "$rc"
            fi
            sleep $((attempt * 2))
        done
    fi

    current="$(git -c safe.directory="$worktree" -C "$worktree" rev-parse HEAD)"
    expected="$(git -c safe.directory="$repo_src" -C "$repo_src" rev-parse "$commit^{commit}")"
    if [ "$current" != "$expected" ]; then
        log "ERROR: training repo worktree commit mismatch: expected $expected got $current"
        exit 1
    fi
    if [ -d "$repo_src/.venv" ] && [ ! -e "$worktree/.venv" ]; then
        ln -s "$repo_src/.venv" "$worktree/.venv"
    fi
    TRAIN_REPO_DIR="$worktree"
    log "Pinned training repo: $TRAIN_REPO_DIR ($current)"
}

finish_eval_launch_phase() {
    local launched="$1"
    local failed="$2"
    local results_path="$3"

    if [ "$failed" -ne 0 ]; then
        log "ERROR: one or more eval runs failed"
        exit 1
    fi
    if [ "$launched" -eq 0 ] && [ "${EVAL_OVERWRITE_RESULTS:-0}" != "1" ]; then
        log "No new eval runs launched; leaving aggregate unchanged: $results_path"
        exit 0
    fi
}

# ── Shared eval validation helpers (Isaac + DexJoCo harnesses) ────────────────
# Used by eval_body.sh (both n1.5 and n1.6) and eval_body_dexjoco.sh. The Isaac
# helpers read/modify caller-scope variables (N_EPISODES, EVAL_SETS, LAST_CKPT,
# …) because the body scripts source this file into the same shell.

# Validate that $2 is a positive integer; exit 1 with a "<name> must be a
# positive integer" message otherwise. $1 is the variable name for the message.
require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
        echo "ERROR: $name must be a positive integer, got '$value'"
        exit 1
    fi
}

# DexJoCo checkpoint validation: EVAL_CHECKPOINT must be set and the path must
# exist (file or dir — pi0.5/openpi checkpoints are not always directories).
# Sets LAST_CKPT. Isaac uses validate_eval_checkpoint (requires a directory).
require_eval_checkpoint_path() {
    if [ -z "${EVAL_CHECKPOINT:-}" ]; then
        echo "ERROR: EVAL_CHECKPOINT is required"; exit 1
    fi
    LAST_CKPT="$EVAL_CHECKPOINT"
    if [ ! -e "$LAST_CKPT" ]; then
        echo "ERROR: checkpoint path not found: $LAST_CKPT"; exit 1
    fi
}

# ── Shared eval helpers (Isaac harness: eval_body.sh, both n1.5 and n1.6) ──────
# These were duplicated verbatim between eval_body.sh and eval_body_n16.sh.
# They read/modify caller-scope variables (PIDS, PORTS, FAILED, EVAL_*, etc.)
# because eval_body.sh sources this file into the same shell.

# Validate EVAL_CHECKPOINT is set and points at a checkpoint dir; export LAST_CKPT.
validate_eval_checkpoint() {
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
}

# Apply submit-time overrides then validate N_EPISODES/N_RUNS/EVAL_SETS.
validate_eval_counts() {
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
}

# Background-worker PID pool shared by the eval launch loop. cleanup_all and the
# trap are installed by the caller after PIDS=() is declared.
cleanup_all() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}

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

# Aggregate per-run results.json into the experiment-level results.json.
# $1 = family (n1.5 | n1.6). The two families differ only in the stats engine
# (numpy vs statistics) and the family-specific top-level keys (n1.5 emits
# 'data_config'; n1.6 emits 'model_version' + 'modality_config'). All other
# interpolated values are read from caller scope (this file is sourced).
aggregate_eval_results() {
    local family="$1"
    # n1.5 historically invoked `python`; n1.6 invoked `python3`. Preserve each
    # family's interpreter so the emitted commands stay byte-equivalent.
    local PY="python3"
    [ "$family" = "n1.5" ] && PY="python"

    log "Aggregating results..."

    # Dump the TASKS list as JSON so the python block can iterate it cleanly
    # (avoids quoting/escaping instructions through a bash-built python literal).
    TASKS_JSON="$EVAL_DIR/.eval_tasks.json"
    "$PY" - "$TASKS_JSON" "${TASKS[@]}" <<'PYDUMP'
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

    if [ "$family" = "n1.5" ]; then
        "$PY" - <<PYEOF
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
    'output_namespace': '${OUTPUT_NAMESPACE}',
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
    else
        "$PY" - <<PYEOF
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
    fi
}
