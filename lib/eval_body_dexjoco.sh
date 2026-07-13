#!/usr/bin/env bash
# DexJoCo (MuJoCo benchmark) eval harness for train-eval-web.
#
# Unlike eval_body.sh (Isaac Sim server + gr00t client), DexJoCo eval is a
# policy server (holds the model) + a MuJoCo client (dexjoco-openpi-eval) talking
# the openpi-client websocket protocol. Two server backends are supported:
#   - groot : a GR00T N1.6 / PhysiXel finetune served via lib/dexjoco/gr00t_dexjoco_server.py
#             (run with the model repo's .venv python).
#   - openpi: the released pi0.5 baseline served via $DEXJOCO_DIR/openpi/scripts/serve_policy.py
#             (run in the 'openpi' micromamba env).
# The MuJoCo client runs in the 'dexjoco' micromamba env.
#
# DexJoCo writes no machine-readable summary (only episode_NN_{success,failure}/ dirs
# and a zero-byte success_rate_<pass>_<total>.txt), so this script synthesises the
# results.json that results.py / details.py expect.
#
# Reads $REPO_ROOT, $CLUSTER, $VARIANT from the environment (set by submit --export).
set -euo pipefail
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

TRAIN_REPO_DIR="${SUBMIT_TRAIN_REPO_DIR:-${TRAIN_REPO_DIR:-}}"

resolve_eval_output_paths

# ── Config + submit-time overrides ──────────────────────────────────────────
DEXJOCO_SERVER_TYPE="${DEXJOCO_SERVER_TYPE:-groot}"
DEXJOCO_TASK="${SUBMIT_DEXJOCO_TASK:-${DEXJOCO_TASK:-}}"
DEXJOCO_PAD_STATE_DIM46="${DEXJOCO_PAD_STATE_DIM46:-0}"
# Embodiment tag handed to the GR00T policy server. Multi-embodiment checkpoints
# set DEXJOCO_EMBODIMENT_TAG (single-arm tasks) and DEXJOCO_EMBODIMENT_TAG_BIMANUAL
# (bimanual_* tasks); legacy single-tag checkpoints keep the new_embodiment default.
# Base embodiment tag (single-arm / legacy). bimanual_* tasks use the bimanual
# tag. Both are resolved per task in the eval loop below so a multi-task run
# picks the right tag for each task.
DEXJOCO_EMBODIMENT_TAG_BASE="${DEXJOCO_EMBODIMENT_TAG:-new_embodiment}"
DEXJOCO_EMBODIMENT_TAG_BIMANUAL="${DEXJOCO_EMBODIMENT_TAG_BIMANUAL:-$DEXJOCO_EMBODIMENT_TAG_BASE}"
DEXJOCO_EMBODIMENT_TAG="$DEXJOCO_EMBODIMENT_TAG_BASE"
SERVER_PROMPT="${INSTRUCTION:-${DEXJOCO_PROMPT:-}}"
N_EPISODES="${SUBMIT_EVAL_N_EPISODES:-${N_EPISODES:-50}}"
N_RUNS="${SUBMIT_EVAL_N_RUNS:-${N_RUNS:-1}}"
EVAL_BASE_SEED="${EVAL_BASE_SEED:-0}"
EVAL_OVERWRITE_RESULTS="${SUBMIT_EVAL_OVERWRITE_RESULTS:-${EVAL_OVERWRITE_RESULTS:-0}}"
DEXJOCO_HEALTHZ_TIMEOUT_SECONDS="${DEXJOCO_HEALTHZ_TIMEOUT_SECONDS:-600}"
if [ -n "${SUBMIT_EVAL_SETS:-}" ]; then
    read -r -a EVAL_SETS <<< "$SUBMIT_EVAL_SETS"
fi
# EVAL_SETS holds the DexJoCo config families (rand_obj, rand_full, ...).
if [[ "${EVAL_SETS+set}" != set ]] || [ "${#EVAL_SETS[@]}" -eq 0 ]; then
    EVAL_SETS=(rand_obj)
fi

# Task mode:
#   (a) TASKS=("short|task_name|instruction" ...) — multi-task eval matrix; the
#       loop evaluates every task in one job (task_name is the DexJoCo config
#       stem configs/<family>/<task_name>.yaml, results land under <short>/).
#   (b) DEXJOCO_TASK + INSTRUCTION — legacy single-task.
# Synthesize a one-element list for single-task so the loop handles both.
if [[ "${TASKS+set}" == set ]] && [ "${#TASKS[@]}" -gt 0 ]; then
    MULTI_TASK=1
    apply_eval_task_selection
    log "Mode: multi-task over ${#TASKS[@]} tasks"
else
    MULTI_TASK=0
    TASKS=("__single__|${DEXJOCO_TASK}|${SERVER_PROMPT}")
fi

# ── Validation ──────────────────────────────────────────────────────────────
: "${DEXJOCO_DIR:?DEXJOCO_DIR not set in cluster env}"
: "${MICROMAMBA_BIN:?MICROMAMBA_BIN not set in cluster env}"
: "${MAMBA_ROOT_PREFIX:?MAMBA_ROOT_PREFIX not set in cluster env}"
: "${DEXJOCO_EVAL_ENV:?DEXJOCO_EVAL_ENV not set in cluster env}"
export MAMBA_ROOT_PREFIX
[ "$MULTI_TASK" = "1" ] || [ -n "$DEXJOCO_TASK" ] || { log "ERROR: DEXJOCO_TASK not set (config.sh or submit picker)"; exit 1; }
[ -d "$DEXJOCO_DIR" ] || { log "ERROR: DEXJOCO_DIR not found: $DEXJOCO_DIR"; exit 1; }
[ -x "$MICROMAMBA_BIN" ] || { log "ERROR: micromamba not executable: $MICROMAMBA_BIN"; exit 1; }
# Shared validators (lib/_common.sh): positive-int counts + checkpoint path.
require_positive_int "N_EPISODES" "$N_EPISODES"
require_positive_int "N_RUNS" "$N_RUNS"
require_eval_checkpoint_path

ADAPTER="$REPO_ROOT/lib/dexjoco/gr00t_dexjoco_server.py"
if [ "$DEXJOCO_SERVER_TYPE" = "groot" ]; then
    [ -n "$TRAIN_REPO_DIR" ] || { log "ERROR: SUBMIT_TRAIN_REPO_DIR not set for groot server"; exit 1; }
    SUBMIT_GIT_COMMIT="${SUBMIT_GIT_COMMIT:-${TRAIN_GIT_COMMIT:-}}"
    pin_training_repo_dir "$TRAIN_REPO_DIR" "$SUBMIT_GIT_COMMIT" "${SLURM_JOB_ID:-$OUTPUT_NAMESPACE}"
    [ -x "$TRAIN_REPO_DIR/.venv/bin/python" ] || { log "ERROR: model venv python not found: $TRAIN_REPO_DIR/.venv/bin/python"; exit 1; }
    [ -f "$ADAPTER" ] || { log "ERROR: adapter not found: $ADAPTER"; exit 1; }
elif [ "$DEXJOCO_SERVER_TYPE" = "openpi" ]; then
    : "${DEXJOCO_OPENPI_ENV:?DEXJOCO_OPENPI_ENV not set in cluster env}"
    [ -f "$DEXJOCO_DIR/openpi/scripts/serve_policy.py" ] || { log "ERROR: serve_policy.py not found under $DEXJOCO_DIR/openpi"; exit 1; }
else
    log "ERROR: DEXJOCO_SERVER_TYPE must be 'groot' or 'openpi', got '$DEXJOCO_SERVER_TYPE'"; exit 1
fi

log "========================================================"
log "$EXP_NAME - DexJoCo eval ($DEXJOCO_SERVER_TYPE)"
log "  cluster=$CLUSTER  partition=${SUBMIT_PARTITION:-$PARTITION}  gpu=$GPU_INSTANCE"
log "  task=$DEXJOCO_TASK  families(eval_sets)=${EVAL_SETS[*]}"
log "  episodes=$N_EPISODES  runs=$N_RUNS  base_seed=$EVAL_BASE_SEED"
if [ "$DEXJOCO_SERVER_TYPE" = "groot" ]; then
    log "  train repo=$TRAIN_REPO_DIR"
fi
log "  checkpoint=$LAST_CKPT"
log "  eval results=$EVAL_DIR"
log "========================================================"

[ "$EVAL_OVERWRITE_RESULTS" = "1" ] && rm -f "$RESULTS_PATH"

FAILED=0
EVAL_LAUNCHED=0
SERVER_PID=""
PORT=""
trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT TERM

# Derive the pi0.5 serve_policy --policy.config name from task + family.
openpi_policy_config() {
    local family="$1"
    if [ -n "${DEXJOCO_OPENPI_POLICY_CONFIG:-}" ]; then
        echo "$DEXJOCO_OPENPI_POLICY_CONFIG"; return 0
    fi
    case "$family" in
        rand_full)  echo "${DEXJOCO_TASK}_rand_full" ;;
        multi_task) echo "multi_task" ;;
        *)          echo "$DEXJOCO_TASK" ;;
    esac
}

start_server() {
    local family="$1"
    local port="$2"
    local server_log="$3"
    if [ "$DEXJOCO_SERVER_TYPE" = "groot" ]; then
        local img_args=()
        [ -n "${DEXJOCO_IMAGE_SIZE:-}" ] && img_args=(--image_size "$DEXJOCO_IMAGE_SIZE")
        ( cd "$REPO_ROOT/lib/dexjoco" \
            && PYTHONPATH="$TRAIN_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
               CUDA_VISIBLE_DEVICES=0 "$TRAIN_REPO_DIR/.venv/bin/python" gr00t_dexjoco_server.py \
                --model_path "$LAST_CKPT" --port "$port" --prompt "$SERVER_PROMPT" \
                --embodiment_tag "$DEXJOCO_EMBODIMENT_TAG" "${img_args[@]}" ) \
            > "$server_log" 2>&1 &
        SERVER_PID=$!
    else
        local pcfg; pcfg="$(openpi_policy_config "$family")"
        log "  openpi policy.config=$pcfg"
        ( cd "$DEXJOCO_DIR/openpi" \
            && XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 CUDA_VISIBLE_DEVICES=0 \
               "$MICROMAMBA_BIN" run -n "$DEXJOCO_OPENPI_ENV" python ./scripts/serve_policy.py \
                --port="$port" policy:checkpoint --policy.config="$pcfg" --policy.dir="$LAST_CKPT" ) \
            > "$server_log" 2>&1 &
        SERVER_PID=$!
    fi
}

cleanup_server() {
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
    [ -n "${SERVER_PID:-}" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
    if [ -n "${PORT:-}" ]; then
        pkill -9 -f "gr00t_dexjoco_server.py.*--port $PORT" 2>/dev/null || true
        pkill -9 -f "serve_policy.py.*--port=$PORT" 2>/dev/null || true
    fi
    SERVER_PID=""
}

wait_for_server() {
    local port="$1"
    local server_log="$2"
    local elapsed=0
    while [ "$elapsed" -le "$DEXJOCO_HEALTHZ_TIMEOUT_SECONDS" ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: policy server exited before ready (port=$port)"
            tail -n 60 "$server_log" 2>/dev/null | sed 's/^/[server] /' | tee -a "$LOG_FILE" || true
            return 1
        fi
        if curl -sf "http://localhost:$port/healthz" >/dev/null 2>&1; then
            log "  server healthz OK on port $port"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    log "  (healthz not 200 after ${DEXJOCO_HEALTHZ_TIMEOUT_SECONDS}s; proceeding - client will retry)"
    return 0
}

# Parse DexJoCo's output dir into a results.json. Counts episode_*_success vs
# episode_*_failure dirs; cross-checks the success_rate_<pass>_<total>.txt name.
write_results_json() {
    local out_dir="$1"
    local results_json="$2"
    local family="$3"
    local seed="$4"
    python3 - "$out_dir" "$results_json" "$LAST_CKPT" "$DEXJOCO_TASK" "$family" "$seed" "$DEXJOCO_SERVER_TYPE" "$N_EPISODES" <<'PY'
import json, re, sys
from pathlib import Path

out_dir, results_json, ckpt, task, family, seed, server_type, episodes = sys.argv[1:9]
out = Path(out_dir)
eps = sorted(
    [p for p in out.glob("episode_*") if p.is_dir()],
    key=lambda p: int(re.match(r"episode_(\d+)", p.name).group(1)) if re.match(r"episode_(\d+)", p.name) else 0,
)
# DexJoCo names each episode dir episode_<NN>_<status>_<details>, e.g.
# episode_03_success_1_2_3 or episode_01_failure_no_password_input. The status
# is the token right after the index, NOT a suffix — multi-criterion (bimanual)
# tasks append per-goal details — so match it positionally.
def _is_success(name):
    m = re.match(r"episode_\d+_(success|failure)(?:_|$)", name)
    return m is not None and m.group(1) == "success"

success = [_is_success(p.name) for p in eps]
success_count = sum(success)
total = len(eps)

# The eval harness also drops a zero-byte success_rate_<pass>_<total>.txt marker;
# treat it as authoritative for the summary when present.
marker = next(iter(out.glob("success_rate_*_*.txt")), None)
if marker is not None:
    m = re.match(r"success_rate_(\d+)_(\d+)\.txt$", marker.name)
    if m:
        mp, mt = int(m.group(1)), int(m.group(2))
        if (mp, mt) != (success_count, total):
            print(f"WARNING: marker {marker.name} disagrees with dir count {success_count}/{total}; using marker", file=sys.stderr)
        success_count, total = mp, mt

if total == 0:
    raise SystemExit(f"ERROR: no episode_* dirs and no success marker in {out_dir}")

rate = success_count / total
data = {
    "summary": {
        "success_rate": rate,
        "success_count": success_count,
        "total_episodes": total,
        "episode_count": total,
    },
    "success": success,
    "config": {
        "checkpoint": ckpt,
        "task": task,
        "eval_set": family,
        "seed": int(seed),
        "server_type": server_type,
        "episodes": int(episodes),
    },
}
Path(results_json).parent.mkdir(parents=True, exist_ok=True)
with open(results_json, "w") as f:
    json.dump(data, f, indent=2)
print(f"{success_count}/{total} ({rate*100:.1f}%)")
PY
}

run_eval_one() (
    set -euo pipefail
    local FAMILY="$1"
    local RUN_IDX="$2"
    local RUN_SEED="$3"
    local PORT
    local RUN_DIR="${CUR_EVAL_DIR:-$EVAL_DIR}/$FAMILY/run_$RUN_IDX"
    local RUN_RESULTS="$RUN_DIR/results.json"
    local OUT_DIR="$RUN_DIR/dexjoco_out"
    local SERVER_LOG="$JOB_LOG_DIR/server_${CUR_TASK_TAG:+${CUR_TASK_TAG}_}${FAMILY}_run${RUN_IDX}.log"
    local SERVER_PID=""

    trap cleanup_server EXIT
    trap 'cleanup_server; exit 130' INT TERM

    if [ "$EVAL_OVERWRITE_RESULTS" = "1" ] && [ -e "$RUN_DIR" ]; then
        log "  OVERWRITE: removing $RUN_DIR"
        rm -rf -- "$RUN_DIR"
    fi
    if [ -f "$RUN_RESULTS" ]; then
        log "SKIP (results.json already exists): $RUN_DIR"
        exit 0
    fi

    mkdir -p "$RUN_DIR"
    rm -rf -- "$OUT_DIR"
    PORT="$(find_available_port)"

    log ""
    log "  family=$FAMILY run=$RUN_IDX/$N_RUNS seed=$RUN_SEED port=$PORT"
    log "  starting $DEXJOCO_SERVER_TYPE policy server (log: $SERVER_LOG)"
    start_server "$FAMILY" "$PORT" "$SERVER_LOG"
    sleep 2
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: policy server died within 2s of launch (family=$FAMILY run=$RUN_IDX)"
        tail -n 40 "$SERVER_LOG" 2>/dev/null | sed 's/^/[server] /' | tee -a "$LOG_FILE" || true
        return 1
    fi
    if ! wait_for_server "$PORT" "$SERVER_LOG"; then
        return 1
    fi

    log "  running dexjoco-openpi-eval"
    PAD_ARGS=()
    [ "$DEXJOCO_PAD_STATE_DIM46" = "1" ] && PAD_ARGS=(--pad-state-dim46)
    REPLAN_ARGS=()
    [ -n "${DEXJOCO_REPLAN_RATIO:-}" ] && REPLAN_ARGS=(--replan-ratio "$DEXJOCO_REPLAN_RATIO")
    CLIENT_RC=0
    ( cd "$DEXJOCO_DIR" \
        && MUJOCO_GL=egl "$MICROMAMBA_BIN" run -n "$DEXJOCO_EVAL_ENV" dexjoco-openpi-eval \
            --config="./configs/$FAMILY/$DEXJOCO_TASK.yaml" \
            --seed="$RUN_SEED" --port="$PORT" --episodes="$N_EPISODES" \
            --output="$OUT_DIR" "${PAD_ARGS[@]}" "${REPLAN_ARGS[@]}" ) \
        >> "$LOG_FILE" 2>&1 || CLIENT_RC=$?

    cleanup_server

    if [ "$CLIENT_RC" -ne 0 ]; then
        log "ERROR: dexjoco-openpi-eval exited with status $CLIENT_RC (family=$FAMILY run=$RUN_IDX)"
        tail -n 40 "$SERVER_LOG" 2>/dev/null | sed 's/^/[server] /' | tee -a "$LOG_FILE" || true
        return 1
    fi

    if ! SUMMARY="$(write_results_json "$OUT_DIR" "$RUN_RESULTS" "$FAMILY" "$RUN_SEED")"; then
        log "ERROR: failed to synthesise results.json for $RUN_DIR"
        return 1
    fi
    log "  result: $SUMMARY"
    echo "Results saved to: $RUN_RESULTS" | tee -a "$LOG_FILE"
    trap - EXIT
    return 0
)

# ── Eval matrix: tasks x families (eval_sets) x seeds (runs) ────────────────
# run_eval_one reads DEXJOCO_TASK / SERVER_PROMPT / DEXJOCO_EMBODIMENT_TAG /
# CUR_EVAL_DIR / CUR_TASK_TAG as globals and evaluates each unit serially.
for task_entry in "${TASKS[@]}"; do
    IFS='|' read -r TASK_SHORT TASK_NAME_LOOP TASK_INSTR_LOOP <<<"$task_entry"
    DEXJOCO_TASK="$TASK_NAME_LOOP"
    SERVER_PROMPT="$TASK_INSTR_LOOP"
    DEXJOCO_EMBODIMENT_TAG="$DEXJOCO_EMBODIMENT_TAG_BASE"
    case "$DEXJOCO_TASK" in
        bimanual_*) DEXJOCO_EMBODIMENT_TAG="$DEXJOCO_EMBODIMENT_TAG_BIMANUAL" ;;
    esac
    if [ "$MULTI_TASK" -eq 1 ]; then
        CUR_EVAL_DIR="$EVAL_DIR/$TASK_SHORT"
        CUR_TASK_TAG="$TASK_SHORT"
        log ""
        log "========== Task: $TASK_SHORT ($DEXJOCO_TASK)  tag=$DEXJOCO_EMBODIMENT_TAG =========="
    else
        CUR_EVAL_DIR="$EVAL_DIR"
        CUR_TASK_TAG=""
    fi
    mkdir -p "$CUR_EVAL_DIR"

    for FAMILY in "${EVAL_SETS[@]}"; do
        CONFIG_YAML="$DEXJOCO_DIR/configs/$FAMILY/$DEXJOCO_TASK.yaml"
        if [ ! -f "$CONFIG_YAML" ]; then
            log "ERROR: dexjoco config not found: $CONFIG_YAML"
            FAILED=1
            continue
        fi
        for i in $(seq 1 "$N_RUNS"); do
            RUN_SEED=$((EVAL_BASE_SEED + (i - 1)))
            RUN_DIR="$CUR_EVAL_DIR/$FAMILY/run_$i"
            RUN_RESULTS="$RUN_DIR/results.json"
            if [ "$EVAL_OVERWRITE_RESULTS" != "1" ] && [ -f "$RUN_RESULTS" ]; then
                log ""
                log "  task=${CUR_TASK_TAG:-$DEXJOCO_TASK} family=$FAMILY run=$i/$N_RUNS seed=$RUN_SEED"
                log "  SKIP (results.json already exists): $RUN_DIR"
                continue
            fi

            EVAL_LAUNCHED=$((EVAL_LAUNCHED + 1))
            if ! run_eval_one "$FAMILY" "$i" "$RUN_SEED"; then
                FAILED=1
            fi
        done
    done
done

trap - EXIT
finish_eval_launch_phase "$EVAL_LAUNCHED" "$FAILED" "$RESULTS_PATH"

# ── Aggregate ───────────────────────────────────────────────────────────────
# Dynamic values are passed as argv into a QUOTED heredoc so free text
# (TRAIN_NOTE, paths, names) cannot break Python parsing. EVAL_SETS is variadic
# at the tail.
log "Aggregating results..."
# Dump TASKS as JSON so the aggregator iterates it without quoting free-text
# instructions through the heredoc (mirrors _common.sh's aggregate_eval_results).
TASKS_JSON="$EVAL_DIR/.eval_tasks.json"
python3 - "$TASKS_JSON" "${TASKS[@]}" <<'PYDUMP'
import json, sys
out_path = sys.argv[1]
tasks = []
for entry in sys.argv[2:]:
    parts = entry.split('|', 2)
    tasks.append({
        'short': parts[0],
        'task_name': parts[1] if len(parts) > 1 else parts[0],
        'instruction': parts[2] if len(parts) > 2 else '',
    })
with open(out_path, 'w') as f:
    json.dump(tasks, f)
PYDUMP

python3 - \
    "$EVAL_DIR" "$RESULTS_PATH" "$N_RUNS" "$N_EPISODES" "$EVAL_BASE_SEED" \
    "$EXP_NAME" "$OUTPUT_NAMESPACE" "$CLUSTER" "$GPU_INSTANCE" \
    "$LAST_CKPT" "$DEXJOCO_TASK" "$DEXJOCO_SERVER_TYPE" "${TRAIN_NOTE:-}" \
    "1" "$MULTI_TASK" "$TASKS_JSON" \
    "${EVAL_SETS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path

(eval_dir, results_path, n_runs, n_episodes, base_seed,
 exp_name, output_namespace, cluster, gpu,
 checkpoint, task_name, server_type, note,
 server_workers, multi_task, tasks_json) = sys.argv[1:17]
eval_sets = sys.argv[17:]
n_runs = int(n_runs)
server_workers = int(server_workers)
multi_task = multi_task == '1'
base = Path(eval_dir)
with open(tasks_json) as f:
    tasks = json.load(f)

def aggregate(family_dir):
    rates, counts, totals = [], [], []
    for i in range(1, n_runs + 1):
        p = family_dir / f'run_{i}' / 'results.json'
        if not p.exists():
            print(f'WARNING: {p} not found')
            continue
        s = json.load(open(p))['summary']
        rates.append(float(s['success_rate']))
        counts.append(int(s['success_count']))
        totals.append(int(s.get('total_episodes') or s.get('episode_count')))
    if not rates:
        return None
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    return {
        'per_run_success_rate': rates,
        'success_counts': counts,
        'episode_counts': totals,
        'mean_success_rate': mean,
        'std_success_rate': var ** 0.5,
    }

def aggregate_sets(task_base):
    out = {}
    for es in eval_sets:
        res = aggregate(task_base / es)
        if res is not None:
            out[es] = res
            print(f"  {es}: {res['mean_success_rate']:.4f} +/- {res['std_success_rate']:.4f}  {res['per_run_success_rate']}")
    return out

agg = {
    'experiment': exp_name,
    'output_namespace': output_namespace,
    'cluster': cluster,
    'gpu': gpu,
    'note': note,
    'checkpoint': checkpoint,
    'server_type': server_type,
    'n_episodes': int(n_episodes),
    'n_runs': n_runs,
    'server_workers': server_workers,
    'num_envs_per_gpu': 1,
    'total_num_envs': server_workers,
    'eval_base_seed': int(base_seed),
}
if multi_task:
    tasks_out = {}
    for t in tasks:
        ts = t['short']
        print(f'=== {ts} ({t["task_name"]}) ===')
        tasks_out[ts] = {
            'task_name': t['task_name'],
            'instruction': t['instruction'],
            'eval_sets': aggregate_sets(base / ts),
        }
    agg['tasks'] = tasks_out
else:
    agg['task_name'] = task_name
    agg['eval_sets'] = aggregate_sets(base)

out = Path(results_path)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(agg, f, indent=2)
print(f'Saved to {out}')
PYEOF

emit_done_marker "$RESULTS_PATH"
