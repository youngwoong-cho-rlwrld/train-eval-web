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
TRAIN_NUM_GPUS="${SUBMIT_TRAIN_NUM_GPUS:-${TRAIN_NUM_GPUS:-1}}"

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
# One persistent policy server is kept per GPU worker. The watchdog applies to
# individual client units; a failed unit is retried once with a fresh server.
DEXJOCO_NO_PROGRESS_TIMEOUT_SECONDS="${DEXJOCO_NO_PROGRESS_TIMEOUT_SECONDS:-2400}"
DEXJOCO_WATCHDOG_POLL_SECONDS="${DEXJOCO_WATCHDOG_POLL_SECONDS:-30}"
DEXJOCO_MUJOCO_GL="${DEXJOCO_MUJOCO_GL:-egl}"
DEXJOCO_WORKER_START_STAGGER_SECONDS="${DEXJOCO_WORKER_START_STAGGER_SECONDS:-2}"
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
require_positive_int "DEXJOCO_NO_PROGRESS_TIMEOUT_SECONDS" "$DEXJOCO_NO_PROGRESS_TIMEOUT_SECONDS"
require_positive_int "DEXJOCO_WATCHDOG_POLL_SECONDS" "$DEXJOCO_WATCHDOG_POLL_SECONDS"
if ! [[ "$DEXJOCO_WORKER_START_STAGGER_SECONDS" =~ ^[0-9]+$ ]]; then
    log "ERROR: DEXJOCO_WORKER_START_STAGGER_SECONDS must be a non-negative integer"
    exit 1
fi
require_eval_checkpoint_path

GROOT_ADAPTER="$REPO_ROOT/lib/dexjoco/gr00t_dexjoco_server.py"
GAM_ADAPTER="$REPO_ROOT/lib/dexjoco/gam_dexjoco_server.py"
# groot and gam are both served from the training repo's .venv python against a
# staged adapter in lib/dexjoco; only the adapter file and CLI shape differ.
if [ "$DEXJOCO_SERVER_TYPE" = "groot" ] || [ "$DEXJOCO_SERVER_TYPE" = "gam" ]; then
    [ -n "$TRAIN_REPO_DIR" ] || { log "ERROR: SUBMIT_TRAIN_REPO_DIR not set for $DEXJOCO_SERVER_TYPE server"; exit 1; }
    SUBMIT_GIT_COMMIT="${SUBMIT_GIT_COMMIT:-${TRAIN_GIT_COMMIT:-}}"
    # Capture the main checkout before pin_training_repo_dir swaps TRAIN_REPO_DIR
    # for a per-job worktree. The GAM server builds the DA3 backbone from
    # checkpoints/track4world_da3.pth, an untracked asset present only in the
    # main repo — the gam start_server branch points DA3_ROOT here.
    GAM_MAIN_REPO_DIR="$TRAIN_REPO_DIR"
    pin_training_repo_dir "$TRAIN_REPO_DIR" "$SUBMIT_GIT_COMMIT" "${SLURM_JOB_ID:-$OUTPUT_NAMESPACE}"
    [ -x "$TRAIN_REPO_DIR/.venv/bin/python" ] || { log "ERROR: model venv python not found: $TRAIN_REPO_DIR/.venv/bin/python"; exit 1; }
    if [ "$DEXJOCO_SERVER_TYPE" = "gam" ]; then
        [ -f "$GAM_ADAPTER" ] || { log "ERROR: adapter not found: $GAM_ADAPTER"; exit 1; }
    else
        [ -f "$GROOT_ADAPTER" ] || { log "ERROR: adapter not found: $GROOT_ADAPTER"; exit 1; }
    fi
elif [ "$DEXJOCO_SERVER_TYPE" = "openpi" ]; then
    : "${DEXJOCO_OPENPI_ENV:?DEXJOCO_OPENPI_ENV not set in cluster env}"
    [ -f "$DEXJOCO_DIR/openpi/scripts/serve_policy.py" ] || { log "ERROR: serve_policy.py not found under $DEXJOCO_DIR/openpi"; exit 1; }
else
    log "ERROR: DEXJOCO_SERVER_TYPE must be 'groot', 'gam' or 'openpi', got '$DEXJOCO_SERVER_TYPE'"; exit 1
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

EVAL_GPU_COUNT="${EVAL_NUM_GPUS:-$TRAIN_NUM_GPUS}"
if ! [[ "$EVAL_GPU_COUNT" =~ ^[0-9]+$ ]] || [ "$EVAL_GPU_COUNT" -lt 1 ]; then
    log "ERROR: EVAL_NUM_GPUS must be a positive integer, got '$EVAL_GPU_COUNT'"
    exit 1
fi

FAILED=0
EVAL_LAUNCHED=0
WORKER_PIDS=()

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
    local cuda_device="$4"
    if [ "$DEXJOCO_SERVER_TYPE" = "groot" ]; then
        local img_args=()
        [ -n "${DEXJOCO_IMAGE_SIZE:-}" ] && img_args=(--image_size "$DEXJOCO_IMAGE_SIZE")
        ( cd "$REPO_ROOT/lib/dexjoco" \
            && PYTHONPATH="$TRAIN_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
               CUDA_VISIBLE_DEVICES="$cuda_device" "$TRAIN_REPO_DIR/.venv/bin/python" gr00t_dexjoco_server.py \
                --model_path "$LAST_CKPT" --port "$port" --prompt "$SERVER_PROMPT" \
                --embodiment_tag "$DEXJOCO_EMBODIMENT_TAG" "${img_args[@]}" ) \
            > "$server_log" 2>&1 &
        SERVER_PID=$!
    elif [ "$DEXJOCO_SERVER_TYPE" = "gam" ]; then
        # GAM server: resolves checkpoint-final.pt + config.yaml + action_stats/
        # from the checkpoint dir and maps the embodiment tag to dims itself.
        # PYTHONPATH is the fork's src/; the adapter is staged in lib/dexjoco.
        # DA3_ROOT points at the main checkout (not the worktree) so the server
        # finds the untracked DA3 backbone ckpt checkpoints/track4world_da3.pth.
        ( cd "$REPO_ROOT/lib/dexjoco" \
            && PYTHONPATH="$TRAIN_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
               DA3_ROOT="${GAM_MAIN_REPO_DIR:-$TRAIN_REPO_DIR}" \
               CUDA_VISIBLE_DEVICES="$cuda_device" "$TRAIN_REPO_DIR/.venv/bin/python" gam_dexjoco_server.py \
                --checkpoint-path "$LAST_CKPT" --port "$port" \
                --embodiment-tag "$DEXJOCO_EMBODIMENT_TAG" --host 127.0.0.1 ) \
            > "$server_log" 2>&1 &
        SERVER_PID=$!
    else
        local pcfg; pcfg="$(openpi_policy_config "$family")"
        log "  openpi policy.config=$pcfg"
        ( cd "$DEXJOCO_DIR/openpi" \
            && XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 CUDA_VISIBLE_DEVICES="$cuda_device" \
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

cleanup_client() {
    local client_pid="${CLIENT_PID:-}"
    [ -n "$client_pid" ] && kill -9 -- -"$client_pid" 2>/dev/null || true
    [ -n "$client_pid" ] && kill -9 "$client_pid" 2>/dev/null || true
    if [ -n "${PORT:-}" ]; then
        pkill -9 -f "dexjoco-openpi-eval.*--port=$PORT" 2>/dev/null || true
    fi
    CLIENT_PID=""
}

cleanup_workers() {
    local pid
    for pid in "${WORKER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}

trap cleanup_workers EXIT
trap 'cleanup_workers; exit 130' INT TERM

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

# Count the completed-episode prefix of a dexjoco output dir, deleting any
# in-flight episode_*_temp leftovers first. Episodes run strictly in order, so
# the count is also the index the next invocation should start from.
count_completed_episodes() {
    local out_dir="$1"
    [ -d "$out_dir" ] || { echo 0; return 0; }
    find "$out_dir" -maxdepth 1 -type d -name 'episode_*_temp' -exec rm -rf -- {} + 2>/dev/null || true
    find "$out_dir" -maxdepth 1 -type d 2>/dev/null \
        | grep -cE '/episode_[0-9]+_(success|failure)(_|$)' || true
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

run_client_once() {
    local family="$1"
    local run_seed="$2"
    local out_dir="$3"
    local cuda_device="$4"
    local start_episode="${5:-0}"
    local client_pid client_rc now last_progress signature previous_signature
    local stalled=0
    local -a pad_args=() replan_args=() resume_args=()
    [ "$DEXJOCO_PAD_STATE_DIM46" = "1" ] && pad_args=(--pad-state-dim46)
    [ -n "${DEXJOCO_REPLAN_RATIO:-}" ] && replan_args=(--replan-ratio "$DEXJOCO_REPLAN_RATIO")
    [ "$start_episode" -gt 0 ] && resume_args=(--start-episode="$start_episode")

    ( cd "$DEXJOCO_DIR" \
        && exec env CUDA_VISIBLE_DEVICES="$cuda_device" \
           MUJOCO_GL="$DEXJOCO_MUJOCO_GL" MUJOCO_EGL_DEVICE_ID="$cuda_device" \
           PYTHONUNBUFFERED=1 setsid "$MICROMAMBA_BIN" run -n "$DEXJOCO_EVAL_ENV" \
            dexjoco-openpi-eval \
            --config="./configs/$family/$DEXJOCO_TASK.yaml" \
            --seed="$run_seed" --port="$PORT" --episodes="$N_EPISODES" \
            --output="$out_dir" "${pad_args[@]}" "${replan_args[@]}" "${resume_args[@]}" ) \
        >> "$LOG_FILE" 2>&1 &
    client_pid=$!
    CLIENT_PID="$client_pid"
    last_progress="$(date +%s)"
    previous_signature=""

    while kill -0 "$client_pid" 2>/dev/null; do
        sleep "$DEXJOCO_WATCHDOG_POLL_SECONDS"
        now="$(date +%s)"
        signature="$(find "$out_dir" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1);$(find "$out_dir" 2>/dev/null | wc -l)"
        if [ "$signature" != "$previous_signature" ]; then
            previous_signature="$signature"
            last_progress="$now"
        elif [ $((now - last_progress)) -ge "$DEXJOCO_NO_PROGRESS_TIMEOUT_SECONDS" ]; then
            stalled=1
            log "WATCHDOG: no output progress for $((now - last_progress))s; killing client pid=$client_pid"
            cleanup_client
            break
        fi
    done

    if wait "$client_pid" 2>/dev/null; then client_rc=0; else client_rc=$?; fi
    CLIENT_PID=""
    [ "$stalled" -eq 0 ] || return 124
    return "$client_rc"
}

ensure_worker_server() {
    local server_key="$1"
    local family="$2"
    local cuda_device="$3"
    local gpu_slot="$4"
    local safe_key

    if [ "$ACTIVE_SERVER_KEY" = "$server_key" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        return 0
    fi

    cleanup_server
    ACTIVE_SERVER_KEY=""
    [ -n "$PORT" ] || PORT="$(find_available_port)"
    safe_key="$(printf '%s' "$server_key" | sed -E 's/[^A-Za-z0-9_.-]+/_/g')"
    SERVER_LOG="$JOB_LOG_DIR/server_gpu${gpu_slot}_${safe_key}.log"
    log "  gpu_slot=$gpu_slot cuda=$cuda_device starting persistent $DEXJOCO_SERVER_TYPE server key=$server_key port=$PORT"
    start_server "$family" "$PORT" "$SERVER_LOG" "$cuda_device"
    sleep 2
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "ERROR: policy server died during startup (gpu_slot=$gpu_slot key=$server_key)"
        tail -n 40 "$SERVER_LOG" 2>/dev/null | sed 's/^/[server] /' | tee -a "$LOG_FILE" || true
        return 1
    fi
    if ! wait_for_server "$PORT" "$SERVER_LOG"; then
        return 1
    fi
    ACTIVE_SERVER_KEY="$server_key"
}

run_unit_on_worker() {
    local unit_idx="$1"
    local cuda_device="$2"
    local gpu_slot="$3"
    local attempt client_rc summary completed_eps effective_seed

    DEXJOCO_TASK="${UNIT_TASK_NAME[$unit_idx]}"
    SERVER_PROMPT="${UNIT_INSTRUCTION[$unit_idx]}"
    DEXJOCO_EMBODIMENT_TAG="${UNIT_EMBODIMENT_TAG[$unit_idx]}"
    FAMILY="${UNIT_FAMILY[$unit_idx]}"
    RUN_IDX="${UNIT_RUN_IDX[$unit_idx]}"
    RUN_SEED="${UNIT_RUN_SEED[$unit_idx]}"
    CUR_EVAL_DIR="${UNIT_EVAL_DIR[$unit_idx]}"
    CUR_TASK_TAG="${UNIT_TASK_SHORT[$unit_idx]}"
    SERVER_KEY="${UNIT_SERVER_KEY[$unit_idx]}"
    RUN_DIR="$CUR_EVAL_DIR/$FAMILY/run_$RUN_IDX"
    RUN_RESULTS="$RUN_DIR/results.json"
    OUT_DIR="$RUN_DIR/dexjoco_out"

    if [ "$EVAL_OVERWRITE_RESULTS" = "1" ] && [ -e "$RUN_DIR" ]; then
        log "  OVERWRITE: removing $RUN_DIR"
        rm -rf -- "$RUN_DIR"
    elif [ -f "$RUN_RESULTS" ]; then
        log "  SKIP (results.json already exists): $RUN_DIR"
        return 0
    fi
    mkdir -p "$RUN_DIR"

    for attempt in 1 2; do
        # Completed episodes survive timeouts, crashes, and the retry below;
        # only the in-flight episode is ever redone.
        completed_eps="$(count_completed_episodes "$OUT_DIR")"
        if [ "$completed_eps" -ge "$N_EPISODES" ]; then
            log "  RESUME: all $N_EPISODES episodes already present; synthesising results"
            client_rc=0
            break
        fi
        effective_seed="$RUN_SEED"
        if [ "$completed_eps" -gt 0 ]; then
            # The client seeds its RNG once at startup, so restarting a segment
            # with the run seed would replay episode 0's initial conditions
            # into the remaining slots. Shift far past the base-seed band.
            effective_seed=$((RUN_SEED + 1000000 + completed_eps))
            log "  RESUME: $completed_eps/$N_EPISODES episodes done; continuing from episode $((completed_eps + 1)) (seed=$effective_seed)"
        fi
        if ! ensure_worker_server "$SERVER_KEY" "$FAMILY" "$cuda_device" "$gpu_slot"; then
            client_rc=1
        else
            log "  gpu_slot=$gpu_slot task=${CUR_TASK_TAG:-$DEXJOCO_TASK} family=$FAMILY run=$RUN_IDX/$N_RUNS seed=$effective_seed attempt=$attempt"
            if run_client_once "$FAMILY" "$effective_seed" "$OUT_DIR" "$cuda_device" "$completed_eps"; then
                client_rc=0
            else
                client_rc=$?
            fi
        fi

        if [ "$client_rc" -eq 0 ]; then
            break
        fi
        cleanup_server
        ACTIVE_SERVER_KEY=""
        if [ "$attempt" -eq 1 ]; then
            log "  retrying failed unit with a fresh server (gpu_slot=$gpu_slot rc=$client_rc)"
            continue
        fi
        log "ERROR: eval unit failed twice (task=${CUR_TASK_TAG:-$DEXJOCO_TASK} family=$FAMILY run=$RUN_IDX rc=$client_rc)"
        return 1
    done

    if ! summary="$(write_results_json "$OUT_DIR" "$RUN_RESULTS" "$FAMILY" "$RUN_SEED")"; then
        log "ERROR: failed to synthesise results.json for $RUN_DIR"
        return 1
    fi
    log "  result: $summary"
    echo "Results saved to: $RUN_RESULTS" | tee -a "$LOG_FILE"
}

run_gpu_worker() (
    set -uo pipefail
    local gpu_slot="$1"
    local queue_file="$2"
    local cuda_device worker_failed=0 unit_idx
    local SERVER_PID="" CLIENT_PID="" PORT="" ACTIVE_SERVER_KEY="" SERVER_LOG=""
    cuda_device="$(select_cuda_device "$gpu_slot")"
    trap 'cleanup_client; cleanup_server' EXIT
    trap 'cleanup_client; cleanup_server; exit 130' INT TERM

    if [ "$gpu_slot" -gt 0 ] && [ "$DEXJOCO_WORKER_START_STAGGER_SECONDS" -gt 0 ]; then
        sleep $((gpu_slot * DEXJOCO_WORKER_START_STAGGER_SECONDS))
    fi
    log "GPU worker $gpu_slot started on CUDA device $cuda_device"
    while IFS= read -r unit_idx; do
        [ -n "$unit_idx" ] || continue
        if ! run_unit_on_worker "$unit_idx" "$cuda_device" "$gpu_slot"; then
            worker_failed=1
        fi
    done < "$queue_file"
    cleanup_client
    cleanup_server
    trap - EXIT
    return "$worker_failed"
)

# ── Build and assign the task x family x run work matrix ────────────────────
UNIT_TASK_SHORT=()
UNIT_TASK_NAME=()
UNIT_INSTRUCTION=()
UNIT_EMBODIMENT_TAG=()
UNIT_FAMILY=()
UNIT_RUN_IDX=()
UNIT_RUN_SEED=()
UNIT_EVAL_DIR=()
UNIT_SERVER_KEY=()
GROUP_KEYS=()
GROUP_UNITS=()
declare -A GROUP_ID_BY_KEY=()

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
        if [ "$DEXJOCO_SERVER_TYPE" = "openpi" ]; then
            SERVER_KEY="$DEXJOCO_SERVER_TYPE:$TASK_SHORT:$(openpi_policy_config "$FAMILY")"
        else
            # groot and gam both reuse one persistent server per embodiment tag
            # (single-arm vs dual-arm), so the tag is the grouping key.
            SERVER_KEY="$DEXJOCO_SERVER_TYPE:$TASK_SHORT:$DEXJOCO_EMBODIMENT_TAG"
        fi
        if [[ "${GROUP_ID_BY_KEY[$SERVER_KEY]+present}" != present ]]; then
            GROUP_ID_BY_KEY[$SERVER_KEY]="${#GROUP_KEYS[@]}"
            GROUP_KEYS+=("$SERVER_KEY")
            GROUP_UNITS+=("")
        fi
        GROUP_ID="${GROUP_ID_BY_KEY[$SERVER_KEY]}"

        for i in $(seq 1 "$N_RUNS"); do
            RUN_SEED=$((EVAL_BASE_SEED + (i - 1)))
            RUN_RESULTS="$CUR_EVAL_DIR/$FAMILY/run_$i/results.json"
            if [ "$EVAL_OVERWRITE_RESULTS" != "1" ] && [ -f "$RUN_RESULTS" ]; then
                log "  SKIP (results.json already exists): ${RUN_RESULTS%/results.json}"
                continue
            fi
            UNIT_IDX="${#UNIT_FAMILY[@]}"
            UNIT_TASK_SHORT+=("$CUR_TASK_TAG")
            UNIT_TASK_NAME+=("$DEXJOCO_TASK")
            UNIT_INSTRUCTION+=("$SERVER_PROMPT")
            UNIT_EMBODIMENT_TAG+=("$DEXJOCO_EMBODIMENT_TAG")
            UNIT_FAMILY+=("$FAMILY")
            UNIT_RUN_IDX+=("$i")
            UNIT_RUN_SEED+=("$RUN_SEED")
            UNIT_EVAL_DIR+=("$CUR_EVAL_DIR")
            UNIT_SERVER_KEY+=("$SERVER_KEY")
            GROUP_UNITS[$GROUP_ID]="${GROUP_UNITS[$GROUP_ID]} $UNIT_IDX"
        done
    done
done

EVAL_LAUNCHED="${#UNIT_FAMILY[@]}"
EVAL_PARALLEL_WORKERS="$EVAL_GPU_COUNT"
if [ "$EVAL_PARALLEL_WORKERS" -gt "$EVAL_LAUNCHED" ]; then
    EVAL_PARALLEL_WORKERS="$EVAL_LAUNCHED"
fi

if [ "$EVAL_LAUNCHED" -gt 0 ]; then
    QUEUE_DIR="$JOB_LOG_DIR/dexjoco_worker_queues"
    rm -rf -- "$QUEUE_DIR"
    mkdir -p "$QUEUE_DIR"
    WORKER_QUEUE_FILES=()
    WORKER_LOADS=()
    for ((slot = 0; slot < EVAL_PARALLEL_WORKERS; slot++)); do
        WORKER_QUEUE_FILES+=("$QUEUE_DIR/gpu_${slot}.queue")
        WORKER_LOADS+=(0)
        : > "${WORKER_QUEUE_FILES[$slot]}"
    done

    if [ "${#GROUP_KEYS[@]}" -ge "$EVAL_PARALLEL_WORKERS" ]; then
        # Keep compatible units together so each GPU reuses its loaded server.
        for group_id in "${!GROUP_KEYS[@]}"; do
            best_slot=0
            for ((slot = 1; slot < EVAL_PARALLEL_WORKERS; slot++)); do
                if [ "${WORKER_LOADS[$slot]}" -lt "${WORKER_LOADS[$best_slot]}" ]; then
                    best_slot="$slot"
                fi
            done
            for unit_idx in ${GROUP_UNITS[$group_id]}; do
                printf '%s\n' "$unit_idx" >> "${WORKER_QUEUE_FILES[$best_slot]}"
                WORKER_LOADS[$best_slot]=$((WORKER_LOADS[$best_slot] + 1))
            done
        done
    else
        # When there are fewer server groups than GPUs, split their runs so all
        # available GPUs still receive useful work.
        slot=0
        for group_id in "${!GROUP_KEYS[@]}"; do
            for unit_idx in ${GROUP_UNITS[$group_id]}; do
                printf '%s\n' "$unit_idx" >> "${WORKER_QUEUE_FILES[$slot]}"
                WORKER_LOADS[$slot]=$((WORKER_LOADS[$slot] + 1))
                slot=$(((slot + 1) % EVAL_PARALLEL_WORKERS))
            done
        done
    fi

    log "MuJoCo eval: $EVAL_LAUNCHED units across $EVAL_PARALLEL_WORKERS persistent GPU workers"
    for ((slot = 0; slot < EVAL_PARALLEL_WORKERS; slot++)); do
        log "  gpu_slot=$slot queued_units=${WORKER_LOADS[$slot]}"
        run_gpu_worker "$slot" "${WORKER_QUEUE_FILES[$slot]}" &
        WORKER_PIDS+=("$!")
    done
    for pid in "${WORKER_PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILED=1
        fi
    done
fi

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
    "$EVAL_PARALLEL_WORKERS" "$MULTI_TASK" "$TASKS_JSON" \
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
