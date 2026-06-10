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
