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
