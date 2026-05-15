# Helpers shared by train_body.sh and eval_body.sh.
# Sourced — does not run on its own.

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE:-/dev/null}"
}

detect_gpu_instance() {
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
