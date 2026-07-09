#!/usr/bin/env bash
# Boot both backend (FastAPI :8000) and frontend (Next.js :3000) in dev mode.
# Ctrl-C kills both.
set -euo pipefail

ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.." && pwd)"
export PATH="/opt/homebrew/bin:$PATH"   # so node, npm, uv are visible from Finder/Spotlight-launched terminals
ulimit -n 4096 2>/dev/null || true      # macOS default soft limit (256) is too low for the backend's subprocess fan-out

# Pull the latest code before booting. --ff-only so we never create a merge
# commit or clobber local work; if it can't fast-forward (dirty tree, diverged
# branch, offline), warn and start the servers with whatever is checked out
# rather than blocking the dev loop.
echo "==> Pulling latest ($(git -C "$ROOT" rev-parse --abbrev-ref HEAD))..."
if git -C "$ROOT" pull --ff-only; then
    echo "==> Up to date."
else
    echo "!!! git pull --ff-only failed (dirty tree, diverged, or offline) — starting with the current checkout." >&2
fi

cleanup() {
    if [[ -n "${BACKEND_PID:-}" ]]; then
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
    fi
    if [[ -n "${FRONTEND_PID:-}" ]]; then
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

(cd "$ROOT/backend" && uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload) &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo
echo "  backend  → http://localhost:8000"
echo "  frontend → http://localhost:3000"
echo

wait
