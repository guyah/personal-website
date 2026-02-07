#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cleanup() {
	if [[ -n "${BACKEND_PID:-}" ]]; then
		kill "$BACKEND_PID" >/dev/null 2>&1 || true
	fi
}
trap cleanup EXIT

(
	cd "$ROOT_DIR/backend"
	uvicorn app.main:app --reload --port 8000
) &
BACKEND_PID=$!

(
	cd "$ROOT_DIR/site"
	npm run dev
)
