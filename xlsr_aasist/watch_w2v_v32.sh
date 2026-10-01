#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec python -u v32_console.py --log "$(cat exp/.latest_v32_log)" "$@"
