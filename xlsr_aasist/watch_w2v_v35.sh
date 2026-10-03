#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec python -u v35_console.py --log "$(cat exp/.latest_v35_log)" "$@"
