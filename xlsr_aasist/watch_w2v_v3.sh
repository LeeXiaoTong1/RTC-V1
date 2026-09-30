#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec python -u live_progress.py --log "$(cat exp/.latest_v3_log)" "$@"
