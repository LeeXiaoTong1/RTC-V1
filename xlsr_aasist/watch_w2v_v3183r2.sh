#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python -u v3183r2_progress.py "${1:-$(cat exp/.latest_v3183r2_log)}"
