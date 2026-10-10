#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python -u v318_progress.py "${1:-$(cat exp/.latest_v318_log)}"
