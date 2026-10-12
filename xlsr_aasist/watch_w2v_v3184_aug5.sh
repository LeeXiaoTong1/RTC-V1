#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
LOG="${1:-$(cat exp/.latest_v3184_aug5_log)}"
exec python -u v3184_progress.py "$LOG"
