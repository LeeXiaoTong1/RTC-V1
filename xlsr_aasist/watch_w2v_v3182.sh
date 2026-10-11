#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python -u v3182_progress.py "${1:-$(cat exp/.latest_v3182_log)}"
