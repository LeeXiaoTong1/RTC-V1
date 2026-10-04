#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RUN=${V36_RUN:-$(cat exp/.latest_v36_run)}
if [[ -f "$RUN/report.md" ]]; then cat "$RUN/report.md"; else
  printf '%s\n' 'Comparison not finished yet. Current phase:'
  python v36_console.py --once
fi
