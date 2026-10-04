#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RUN=${V37_RUN:-}
if [[ -z "$RUN" && -s exp/.latest_v37_run ]]; then RUN=$(cat exp/.latest_v37_run); fi
if [[ -n "$RUN" && -f "$RUN/report.md" ]]; then cat "$RUN/report.md"; else
  printf '%s\n' 'Comparison not finished yet. Current phase:'
  python v37_console.py --once
fi
