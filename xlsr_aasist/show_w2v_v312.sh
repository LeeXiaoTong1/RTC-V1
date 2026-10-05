#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RUN=${V312_RUN:-$(cat exp/.latest_v312_run)}
if [[ -f "$RUN/report.md" ]]; then cat "$RUN/report.md"; else
  printf '%s\n' 'Final Dev comparison is not ready. The log shows current Train fitting stages.'
fi
if [[ -f "$RUN/failure.log" ]]; then cat "$RUN/failure.log"; fi
