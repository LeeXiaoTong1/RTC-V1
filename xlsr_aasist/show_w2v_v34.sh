#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN=${V34_RUN:-$(cat exp/.latest_v34_run)}
if [[ -f "$RUN/report.md" ]]; then
    cat "$RUN/report.md"
else
    echo "No completed baseline/Dev evaluation yet. Run: $RUN"
fi
