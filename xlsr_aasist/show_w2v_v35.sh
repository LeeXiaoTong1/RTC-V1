#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN=${V35_RUN:-$(cat exp/.latest_v35_run)}
if [[ -f "$RUN/report.md" ]]; then
    cat "$RUN/report.md"
else
    echo "No completed baseline/Dev evaluation yet. Run: $RUN"
fi
