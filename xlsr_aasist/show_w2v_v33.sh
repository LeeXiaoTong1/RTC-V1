#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN=${V33_RUN:-$(cat exp/.latest_v33_run)}
if [[ -f "$RUN/report.md" ]]; then
  cat "$RUN/report.md"
else
  echo "No completed baseline/Dev evaluation yet. Run: $RUN"
fi
for ARM in control candidate; do
  if [[ -f "$RUN/$ARM/report.md" ]]; then
    printf '\n%s\n' "=== $ARM arm ==="
    cat "$RUN/$ARM/report.md"
  fi
done
