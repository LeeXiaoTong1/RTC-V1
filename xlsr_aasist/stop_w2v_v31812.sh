#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RUN="${1:-$(cat exp/.latest_v31812_run)}"
test -f "$RUN/config.json"
touch "$RUN/stop_requested"
printf 'Stop requested. Current extraction shard or residual update will finish safely: %s\n' "$RUN"
