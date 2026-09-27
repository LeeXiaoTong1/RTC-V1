#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SOURCE=${1:?Usage: bash run_w2v_adapt.sh PREVIOUS_RUN_DIRECTORY}
test -f "$SOURCE/stage3/config.json" || { echo "Missing source config: $SOURCE/stage3/config.json" >&2; exit 2; }
mkdir -p exp
LOG="$ROOT/exp/adapt_$(date +%Y%m%d_%H%M%S)_$$.log"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
nohup python -u start_w2v_adapt.py --from-run "$SOURCE" --run > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_train_log
printf '%s\n' "$PID" > exp/.latest_adapt_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
echo 'Background launch requested. Check the log for hash/cache verification and progress.'
echo 'View log: tail -n 60 -f "$(cat exp/.latest_train_log)"'
