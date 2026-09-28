#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SOURCE=${1:?Usage: bash run_w2v_pair_gradients.sh PREVIOUS_RUN [--upload-temp]}
shift
test -f "$SOURCE/stage3/config.json" || { echo "Missing source config: $SOURCE/stage3/config.json" >&2; exit 2; }
mkdir -p exp
LOG="$ROOT/exp/pair_gradients_$(date +%Y%m%d_%H%M%S)_$$.log"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
nohup python -u audit_w2v_pair_gradients.py --from-run "$SOURCE" "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_pair_gradients_log
printf '%s\n' "$PID" > exp/.latest_pair_gradients_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
echo 'Read-only Train gradient audit. No optimizer, parameter update or new audio cache.'
echo 'Progress: tail -n 60 -f "$(cat exp/.latest_pair_gradients_log)"'
