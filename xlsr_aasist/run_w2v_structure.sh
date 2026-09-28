#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SOURCE=${1:?Usage: bash run_w2v_structure.sh PREVIOUS_RUN [--upload-temp]}
shift
test -f "$SOURCE/stage3/config.json" || { echo "Missing source config: $SOURCE/stage3/config.json" >&2; exit 2; }
mkdir -p exp
LOG="$ROOT/exp/structure_$(date +%Y%m%d_%H%M%S)_$$.log"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
nohup python -u start_w2v_structure.py --from-run "$SOURCE" "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_structure_log
printf '%s\n' "$PID" > exp/.latest_structure_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
echo 'Default: read-only small mechanism audit. Training requires explicit --mode train --reviewed-audit.'
echo 'Progress: tail -n 60 -f "$(cat exp/.latest_structure_log)"'
