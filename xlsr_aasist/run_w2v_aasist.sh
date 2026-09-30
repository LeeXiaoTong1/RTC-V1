#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python -m w2v_aasist.launch --check
mkdir -p exp
LOG="$ROOT/exp/aasist_$(date +%Y%m%d_%H%M%S)_$$.log"
export AASIST_PROGRESS_FILE="$LOG.progress.json"
export AASIST_PROGRESS_LOG="$LOG"
nohup python -u live_progress.py --run train -- "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_aasist_log
printf '%s\n' "$PID" > exp/.latest_aasist_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
echo 'Live view: bash watch_w2v_progress.sh (Ctrl+C closes viewer only)'
echo 'Detailed log: tail -n 60 -f "$(cat exp/.latest_aasist_log)"'
