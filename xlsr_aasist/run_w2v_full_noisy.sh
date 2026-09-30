#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python -m w2v_aasist.launch --check
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$ROOT/exp/full_noisy_$(date +%Y%m%d_%H%M%S)_$$.log"
export AASIST_PROGRESS_FILE="$LOG.progress.json"
export AASIST_PROGRESS_LOG="$LOG"
nohup python -u live_progress.py --run full -- "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_aasist_log
printf '%s\n' "$PID" > exp/.latest_aasist_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
echo 'Stages: generate/validate two full views -> retire obsolete caches -> Epoch 0 -> training'
echo 'Live view: bash watch_w2v_progress.sh (Ctrl+C closes viewer only)'
