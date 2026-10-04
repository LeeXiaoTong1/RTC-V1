#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
WATCH=1
ARGS=()
for ARG in "$@"; do
  if [[ "$ARG" == --no-watch ]]; then WATCH=0; else ARGS+=("$ARG"); fi
done
python -m w2v_aasist.launch --check
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v36_$(date +%Y%m%d_%H%M%S)_$$.log"
export AASIST_PROGRESS_FILE="$LOG.progress.json"
export AASIST_PROGRESS_LOG="$LOG"
nohup python -u -m w2v_v36.workflow "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_v36_log
printf '%s\n' "$PID" > exp/.latest_v36_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.6: verify submitted best -> extract full features once -> fit final classifier -> guarded comparison'
printf '%s\n' 'Live view: bash watch_w2v_v36.sh; results: bash show_w2v_v36.sh'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec python -u v36_console.py --log "$LOG" --owner-pid "$PID"; fi
