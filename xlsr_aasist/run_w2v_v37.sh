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
LOG="$PWD/exp/v37_$(date +%Y%m%d_%H%M%S)_$$.log"
export AASIST_PROGRESS_FILE="$LOG.progress.json"
export AASIST_PROGRESS_LOG="$LOG"
nohup python -u -m w2v_v37.workflow "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_v37_log
printf '%s\n' "$PID" > exp/.latest_v37_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.7: verify submitted best -> Train-only language teacher -> reuse/extract frozen detector features -> fit one model -> guarded comparison'
printf '%s\n' 'Live view: bash watch_w2v_v37.sh; results: bash show_w2v_v37.sh'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec python -u v37_console.py --log "$LOG" --owner-pid "$PID"; fi
