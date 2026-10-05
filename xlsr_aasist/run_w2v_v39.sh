#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
WATCH=1
ARGS=()
for ARG in "$@"; do
  if [[ "$ARG" == --no-watch ]]; then WATCH=0; else ARGS+=("$ARG"); fi
done
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v39_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash v39_supervise.sh "$LOG" "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_v39_log
printf '%s\n' "$PID" > exp/.latest_v39_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.9: verify V3.8/V3.7 baseline -> reuse vectors -> adaptive hard-example repair -> guarded full Dev comparison'
printf '%s\n' 'Live view: bash watch_w2v_v39.sh; results: bash show_w2v_v39.sh'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec python -u v39_console.py --log "$LOG" --owner-pid "$PID"; fi
