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
for ARG in "${ARGS[@]}"; do
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v318.config "${ARGS[@]}"; fi
done
python -m w2v_v318.config "${ARGS[@]}"
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v318_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash v318_supervise.sh "$LOG" "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_v318_log
printf '%s\n' "$PID" > exp/.latest_v318_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.18: Omni W2V 1B + SSL-AASIST; 2 warmup + 8 joint epochs by default; fixed large Train probe; full Dev'
printf '%s\n' 'View: bash watch_w2v_v318.sh ; Ctrl+C closes the viewer only.'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v318.sh; fi
