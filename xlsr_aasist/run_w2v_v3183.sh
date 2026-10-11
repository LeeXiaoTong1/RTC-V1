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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v3183.config "${ARGS[@]}"; fi
done
python -m w2v_v3183.config "${ARGS[@]}"
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v3183_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python v3183_detach.py "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v3183_log
printf '%s\n' "$PID" > exp/.latest_v3183_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.18.3: verified epoch-10 C3 continuation; 6 additional epochs; zero-init layer fusion; 4/2/7/3 sampling; controlled RTC'
printf '%s\n' 'View: bash watch_w2v_v3183.sh ; Ctrl+C closes the viewer only.'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v3183.sh; fi
