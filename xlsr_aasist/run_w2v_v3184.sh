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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v3184.config "${ARGS[@]}"; fi
done
python -m w2v_v3184.config "${ARGS[@]}"
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v3184_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python v3184_detach.py "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v3184_log
printf '%s\n' "$PID" > exp/.latest_v3184_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.18.4: six target scenes in both noisy views; 1 head/fusion adaptation + 5 joint epochs; 4/2/7/3 sampling; C3'
printf '%s\n' 'Default parent: completed R2 best, including fusion and BN. Fresh optimizer. Legacy RTC is the default; three-profile is optional.'
printf '%s\n' 'View: bash watch_w2v_v3184.sh ; Ctrl+C closes the viewer only.'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v3184.sh; fi
