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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v3184_aug5.config "${ARGS[@]}"; fi
done
python -m w2v_v3184_aug5.config "${ARGS[@]}"
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v3184_aug5_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python v3184_aug5_detach.py "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v3184_aug5_log
printf '%s\n' "$PID" > exp/.latest_v3184_aug5_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.18.4: 5 base views per Online/Offline; six noise scenes; ASP; 4/3/5/4 sampling; 1 warm + 5 joint epochs; C3'
printf '%s\n' 'Default parent: sealed/completed R2 best. Initial logits preserved; fresh optimizer; three-profile RTC.'
printf '%s\n' 'View: bash watch_w2v_v3184_aug5.sh ; Ctrl+C closes the viewer only.'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v3184_aug5.sh; fi
