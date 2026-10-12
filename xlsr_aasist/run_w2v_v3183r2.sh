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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v3183r2.config "${ARGS[@]}"; fi
done
python -m w2v_v3183r2.config "${ARGS[@]}"
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v3183r2_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python v3183r2_detach.py "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v3183r2_log
printf '%s\n' "$PID" > exp/.latest_v3183r2_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.18.3-R2: all-layer projected SwiGLU fusion; 1 head/fusion warmup + 5 joint epochs by default; 4/2/7/3 sampling; C3; controlled RTC'
printf '%s\n' 'Default initialization transfers compatible weights from the completed V3.18.3 best; this is a new architecture, not exact-score continuation.'
printf '%s\n' 'View: bash watch_w2v_v3183r2.sh ; Ctrl+C closes the viewer only.'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v3183r2.sh; fi
