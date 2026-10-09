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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then
    exec python -m w2v_v3161.arguments "${ARGS[@]}"
  fi
done
python -m w2v_v3161.arguments "${ARGS[@]}"
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v3161_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash v3161_supervise.sh "$LOG" "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_v3161_log
printf '%s\n' "$PID" > exp/.latest_v3161_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.16.1: continue V3.16 LAST + Adam; four additional epochs; full SSL bidirectional TFCL; Dev once per epoch'
printf '%s\n' 'Live view: bash watch_w2v_v3161.sh; results: bash show_w2v_v3161.sh'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec python -u v3161_console.py --log "$LOG" --owner-pid "$PID"; fi
