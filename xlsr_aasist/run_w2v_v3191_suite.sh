#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
WATCH=1
ARGS=()
for ARG in "$@"; do
  if [[ "$ARG" == --no-watch ]]; then WATCH=0; else ARGS+=("$ARG"); fi
done
for ARG in "${ARGS[@]}"; do
  if [[ "$ARG" == --help || "$ARG" == -h || "$ARG" == --stop-suite || "$ARG" == --stop-suite=* ]]; then
    exec python -m w2v_v3191.suite "${ARGS[@]}"
  fi
done
python -m w2v_v3191.processes
export TMPDIR="${V3191_TMPDIR:-$PWD/.tmp_v3191}"
mkdir -p "$TMPDIR" exp
LOG="$PWD/exp/v3191_suite_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python -m w2v_v3191.suite_launcher "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v3191_suite_log
printf '%s\n' "$PID" > exp/.latest_v3191_suite_pid
printf 'SUITE_PID=%s\nSUITE_LOG=%s\n' "$PID" "$LOG"
if [[ "$WATCH" == 1 && -t 1 ]]; then
  printf 'Closing this viewer with Ctrl+C leaves the suite running.\n'
  exec tail -n 40 -F "$LOG"
fi
