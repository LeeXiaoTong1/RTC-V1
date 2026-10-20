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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v3191.config "${ARGS[@]}"; fi
done
python -m w2v_v3191.config "${ARGS[@]}"
python -m w2v_v3191.processes
export TMPDIR="${V3191_TMPDIR:-$PWD/.tmp_v3191}"
mkdir -p "$TMPDIR" exp
LOG="$PWD/exp/v3191_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python -m w2v_v3191.launcher "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v3191_log
printf '%s\n' "$PID" > exp/.latest_v3191_pid
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v3191.sh; fi
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
