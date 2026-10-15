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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v3187.config "${ARGS[@]}"; fi
done
python -m w2v_v3187.config "${ARGS[@]}"
python -m w2v_v3187.processes
export TMPDIR="${V3187_TMPDIR:-$PWD/.tmp_v3187}"
mkdir -p "$TMPDIR" exp
LOG="$PWD/exp/v3187_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python -m w2v_v3187.launcher "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v3187_log
printf '%s\n' "$PID" > exp/.latest_v3187_pid
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v3187.sh; fi
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
