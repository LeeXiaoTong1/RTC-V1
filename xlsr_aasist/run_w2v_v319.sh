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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v319.config "${ARGS[@]}"; fi
done
python -m w2v_v319.config "${ARGS[@]}"
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
export TMPDIR="${TMPDIR:-$PWD/.tmp_v319}"
mkdir -p "$TMPDIR"
mkdir -p exp
LOG="$PWD/exp/v319_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python v319_detach.py "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v319_log
printf '%s\n' "$PID" > exp/.latest_v319_pid
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v319.sh; fi
