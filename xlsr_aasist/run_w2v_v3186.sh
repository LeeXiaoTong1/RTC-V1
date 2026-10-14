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
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then exec python -m w2v_v3186.config "${ARGS[@]}"; fi
done
python -m w2v_v3186.config "${ARGS[@]}"
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
export TMPDIR="${V3186_TMPDIR:-$PWD/.tmp_v3186}"
mkdir -p "$TMPDIR"
mkdir -p exp
LOG="$PWD/exp/v3186_$(date +%Y%m%d_%H%M%S)_$$.log"
PID=$(python v3186_detach.py "$LOG" "${ARGS[@]}")
printf '%s\n' "$LOG" > exp/.latest_v3186_log
printf '%s\n' "$PID" > exp/.latest_v3186_pid
if [[ "$WATCH" == 1 && -t 1 ]]; then exec bash watch_w2v_v3186.sh; fi
