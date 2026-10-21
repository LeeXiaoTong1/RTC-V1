#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for ARG in "$@"; do
  if [[ "$ARG" == --help || "$ARG" == -h ]]; then
    exec python -m w2v_v53.config "$@"
  fi
done
python -m w2v_v53.config "$@"
mkdir -p exp
LOG="$PWD/exp/v53_$(date +%Y%m%d_%H%M%S)_$$.log"
printf '%s\n' "$LOG" > exp/.latest_v53_log
printf 'LOG=%s\n' "$LOG"
# The foreground workflow saves a committed boundary on Ctrl+C. tee must keep
# the log pipe open while that handler writes the final checkpoint.
python -u -m w2v_v53.workflow "$@" 2>&1 | tee -i "$LOG"
