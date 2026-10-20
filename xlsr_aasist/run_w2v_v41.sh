#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p exp
LOG="$(pwd)/exp/v41_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$LOG" > exp/.latest_v41_log
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -X faulthandler -u -m w2v_v41.workflow "$@" 2>&1 | tee --ignore-interrupts -a "$LOG"
