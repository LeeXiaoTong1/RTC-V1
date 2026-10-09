#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
python -u v3161_console.py --once "$@"
