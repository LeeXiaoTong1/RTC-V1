#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
export TMPDIR="${TMPDIR:-$PWD/.tmp_v3184_finish}"
mkdir -p "$TMPDIR"
exec python -u finish_w2v_v3184_aug5.py "$@"
