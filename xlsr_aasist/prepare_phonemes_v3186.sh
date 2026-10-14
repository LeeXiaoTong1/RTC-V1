#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
# Use the active Omni environment only as the read-only source of torch/numpy.
# The Python helper installs phoneme dependencies into a separate data-disk venv.
exec "${V3186_BASE_PYTHON:-python}" -u -m w2v_v3186.prepare_environment "$@"
