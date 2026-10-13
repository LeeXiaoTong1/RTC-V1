#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
python -m w2v_aasist.launch --check
python -m w2v_v319.preflight
printf '%s\n' 'V319_SETUP_COMPLETE=True; existing Torch/CUDA preserved; transformers 4.38.2 verified; no packages/models downloaded'
