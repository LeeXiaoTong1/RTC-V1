#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
python -m w2v_aasist.launch --check
python -m unittest discover -s w2v_v313 -t . -p 'test_*.py' -v
printf '%s\n' 'V313_SETUP_COMPLETE=True; existing CUDA environment preserved; no packages/models downloaded'
