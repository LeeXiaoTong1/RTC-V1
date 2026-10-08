#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ${CONDA_DEFAULT_ENV:-} != sdd-omni ]]; then
  printf '%s\n' 'Use a separate environment: conda create -n sdd-omni python=3.11 -y && conda activate sdd-omni' >&2
  exit 2
fi
export PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# An explicit matching torch/fairseq2n ABI. Do not upgrade the working sdd env.
python -m pip install 'torch==2.8.0' 'torchaudio==2.8.0' --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'fairseq2==0.6.0' --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126
python -m pip install -r requirements_w2v_v316.txt
python -m pip check
python -m w2v_v316.preflight
python -m unittest discover -s w2v_v316 -t . -p 'test_*.py' -v
printf '%s\n' 'V316_SETUP_COMPLETE=True; environment isolated; no 7B download/training started'
