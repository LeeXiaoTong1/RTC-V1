#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ${CONDA_DEFAULT_ENV:-} != sdd-v318 ]]; then
  printf '%s\n' 'Activate the isolated sdd-v318 environment (Python 3.11); create it first only if it does not exist.' >&2
  exit 2
fi
export PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pip install 'torch==2.8.0' 'torchaudio==2.8.0' --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'fairseq2==0.6.0' --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126
python -m pip install -r requirements_w2v_v318.txt
python -m pip check
python -m w2v_v318.preflight
python -m unittest w2v_v318.test_core -v
printf '%s\n' 'V318_SETUP_COMPLETE=True; old sdd environment unchanged; model download is a separate resumable step'
