#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ${CONDA_DEFAULT_ENV:-} != sdd-v318 ]]; then
  printf '%s\n' 'Activate the isolated sdd-v318 environment (Python 3.11); create it first only if it does not exist.' >&2
  exit 2
fi
export PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# fairseq2n searches the Conda prefix for libsndfile.so.1; the pip soundfile
# wheel's private bundled library does not satisfy that native dependency.
conda install --prefix "$CONDA_PREFIX" -c conda-forge 'libsndfile=1.0.31' --freeze-installed -y
python -c 'import ctypes, os; from pathlib import Path; p=Path(os.environ["CONDA_PREFIX"])/"lib"/"libsndfile.so.1"; ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL); print("V318_LIBSNDFILE_OK="+str(p))'
python -m pip install 'torch==2.8.0' 'torchaudio==2.8.0' --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'fairseq2==0.6.0' --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126
python -m pip install -r requirements_w2v_v318.txt
python -m pip check
python -m w2v_v318.preflight
python -m unittest w2v_v318.test_core w2v_v318.test_assets w2v_v318.test_runtime -v
printf '%s\n' 'V318_SETUP_COMPLETE=True; old sdd environment unchanged; model download is a separate resumable step'
