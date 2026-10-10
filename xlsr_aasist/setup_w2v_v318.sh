#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ${CONDA_DEFAULT_ENV:-} != sdd-v318 ]]; then
  printf '%s\n' 'Activate the isolated sdd-v318 environment (Python 3.11); create it first only if it does not exist.' >&2
  exit 2
fi
export PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11 in sdd-v318"'
# fairseq2n searches the Conda prefix for libsndfile.so.1; the pip soundfile
# wheel's private bundled library does not satisfy that native dependency.
if ! python -c 'import ctypes, os; from pathlib import Path; ctypes.CDLL(str(Path(os.environ["CONDA_PREFIX"])/"lib"/"libsndfile.so.1"))' >/dev/null 2>&1; then
  conda install --prefix "$CONDA_PREFIX" -c conda-forge 'libsndfile=1.0.31' --freeze-installed -y
fi
python -c 'import ctypes, os; from pathlib import Path; p=Path(os.environ["CONDA_PREFIX"])/"lib"/"libsndfile.so.1"; ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL); print("V318_LIBSNDFILE_OK="+str(p))'
# Resolve the whole pinned scientific/native stack together. Already matching
# Torch/model packages are reused; incompatible Arrow/Pandas builds are replaced.
python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126 \
  -r requirements_w2v_v318.txt
python -m w2v_v318.environment
python -m unittest w2v_v318.test_core w2v_v318.test_assets w2v_v318.test_runtime w2v_v318.test_environment w2v_v318.test_augment -v
printf '%s\n' 'V318_SETUP_COMPLETE=True; old sdd environment unchanged; model download is a separate resumable step'
