#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CUDA_PROFILE=auto
if [[ $# -gt 0 ]]; then
  if [[ $# != 2 || "$1" != --cuda || ! "$2" =~ ^(auto|11\.8|12\.6)$ ]]; then
    printf '%s\n' 'Usage: bash setup_w2v_v318.sh [--cuda auto|11.8|12.6]' >&2
    exit 2
  fi
  CUDA_PROFILE="$2"
fi
# bootstrap validates the actual Python/Conda prefix, including full-path activation.
export PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m w2v_v318.bootstrap guard
PROFILE="$(python -m w2v_v318.bootstrap profile --cuda "$CUDA_PROFILE")"
printf 'V318_INSTALL_PROFILE=%s\n' "$PROFILE"
mkdir -p exp/runtime_builds
python -m pip freeze >"exp/runtime_builds/packages_before_$(date +%Y%m%d_%H%M%S)_$$.txt"
if [[ "$PROFILE" == cu118 ]]; then
  python -m pip install --index-url https://pypi.org/simple numpy==1.26.4 packaging==24.2
  # Do not use the old cu126 fairseq2n wheel with a different Torch ABI.
  python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    'torch==2.6.0+cu118' 'torchaudio==2.6.0+cu118'
  # Fail early on actual GPU operations, before source compilation or 3B download.
  python -m w2v_v318.environment --stage cuda-torch
  if ! python -c 'from w2v_v318.runtime import native_abi; native_abi("cu118")' >/dev/null 2>&1; then
    bash build_fairseq2_cu118_v318.sh
  fi
  python -m pip install --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r requirements_w2v_v318_cu118.txt
else
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
fi
python -m w2v_v318.environment
REGRESSION_LOG="exp/runtime_builds/regression_$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'V318_REGRESSION_LOG=%s; synthetic test data only\n' "$REGRESSION_LOG"
if ! python -m unittest w2v_v318.test_core w2v_v318.test_assets w2v_v318.test_runtime w2v_v318.test_environment w2v_v318.test_augment w2v_v318.test_bootstrap -v >"$REGRESSION_LOG" 2>&1; then
  tail -n 60 "$REGRESSION_LOG" >&2
  exit 1
fi
printf '%s\n' 'V318_REGRESSIONS_PASSED=True'
printf '%s\n' 'V318_SETUP_COMPLETE=True; old sdd environment unchanged; model download is a separate resumable step'
