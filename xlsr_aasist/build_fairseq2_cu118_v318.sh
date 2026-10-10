#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m w2v_v318.bootstrap guard
python -c 'from w2v_v318.runtime import execution_profile; assert execution_profile()=="cu118", "Install Torch 2.6.0+cu118 first"'
BUILD_ROOT="$(pwd)/exp/runtime_builds/fairseq2-cu118-6fa0aaf-py311"
SOURCE="$BUILD_ROOT/source"
COMMIT=6fa0aaf178db437bde0fae125b36105dc123119d
BUILD_JOBS="${V318_BUILD_JOBS:-2}"
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' 'V318_BUILD_JOBS must be a positive integer' >&2; exit 2; }
mkdir -p "$BUILD_ROOT/wheels"
LOG="$BUILD_ROOT/build_$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'V318_NATIVE_BUILD_LOG=%s\n' "$LOG"
on_error() {
  local status=$?
  printf '\nNative build failed (exit %s). Full log: %s\n' "$status" "$LOG" >&2
  tail -n 65 "$LOG" >&2
  exit "$status"
}
trap on_error ERR

printf '%s\n' '[Build 1/5] Installing isolated C/C++ tools, libsndfile and oneTBB'
V318_PREFIX="$CONDA_PREFIX"
conda install --prefix "$V318_PREFIX" -c conda-forge --freeze-installed -y \
  c-compiler=1.9.0 cxx-compiler=1.9.0 libsndfile=1.0.31 tbb-devel=2021.8.0 pkg-config >>"$LOG" 2>&1
# Compiler activation sets CC, CXX and CONDA_BUILD_SYSROOT, required upstream.
CONDA_BASE="$(conda info --base)"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$V318_PREFIX"
set -u
: "${CONDA_BUILD_SYSROOT:?Conda compiler activation did not set CONDA_BUILD_SYSROOT}"
python -m pip install --index-url https://pypi.org/simple \
  cmake==3.31.6 ninja==1.11.1.4 setuptools==80.9.0 wheel==0.45.1 tbb==2021.8.0 >>"$LOG" 2>&1

printf '%s\n' '[Build 2/5] Fetching the pinned official fairseq2 0.6 source'
if [[ ! -e "$SOURCE" ]]; then
  git clone --depth 1 --branch v0.6.0 https://github.com/facebookresearch/fairseq2.git "$SOURCE" >>"$LOG" 2>&1
fi
[[ "$(git -C "$SOURCE" remote get-url origin)" == https://github.com/facebookresearch/fairseq2.git ]]
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$COMMIT" ]]
git -C "$SOURCE" diff --quiet HEAD --
SUBMODULES=(native/third-party/pybind11 native/third-party/fmt native/third-party/sentencepiece native/third-party/zip native/third-party/kaldi-native-fbank)
git -C "$SOURCE" submodule update --init --recursive --depth 1 -- "${SUBMODULES[@]}" >>"$LOG" 2>&1
git -C "$SOURCE" submodule status --recursive >>"$LOG" 2>&1

printf '%s\n' '[Build 3/5] Compiling against the installed Torch CUDA 11.8 libraries (no nvcc required)'
# Only optional text-generation ngram CUDA kernels and image decoders are off.
# FindTorch.cmake still links torch_cuda/c10_cuda; W2V executes on the GPU.
cmake -S "$SOURCE/native" -B "$SOURCE/native/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DBUILD_TESTING=OFF -DFAIRSEQ2N_USE_CUDA=OFF -DFAIRSEQ2N_SUPPORT_IMAGE=OFF \
  -DFAIRSEQ2N_INSTALL_STANDALONE=ON -DFAIRSEQ2N_PYTHON_DEVEL=OFF >>"$LOG" 2>&1
cmake --build "$SOURCE/native/build" --parallel "$BUILD_JOBS" >>"$LOG" 2>&1

printf '%s\n' '[Build 4/5] Packaging the matching native extension'
# Upstream setup.py resolves ../build relative to native/python.
(
  cd "$SOURCE/native/python"
  python -m pip wheel --no-deps --no-build-isolation --wheel-dir "$BUILD_ROOT/wheels" .
) >>"$LOG" 2>&1
shopt -s nullglob
WHEELS=("$BUILD_ROOT"/wheels/fairseq2n-0.6-cp311-*.whl)
[[ ${#WHEELS[@]} == 1 ]] || { printf 'Expected one cp311 wheel, found %s\n' "${#WHEELS[@]}" >>"$LOG"; false; }
python -m pip install --no-deps --force-reinstall "${WHEELS[0]}" >>"$LOG" 2>&1

printf '%s\n' '[Build 5/5] Checking the native Torch/CUDA ABI and recording provenance'
python -m w2v_v318.bootstrap record-build --output "$BUILD_ROOT/build_manifest.json" >>"$LOG" 2>&1
python -c 'from w2v_v318.runtime import native_abi; print("V318_NATIVE_ABI=",native_abi("cu118"))'
printf 'V318_NATIVE_BUILD_COMPLETE=True; manifest=%s\n' "$BUILD_ROOT/build_manifest.json"
