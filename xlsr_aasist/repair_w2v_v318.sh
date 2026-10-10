#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SIZE=3b
CUDA_PROFILE=auto
RUNTIME_ROOT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --omni-size) [[ $# -ge 2 && "$2" =~ ^(1b|3b)$ ]] || exit 2; SIZE="$2"; shift 2 ;;
    --cuda) [[ $# -ge 2 && "$2" =~ ^(auto|11\.8|12\.6)$ ]] || exit 2; CUDA_PROFILE="$2"; shift 2 ;;
    --runtime-root) [[ $# -ge 2 && -n "$2" ]] || exit 2; RUNTIME_ROOT="$2"; shift 2 ;;
    *) printf '%s\n' 'Usage: bash repair_w2v_v318.sh [--omni-size 1b|3b] [--cuda auto|11.8|12.6] [--runtime-root /absolute/path]' >&2; exit 2 ;;
  esac
done
if [[ -n "$RUNTIME_ROOT" ]]; then
  # Do not clone the incompatible cu126 environment or remove its packages.
  # Stage downloads and the new environment on the explicitly selected volume.
  RUNTIME_ROOT="$(python -m w2v_v318.bootstrap prepare-runtime --root "$RUNTIME_ROOT")"
  export TMPDIR="$RUNTIME_ROOT/tmp" CONDA_PKGS_DIRS="$RUNTIME_ROOT/pkgs"
  V318_PREFIX="$RUNTIME_ROOT/envs/sdd-v318"
  CONDA_BASE="$(conda info --base)"
  set +u
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  set -u
  if [[ ! -f "$V318_PREFIX/conda-meta/history" ]]; then
    conda create --prefix "$V318_PREFIX" python=3.11 -y
  fi
  # Future full-prefix activation must keep downloads off the full system disk.
  conda env config vars set --prefix "$V318_PREFIX" TMPDIR="$TMPDIR" CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS"
  set +u
  conda activate "$V318_PREFIX"
  set -u
  printf 'V318_ENVIRONMENT_PREFIX=%s\n' "$V318_PREFIX"
fi
# Setup repairs the active isolated environment and runs every independent
# no-weight check. Stop before a large download if any check failed.
bash setup_w2v_v318.sh --cuda "$CUDA_PROFILE"
python -m w2v_v318.prepare --omni-size "$SIZE"
python -m w2v_v318.preflight --omni-size "$SIZE" --weights
printf '%s\n' 'V318_REPAIR_COMPLETE=True; ready to start a new training run'
if [[ -n "$RUNTIME_ROOT" ]]; then
  printf 'Activate this environment in your terminal before training: conda activate %q\n' "$V318_PREFIX"
fi
