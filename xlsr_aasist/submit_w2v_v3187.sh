#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${1:-}" == --retry ]]; then
  shift
  exec python -u -m w2v_v3187.delivery "$@"
fi
exec python -u -m w2v_v3187.evaluate --checkpoint best --upload-temp "$@"
