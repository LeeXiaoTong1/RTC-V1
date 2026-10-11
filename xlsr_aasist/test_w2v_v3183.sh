#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -m unittest w2v_v3183.test_core w2v_v3183.test_augment w2v_v3183.test_workflow -v
