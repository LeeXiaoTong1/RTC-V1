#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -m unittest w2v_v3183r2.test_core w2v_v3183r2.test_fusion w2v_v3184.test_noise w2v_v3184.test_initialization w2v_v3184.test_augment w2v_v3184.test_workflow w2v_v3184.test_delivery -v
