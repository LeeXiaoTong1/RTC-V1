#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Existing V3 dependencies are sufficient; do not replace CUDA PyTorch.
python -m w2v_aasist.launch --check
OMP_NUM_THREADS=1 python -m unittest w2v_v31.test_data_step w2v_v31.test_control w2v_v31.test_train w2v_v31.test_workflow -v
echo 'V31_SETUP_COMPLETE=True'
