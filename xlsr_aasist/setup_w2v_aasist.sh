#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Preserve the existing CUDA PyTorch; install only the small adaptation dependencies.
python -c 'import torch; assert torch.cuda.is_available(), "Activate the existing GPU sdd environment"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip install -r requirements_w2v_aasist.txt
python -m w2v_aasist.launch --check
OMP_NUM_THREADS=1 python -m unittest w2v_aasist.tests -v
echo 'AASIST_SETUP_COMPLETE=True'
