#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Never replace the user's working CUDA PyTorch environment with a CPU wheel.
python -c 'import torch; assert torch.cuda.is_available(), "Activate the existing GPU sdd environment"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip install -r requirements_w2v_aasist.txt
python -m w2v_aasist.launch --check
OMP_NUM_THREADS=1 python -m unittest w2v_v3.test_model_step w2v_v3.test_control w2v_v3.test_data_workflow -v
echo 'V3_SETUP_COMPLETE=True'
