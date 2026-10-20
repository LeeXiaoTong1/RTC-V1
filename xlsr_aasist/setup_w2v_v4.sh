#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ENV_ROOT="${V4_ENV_ROOT:-/home/ubuntu/LXT/v318-runtime/envs/sdd-v4}"
BASE_PYTHON="${V4_BASE_PYTHON:-python}"
"$BASE_PYTHON" -c 'import torch, soundfile, scipy, imageio_ffmpeg; import webrtc_audio_processing; assert torch.cuda.is_available(), "Activate the existing CUDA training environment first"'
if [[ ! -f "$ENV_ROOT/pyvenv.cfg" ]]; then
  if [[ -e "$ENV_ROOT" ]]; then
    printf 'Refusing to change an existing non-venv directory: %s\n' "$ENV_ROOT" >&2
    exit 1
  fi
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_ROOT"
fi
"$ENV_ROOT/bin/python" -m pip install -r requirements_v4.txt
"$ENV_ROOT/bin/python" -c 'import torch, transformers, soundfile; assert transformers.__version__ == "4.38.2"; assert torch.cuda.is_bf16_supported(); print("V4_RUNTIME_READY", torch.__version__, transformers.__version__)'
"$ENV_ROOT/bin/python" -m w2v_v31810.processes
"$ENV_ROOT/bin/python" -m w2v_v4.interface_smoke --device cuda:0 --backend-precision fp16
printf '\nActivate before every V4 command:\nsource %q/bin/activate\n' "$ENV_ROOT"
