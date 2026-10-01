#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python -m w2v_aasist.launch --check
if ! python -c 'import importlib.metadata; from webrtc_audio_processing import AudioProcessingModule; assert importlib.metadata.version("webrtc-audio-processing")=="0.1.3"'; then
  python -m pip install --no-deps -r requirements_rtc_diverse.txt
fi
python -c 'from rtc_noisy.diverse import webrtc_process; import numpy as np; x=np.sin(np.arange(3200)*.1).astype(np.float32)*.03; y=webrtc_process(x,1,12); assert y.shape==x.shape and np.isfinite(y).all(); print("V33_WEBRTC_DSP_OK=True")'
python -m unittest discover -s w2v_v33 -t . -p 'test_*.py' -v
python -m unittest w2v_v32.test_runtime w2v_v32.test_runtime_efficiency -v
echo 'V33_SETUP_COMPLETE=True'
