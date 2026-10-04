#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python -m w2v_aasist.launch --check
# Never let a dependency resolver replace the working CUDA torch/torchaudio pair.
python -m pip install --no-deps -r requirements_w2v_v37.txt
python - <<'PY'
from importlib.metadata import version
import torch
import torchaudio
import hyperpyyaml
import sentencepiece
import threadpoolctl
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

for package in ('speechbrain', 'hyperpyyaml', 'sentencepiece', 'ruamel.yaml', 'ruamel.yaml.clib', 'threadpoolctl'):
    print(package + '=' + version(package), flush=True)
print('EXISTING_TORCH=' + torch.__version__, flush=True)
print('EXISTING_TORCHAUDIO=' + torchaudio.__version__, flush=True)
PY
python -m unittest discover -s w2v_v37 -t . -p 'test_*.py' -v
printf '%s\n' 'V37_SETUP_COMPLETE=True; existing CUDA Torch, checkpoints and audio/feature caches retained'
printf '%s\n' 'The full workflow prepares the pinned Train-only language teacher once; it is not used by submission inference.'
