#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python -m w2v_aasist.launch --check
python -m unittest discover -s w2v_v36 -t . -p 'test_*.py' -v
printf '%s\n' 'V36_SETUP_COMPLETE=True; existing dependencies and audio caches reused'
