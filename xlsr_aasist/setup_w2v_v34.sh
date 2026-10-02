#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python -m w2v_aasist.launch --check
# V3.4 reuses the V3.3 environment and completed cache; installs no packages.
python -m unittest discover -s w2v_v34 -t . -p 'test_*.py' -v
python -m unittest w2v_v33.test_transport test_live_progress w2v_v32.test_runtime w2v_v32.test_runtime_efficiency -v
echo 'V34_SETUP_COMPLETE=True'
