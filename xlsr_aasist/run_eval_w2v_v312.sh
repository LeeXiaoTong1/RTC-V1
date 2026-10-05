#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
if [[ -z ${V312_RUN:-} && ! -s exp/.latest_v312_run ]]; then
  printf '%s\n' 'No V3.12 run found; set V312_RUN to a completed run.' >&2
  exit 2
fi
RUN=${V312_RUN:-$(cat exp/.latest_v312_run)}
DATASET=/home/ubuntu/LXT/RTC/xlsr_aasist/dataset
python -u -m w2v_v312.evaluate --run "$RUN" \
  --protocol "${EVAL_PROTOCOL:-$DATASET/progress.txt}" \
  --audio-root "${EVAL_AUDIO_ROOT:-$DATASET/wav/progress}" \
  --out "${SUBMISSION_DIR:-/home/ubuntu/LXT/temp/$(basename "$RUN")_submission}" "$@"
