#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
if [[ -z ${V37_RUN:-} && ! -s exp/.latest_v37_run ]]; then
  printf '%s\n' 'No V3.7 run found. Set V37_RUN to a completed run.' >&2
  exit 2
fi
RUN=${V37_RUN:-$(cat exp/.latest_v37_run)}
DATASET=/home/ubuntu/LXT/RTC/xlsr_aasist/dataset
python -u -m w2v_v37.evaluate --run "$RUN" \
  --protocol "${EVAL_PROTOCOL:-$DATASET/progress.txt}" \
  --audio-root "${EVAL_AUDIO_ROOT:-$DATASET/wav/progress}" \
  --out "${SUBMISSION_DIR:-/home/ubuntu/LXT/temp/$(basename "$RUN")_submission}" "$@"
