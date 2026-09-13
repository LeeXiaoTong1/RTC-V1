#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CHECKPOINT=${1:?Usage: bash run_eval_rtc_noisy.sh CHECKPOINT [OUTPUT_DIR] [DEVICE]}
OUTPUT_DIR=${2:-./exp/eval/RTC_noisy_pair}
DEVICE=${3:-cuda:0}
DATASET_ROOT=${DATASET_ROOT:-"$SCRIPT_DIR/dataset"}
DATA_ROOT=${DATA_ROOT:-"$DATASET_ROOT/wav"}
export XLSR_PRETRAINED=${XLSR_PRETRAINED:-"$SCRIPT_DIR/../pretrained/xlsr2_300m.pt"}
python main_eval_rtc_noisy.py \
  --model_path "$CHECKPOINT" \
  --eval_data_path "$DATA_ROOT/progress" \
  --protocol_path "$DATASET_ROOT/progress.txt" \
  --score_path "$OUTPUT_DIR/progress_scores.txt" \
  --submission_zip "$OUTPUT_DIR/submission.zip" \
  --device "$DEVICE" \
  --batch_size "${BATCH_SIZE:-40}" \
  --num_workers "${NUM_WORKERS:-4}"
