#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CHECKPOINT=${1:?Usage: bash run_eval_rtc_noisy_w2vbert.sh CHECKPOINT [OUTPUT_DIR] [DEVICE]}
OUTPUT_DIR=${2:-./exp/eval/w2vbert_RTC_noisy_v2}
DEVICE=${3:-cuda:0}
DATASET_ROOT=${DATASET_ROOT:-"$SCRIPT_DIR/dataset"}
DATA_ROOT=${DATA_ROOT:-"$DATASET_ROOT/wav"}
export W2VBERT_PRETRAINED=${W2VBERT_PRETRAINED:-"$SCRIPT_DIR/../pretrained/w2v-bert-2.0"}

python main_eval_rtc_noisy_w2vbert.py \
  --model_path "$CHECKPOINT" \
  --eval_data_path "$DATA_ROOT/progress" \
  --protocol_path "$DATASET_ROOT/progress.txt" \
  --score_path "$OUTPUT_DIR/progress_scores.txt" \
  --submission_zip "$OUTPUT_DIR/submission.zip" \
  --device "$DEVICE" \
  --batch_size "${BATCH_SIZE:-40}" \
  --num_workers "${NUM_WORKERS:-4}"
