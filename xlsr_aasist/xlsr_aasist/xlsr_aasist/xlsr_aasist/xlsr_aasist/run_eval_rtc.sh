#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
MODEL_PATH=${1:?Usage: bash run_eval_rtc.sh CHECKPOINT [OUTPUT_DIR] [DEVICE]}
OUTPUT_DIR=${2:-./exp/eval/rtc_pair}
DEVICE=${3:-cuda:0}
DATA_ROOT=${DATA_ROOT:-"$SCRIPT_DIR/dataset/wav"}
DATASET_ROOT=${DATASET_ROOT:-"$SCRIPT_DIR/dataset"}
mkdir -p "$OUTPUT_DIR"
python main_eval.py \
  --eval_data_path "$DATA_ROOT/progress" \
  --protocol_path "$DATASET_ROOT/progress.txt" \
  --model_path "$MODEL_PATH" \
  --score_path "$OUTPUT_DIR/progress_scores.txt" \
  --device "$DEVICE" \
  --batch_size "${BATCH_SIZE:-32}" \
  --num_workers "${NUM_WORKERS:-8}"
