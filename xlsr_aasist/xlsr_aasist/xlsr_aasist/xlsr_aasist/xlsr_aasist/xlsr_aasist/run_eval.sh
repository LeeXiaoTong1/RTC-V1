#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/ubuntu/.conda/envs/sdd/bin:$PATH"
cd /home/ubuntu/LXT/RTC/xlsr_aasist

MODEL_PATH=${1:?Usage: bash run_eval.sh MODEL_PATH [OUTPUT_DIR] [DEVICE]}
OUTPUT_DIR=${2:-./exp/eval}
DEVICE=${3:-cuda:0}

DATA_ROOT="/home/ubuntu/LXT/RTC/xlsr_aasist/dataset/wav"
DATASET_ROOT="/home/ubuntu/LXT/RTC/xlsr_aasist/dataset"

BATCH_SIZE=${BATCH_SIZE:-40}
NUM_WORKERS=${NUM_WORKERS:-8}

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "找不到模型权重：$MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

args=(
  --eval_data_path "${DATA_ROOT}/progress"
  --protocol_path "${DATASET_ROOT}/progress.txt"
  --model_path "$MODEL_PATH"
  --score_path "${OUTPUT_DIR}/progress_scores.txt"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
)

python main_eval.py "${args[@]}"
  