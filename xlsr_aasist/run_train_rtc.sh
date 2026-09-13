#!/usr/bin/env bash
set -euo pipefail

# Activate the existing sdd environment first. Paths are relative to this script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if (( $# < 2 )); then
  echo "Usage: bash run_train_rtc.sh V1_CHECKPOINT TRAIN_PAIRS_JSONL [extra Python arguments]" >&2
  exit 2
fi
CHECKPOINT=$1
PAIR_MANIFEST=$2
shift 2

DATA_ROOT=${DATA_ROOT:-"$SCRIPT_DIR/dataset/wav"}
DATASET_ROOT=${DATASET_ROOT:-"$SCRIPT_DIR/dataset"}
export RTC_B_NOISE_MANIFEST=${RTC_B_NOISE_MANIFEST:-"$SCRIPT_DIR/../external_noise/noise_split/train.jsonl"}
export RTC_B_NOISE_PROB=${RTC_B_NOISE_PROB:-0.5}
export RTC_B_SNR_MIN=${RTC_B_SNR_MIN:-10}
export RTC_B_SNR_MAX=${RTC_B_SNR_MAX:-30}

args=(
  --track "${TRACK:-xlsr_aasist_V1_RTC_pair}"
  --model_path "$CHECKPOINT"
  --rtc_pairs "$PAIR_MANIFEST"
  --train_data_path "$DATA_ROOT/train"
  --dev_data_path "$DATA_ROOT/dev"
  --train_protocol "$DATASET_ROOT/train_label.txt"
  --dev_protocol "$DATASET_ROOT/dev_label.txt"
  --out_path "${OUT_PATH:-./exp}"
  --device "${DEVICE:-cuda:0}"
  --batch_size "${BATCH_SIZE:-32}"
  --rtc_pairs_per_batch "${RTC_PAIRS_PER_BATCH:-4}"
  --rtc_weight "${RTC_WEIGHT:-0.1}"
  --rtc_temperature "${RTC_TEMPERATURE:-0.1}"
  --num_epochs "${NUM_EPOCHS:-5}"
  --earlystop_epoch "${EARLYSTOP_EPOCH:-3}"
  --lr "${LR:-1e-6}"
  --num_workers "${NUM_WORKERS:-8}"
  --amp "${AMP:-bf16}"
  --selection_metric "${SELECTION_METRIC:-online_f1}"
)
python main_train_rtc.py "${args[@]}" "$@"
