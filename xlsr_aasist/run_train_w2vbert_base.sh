#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_ROOT=${DATASET_ROOT:-"$SCRIPT_DIR/dataset"}
DATA_ROOT=${DATA_ROOT:-"$DATASET_ROOT/wav"}
export W2VBERT_PRETRAINED=${W2VBERT_PRETRAINED:-"$SCRIPT_DIR/../pretrained/w2v-bert-2.0"}
export RTC_B_NOISE_MANIFEST=${RTC_B_NOISE_MANIFEST:-"$SCRIPT_DIR/../external_noise/noise_split/train.jsonl"}
export RTC_B_NOISE_PROB=${RTC_B_NOISE_PROB:-0.5}
export RTC_B_SNR_MIN=${RTC_B_SNR_MIN:-10}
export RTC_B_SNR_MAX=${RTC_B_SNR_MAX:-30}

args=(
  --track "${TRACK:-w2vbert_aasist_base}"
  --ssl_path "$W2VBERT_PRETRAINED"
  --train_data_path "$DATA_ROOT/train"
  --dev_data_path "$DATA_ROOT/dev"
  --train_protocol "$DATASET_ROOT/train_label.txt"
  --dev_protocol "$DATASET_ROOT/dev_label.txt"
  --out_path "${OUT_PATH:-./exp}"
  --device "${DEVICE:-cuda:0}"
  --batch_size "${BATCH_SIZE:-40}"
  --num_epochs "${NUM_EPOCHS:-100}"
  --earlystop_epoch "${EARLYSTOP_EPOCH:-10}"
  --encoder_lr "${ENCODER_LR:-1e-6}"
  --backend_lr "${BACKEND_LR:-1e-4}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}"
  --num_workers "${NUM_WORKERS:-12}"
  --algo "${RAWBOOST_ALGO:-5}"
)
python main_train_w2vbert.py "${args[@]}" "$@"
