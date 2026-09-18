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
  --num_epochs "${NUM_EPOCHS:-20}"
  --earlystop_epoch "${EARLYSTOP_EPOCH:-5}"
  --encoder_lr "${ENCODER_LR:-1e-7}"
  --backend_lr "${BACKEND_LR:-1e-5}"
  --encoder_trainable_layers "${ENCODER_TRAINABLE_LAYERS:-8}"
  --class_weight_power "${CLASS_WEIGHT_POWER:-0.5}"
  --grad_clip "${GRAD_CLIP:-1.0}"
  --lr_factor "${LR_FACTOR:-0.5}"
  --lr_patience "${LR_PATIENCE:-2}"
  --min_encoder_lr "${MIN_ENCODER_LR:-5e-8}"
  --min_backend_lr "${MIN_BACKEND_LR:-5e-6}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}"
  --num_workers "${NUM_WORKERS:-12}"
  --algo "${RAWBOOST_ALGO:-5}"
)
python main_train_w2vbert.py "${args[@]}" "$@"
