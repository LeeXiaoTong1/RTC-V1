#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if (( $# < 1 )); then
  echo "Usage: bash run_train_rtc_w2vbert.sh BASE_CHECKPOINT [extra Python arguments]" >&2
  exit 2
fi
CHECKPOINT=$1
shift

DATASET_ROOT=${DATASET_ROOT:-"$SCRIPT_DIR/dataset"}
DATA_ROOT=${DATA_ROOT:-"$DATASET_ROOT/wav"}
export W2VBERT_PRETRAINED=${W2VBERT_PRETRAINED:-"$SCRIPT_DIR/../pretrained/w2v-bert-2.0"}
export RTC_B_NOISE_MANIFEST=${RTC_B_NOISE_MANIFEST:-"$SCRIPT_DIR/../external_noise/noise_split/train.jsonl"}
export RTC_B_NOISE_PROB=${RTC_B_NOISE_PROB:-0.5}
export RTC_B_SNR_MIN=${RTC_B_SNR_MIN:-10}
export RTC_B_SNR_MAX=${RTC_B_SNR_MAX:-30}

args=(
  --track "${TRACK:-w2vbert_aasist_V1_RTC_pair}"
  --model_path "$CHECKPOINT"
  --ssl_path "$W2VBERT_PRETRAINED"
  --rtc_pairs "${RTC_PAIRS_MANIFEST:-$DATASET_ROOT/train_rtc_pairs.jsonl}"
  --train_data_path "$DATA_ROOT/train"
  --dev_data_path "$DATA_ROOT/dev"
  --train_protocol "$DATASET_ROOT/train_label.txt"
  --dev_protocol "$DATASET_ROOT/dev_label.txt"
  --out_path "${OUT_PATH:-./exp}"
  --device "${DEVICE:-cuda:0}"
  --batch_size "${BATCH_SIZE:-32}"
  --rtc_pairs_per_batch "${RTC_PAIRS_PER_BATCH:-4}"
  --rtc_weight "${RTC_WEIGHT:-0.1}"
  --rtc_warmup_epochs "${RTC_WARMUP_EPOCHS:-2}"
  --rtc_temperature "${RTC_TEMPERATURE:-0.1}"
  --num_epochs "${NUM_EPOCHS:-8}"
  --earlystop_epoch "${EARLYSTOP_EPOCH:-4}"
  --encoder_lr "${ENCODER_LR:-1e-7}"
  --backend_lr "${BACKEND_LR:-2e-6}"
  --encoder_trainable_layers "${ENCODER_TRAINABLE_LAYERS:-8}"
  --grad_clip "${GRAD_CLIP:-1.0}"
  --lr_factor "${LR_FACTOR:-0.5}"
  --lr_patience "${LR_PATIENCE:-2}"
  --min_encoder_lr "${MIN_ENCODER_LR:-2.5e-8}"
  --min_backend_lr "${MIN_BACKEND_LR:-5e-7}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}"
  --num_workers "${NUM_WORKERS:-8}"
  --amp "${AMP:-bf16}"
  --selection_metric "${SELECTION_METRIC:-online_f1}"
)
python main_train_rtc_w2vbert.py "${args[@]}" "$@"
