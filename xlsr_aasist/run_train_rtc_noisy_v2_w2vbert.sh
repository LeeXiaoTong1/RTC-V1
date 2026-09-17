#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CHECKPOINT=${1:?Usage: bash run_train_rtc_noisy_v2_w2vbert.sh RTC_PAIR_CHECKPOINT [extra Python arguments]}
shift

DATASET_ROOT=${DATASET_ROOT:-"$SCRIPT_DIR/dataset"}
DATA_ROOT=${DATA_ROOT:-"$DATASET_ROOT/wav"}
CACHE_ROOT=${RTC_V2_CACHE_ROOT:-"$DATASET_ROOT/rtc_noisy_cache_v2"}
export W2VBERT_PRETRAINED=${W2VBERT_PRETRAINED:-"$SCRIPT_DIR/../pretrained/w2v-bert-2.0"}
export RTC_B_NOISE_MANIFEST=${RTC_B_NOISE_MANIFEST:-"$SCRIPT_DIR/../external_noise/noise_split/train.jsonl"}
export RTC_B_NOISE_PROB=${RTC_B_NOISE_PROB:-0.5}
export RTC_B_SNR_MIN=${RTC_B_SNR_MIN:-10}
export RTC_B_SNR_MAX=${RTC_B_SNR_MAX:-30}

args=(
  --model_path "$CHECKPOINT"
  --ssl_path "$W2VBERT_PRETRAINED"
  --train_data_path "$DATA_ROOT/train" --dev_data_path "$DATA_ROOT/dev"
  --train_protocol "$DATASET_ROOT/train_label.txt" --dev_protocol "$DATASET_ROOT/dev_label.txt"
  --rtc_pairs "${RTC_PAIRS_MANIFEST:-$DATASET_ROOT/train_rtc_pairs.jsonl}"
  --train_noisy_cache "${RTC_V2_TRAIN_CACHE:-$CACHE_ROOT/train_g0}"
  --dev_noisy_cache "${RTC_V2_DEV_SEEN_CACHE:-$CACHE_ROOT/dev_seen}"
  --dev_heldout_cache "${RTC_V2_DEV_HELDOUT_CACHE:-$CACHE_ROOT/dev_heldout}"
  --train_noise_manifest "$RTC_B_NOISE_MANIFEST"
  --track "${TRACK:-w2vbert_aasist_RTC_noisy_v2}" --out_path "${OUT_PATH:-./exp}"
  --batch_size "${ORDINARY_BATCH_SIZE:-24}"
  --rtc_pairs_per_batch "${RTC_PAIRS_PER_BATCH:-4}"
  --noisy_pairs_per_batch "${NOISY_PAIRS_PER_BATCH:-4}"
  --noisy_ce_weight "${NOISY_CE_WEIGHT:-0.3}"
  --rtc_weight "${RTC_WEIGHT:-0.1}" --noisy_weight "${NOISY_WEIGHT:-0.1}"
  --noisy_warmup_epochs 2 --rtc_temperature 0.1
  --num_epochs "${NUM_EPOCHS:-30}" --earlystop_epoch "${EARLYSTOP_EPOCH:-10}"
  --encoder_lr "${ENCODER_LR:-5e-7}"
  --backend_lr "${BACKEND_LR:-5e-6}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}" --algo "${RAWBOOST_ALGO:-5}"
  --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE:-cuda:0}" --amp "${AMP:-bf16}"
)
python main_train_rtc_noisy_v2_w2vbert.py "${args[@]}" "$@"
