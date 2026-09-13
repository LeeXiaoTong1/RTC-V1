#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DATASET_ROOT=${DATASET_ROOT:-"$SCRIPT_DIR/dataset"}
CACHE_ROOT=${RTC_V2_CACHE_ROOT:-"$DATASET_ROOT/rtc_noisy_cache_v2"}
NOISE_ROOT=${NOISE_ROOT:-"$SCRIPT_DIR/../external_noise/noise_split"}
WORKERS=${CACHE_WORKERS:-4}
SEED=${CACHE_SEED:-1234}
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}
python check_rtc_noisy_v2.py --rtc --ffmpeg "$FFMPEG_BIN"
python prepare_rtc_noisy_v2.py --dataset_root "$DATASET_ROOT" --role train \
  --noise_manifest "$NOISE_ROOT/train.jsonl" --output "$CACHE_ROOT/train_g0" \
  --seed "$SEED" --generation 0 --workers "$WORKERS" --ffmpeg "$FFMPEG_BIN"
python prepare_rtc_noisy_v2.py --dataset_root "$DATASET_ROOT" --role dev_seen \
  --noise_manifest "$NOISE_ROOT/dev.jsonl" --output "$CACHE_ROOT/dev_seen" \
  --seed "$SEED" --workers "$WORKERS" --ffmpeg "$FFMPEG_BIN"
python prepare_rtc_noisy_v2.py --dataset_root "$DATASET_ROOT" --role dev_heldout \
  --noise_manifest "$NOISE_ROOT/dev.jsonl" --output "$CACHE_ROOT/dev_heldout" \
  --seed "$SEED" --workers "$WORKERS" --ffmpeg "$FFMPEG_BIN"
