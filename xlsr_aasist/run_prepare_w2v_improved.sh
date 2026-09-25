#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE=${1:-full}
DATASET_ROOT=${DATASET_ROOT:-"$ROOT/dataset"}
CACHE=${RTC_IMPROVED_CACHE_ROOT:-"$DATASET_ROOT/rtc_noisy_improved_v1"}
NOISE_ROOT=${NOISE_ROOT:-"$ROOT/../external_noise/noise_split"}
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export RTC_NOISE_CACHE_MB=${RTC_NOISE_CACHE_MB:-128}
common=(--dataset_root "$DATASET_ROOT" --workers "${CACHE_WORKERS:-4}"
        --seed "${CACHE_SEED:-1234}" --ffmpeg "$FFMPEG_BIN")
case "$MODE" in
 smoke) CACHE="$CACHE/smoke"; common+=(--limit 2) ;;
 full) ;;
 *) echo 'Usage: bash run_prepare_w2v_improved.sh smoke|full' >&2; exit 2 ;;
esac
python -u prepare_rtc_noisy_v2.py "${common[@]}" --role train --generation 1 \
  --processing_profile diverse --noise_manifest "${RTC_B_NOISE_MANIFEST:-$NOISE_ROOT/train.jsonl}" --output "$CACHE/train_g1"
python -u prepare_rtc_noisy_v2.py "${common[@]}" --role dev_seen \
  --processing_profile diverse --noise_manifest "${RTC_DEV_NOISE_MANIFEST:-$NOISE_ROOT/dev.jsonl}" --output "$CACHE/dev_seen"
python -u prepare_rtc_noisy_v2.py "${common[@]}" --role dev_heldout \
  --processing_profile unseen --noise_manifest "${RTC_DEV_NOISE_MANIFEST:-$NOISE_ROOT/dev.jsonl}" --output "$CACHE/dev_heldout"
