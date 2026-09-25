#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
BASELINE=${1:?Usage: bash run_w2v_improved.sh BASELINE_STAGE3_BEST NEW_RUN [--preflight|--check_data|--resume FILE]}
RUN=${2:?Specify a NEW experiment directory}
shift 2
test -f "$BASELINE" || { echo "Missing baseline: $BASELINE" >&2; exit 2; }
DATASET_ROOT=${DATASET_ROOT:-"$ROOT/dataset"}
DATA_ROOT=${DATA_ROOT:-"$DATASET_ROOT/wav"}
OLD_CACHE=${RTC_V2_CACHE_ROOT:-"$DATASET_ROOT/rtc_noisy_cache_v2"}
NEW_CACHE=${RTC_IMPROVED_CACHE_ROOT:-"$DATASET_ROOT/rtc_noisy_improved_v1"}
export W2VBERT_PRETRAINED=${W2VBERT_PRETRAINED:-"$ROOT/../pretrained/w2v-bert-2.0"}
export RTC_B_NOISE_MANIFEST=${RTC_B_NOISE_MANIFEST:-"$ROOT/../external_noise/noise_split/train.jsonl"}
export RTC_B_NOISE_PROB=${RTC_B_NOISE_PROB:-0.5}
export RTC_B_SNR_MIN=${RTC_B_SNR_MIN:-10}
export RTC_B_SNR_MAX=${RTC_B_SNR_MAX:-30}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
common=(--stage 3 --out "$RUN/stage3" --ssl_path "$W2VBERT_PRETRAINED"
 --train_data_path "$DATA_ROOT/train" --dev_data_path "$DATA_ROOT/dev"
 --train_protocol "$DATASET_ROOT/train_label.txt" --dev_protocol "$DATASET_ROOT/dev_label.txt"
 --rtc_pairs "${RTC_PAIRS_MANIFEST:-$DATASET_ROOT/train_rtc_pairs.jsonl}"
 --train_noise_manifest "$RTC_B_NOISE_MANIFEST"
 --train_noisy_cache "${RTC_V2_TRAIN_CACHE:-$OLD_CACHE/train_g0}"
 --extra_train_noisy_cache "$NEW_CACHE/train_g1"
 --dev_noisy_cache "$NEW_CACHE/dev_seen" --dev_heldout_cache "$NEW_CACHE/dev_heldout"
 --ordinary_sampling balanced --noisy_bank_policy mixed --consistency_weight "${CONSISTENCY_WEIGHT:-0.02}" --consistency_confidence 0.8
 --feature_cache "${FEATURE_CACHE:-$DATASET_ROOT/w2v_feature_cache}"
 --noise_cache_mb "${RTC_NOISE_CACHE_MB:-128}"
 --device "${DEVICE:-cuda:0}" --amp bf16 --microbatch 4 --eval_microbatch 4 --eval_batch 16
 --num_workers "${NUM_WORKERS:-4}" --epochs "${EPOCHS:-8}"
 --encoder_lr "${ENCODER_LR:-2e-7}" --head_lr "${HEAD_LR:-5e-6}")
resume=0
for arg in "$@"; do
 [[ "$arg" != --resume ]] || resume=1
done
if ((resume == 0)); then
 common+=(--finetune_from "$BASELINE")
fi
python -u -m w2v_rebuild.train "${common[@]}" "$@"
