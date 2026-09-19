#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
STAGE=${1:?Usage: bash run_w2v_rebuild.sh 1|2|3|all RUN_DIR [--preflight|--check_data|--resume FILE]}
RUN=${2:?Provide one explicit run directory; never select checkpoints with ls -t}
shift 2
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
DATASET_ROOT=${DATASET_ROOT:-"$ROOT/dataset"}
DATA_ROOT=${DATA_ROOT:-"$DATASET_ROOT/wav"}
CACHE=${RTC_V2_CACHE_ROOT:-"$DATASET_ROOT/rtc_noisy_cache_v2"}
export W2VBERT_PRETRAINED=${W2VBERT_PRETRAINED:-"$ROOT/../pretrained/w2v-bert-2.0"}
export RTC_B_NOISE_MANIFEST=${RTC_B_NOISE_MANIFEST:-"$ROOT/../external_noise/noise_split/train.jsonl"}
export RTC_B_NOISE_PROB=${RTC_B_NOISE_PROB:-0.5}
export RTC_B_SNR_MIN=${RTC_B_SNR_MIN:-10}
export RTC_B_SNR_MAX=${RTC_B_SNR_MAX:-30}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
common=(
 --ssl_path "$W2VBERT_PRETRAINED"
 --train_data_path "$DATA_ROOT/train" --dev_data_path "$DATA_ROOT/dev"
 --train_protocol "$DATASET_ROOT/train_label.txt" --dev_protocol "$DATASET_ROOT/dev_label.txt"
 --rtc_pairs "${RTC_PAIRS_MANIFEST:-$DATASET_ROOT/train_rtc_pairs.jsonl}"
 --train_noise_manifest "$RTC_B_NOISE_MANIFEST"
 --train_noisy_cache "${RTC_V2_TRAIN_CACHE:-$CACHE/train_g0}"
 --dev_noisy_cache "${RTC_V2_DEV_SEEN_CACHE:-$CACHE/dev_seen}"
 --dev_heldout_cache "${RTC_V2_DEV_HELDOUT_CACHE:-$CACHE/dev_heldout}"
 --device "${DEVICE:-cuda:0}" --amp "${AMP:-bf16}"
 --microbatch "${MICROBATCH:-4}" --num_workers "${NUM_WORKERS:-4}"
)
resume=0
check=0
for x in "$@"; do
 [[ "$x" != --resume ]] || resume=1
 [[ "$x" != --check_data ]] || check=1
done
run_stage() {
 local s=$1
 shift
 local a=(--stage "$s" --out "$RUN/stage$s")
 if ((s > 1 && resume == 0 && check == 0)); then
   local previous="$RUN/stage$((s-1))/best_model.pt"
   test -f "$RUN/stage$((s-1))/completed.json" || { echo "Previous stage is not complete" >&2; return 2; }
   test -f "$previous" || { echo "Missing $previous" >&2; return 2; }
   a+=(--init "$previous")
 fi
 python -u -m w2v_rebuild.train "${common[@]}" "${a[@]}" "$@"
}
case "$STAGE" in
 1|2|3) run_stage "$STAGE" "$@" ;;
 all)
  (( $# == 0 )) || { echo "Use individual stages for preflight, resume, or custom arguments" >&2; exit 2; }
  run_stage 1
  run_stage 2
  run_stage 3
  ;;
 *) echo "Stage must be 1, 2, 3, or all" >&2; exit 2 ;;
esac
