#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CHECKPOINT=${1:?Usage: bash run_eval_w2v_rebuild.sh CHECKPOINT OUTPUT_DIR [progress|eval]}
OUT=${2:?Provide an output directory}
SPLIT=${3:-progress}
[[ "$SPLIT" == progress || "$SPLIT" == eval ]] || { echo 'Inference split must be progress or eval' >&2; exit 2; }
DATASET_ROOT=${DATASET_ROOT:-"$ROOT/dataset"}
DATA_ROOT=${DATA_ROOT:-"$DATASET_ROOT/wav"}
python -u -m w2v_rebuild.evaluate \
 --checkpoint "$CHECKPOINT" --out "$OUT" \
 --ssl_path "${W2VBERT_PRETRAINED:-$ROOT/../pretrained/w2v-bert-2.0}" \
 --eval_data_path "$DATA_ROOT/$SPLIT" --protocol_path "$DATASET_ROOT/$SPLIT.txt" \
 --device "${DEVICE:-cuda:0}" --microbatch "${MICROBATCH:-4}" --num_workers "${NUM_WORKERS:-4}"
