#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SOURCE=${1:?Usage: bash run_w2v_train_audit.sh RUN_DIR [--upload-temp]}
shift
test -f "$SOURCE/stage3/config.json" || { echo "Missing $SOURCE/stage3/config.json" >&2; exit 2; }
python -c 'import numpy, soundfile' || { echo 'Use the existing training Python environment.' >&2; exit 2; }
JOB="w2v_train_audit_$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/exp/$JOB"
LOG="$OUT.log"
mkdir -p "$ROOT/exp"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
nohup python -u audit_w2v_train.py --run-dir "$SOURCE" --out "$OUT" --download-dir "$HOME/LXT/temp" --workers 4 "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_train_audit_log
printf '%s\n' "$OUT" > exp/.latest_train_audit_dir
printf '%s\n' "$PID" > exp/.latest_train_audit_pid
printf 'PID=%s\nLOG=%s\nREPORT_DIR=%s\n' "$PID" "$LOG" "$OUT"
echo 'CPU-only Train audit started in background; no model loading or training.'
echo 'Progress: tail -n 30 -f "$(cat exp/.latest_train_audit_log)"'
