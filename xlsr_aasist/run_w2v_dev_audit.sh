#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SOURCE=${1:?Usage: bash run_w2v_dev_audit.sh ADAPTATION_RUN [DOWNLOAD_DIRECTORY]}
DOWNLOAD=${2:-"$HOME/LXT/temp"}
test -f "$SOURCE/stage3/candidate_best.pt" || { echo "Missing epoch-1 candidate in $SOURCE/stage3" >&2; exit 2; }
mkdir -p exp "$DOWNLOAD"
JOB="w2v_dev_audit_$(date +%Y%m%d_%H%M%S)_$$"
OUT="$ROOT/exp/$JOB"
LOG="$ROOT/exp/$JOB.log"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
nohup python -u audit_w2v_dev.py --run-dir "$SOURCE" --out "$OUT" --download-dir "$DOWNLOAD" --log-file "$LOG" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_dev_audit_log
printf '%s\n' "$OUT" > exp/.latest_dev_audit_dir
printf '%s\n' "$PID" > exp/.latest_dev_audit_pid
printf 'PID=%s\nLOG=%s\nREPORT_DIR=%s\nEXPECTED_DOWNLOAD=%s/%s.zip\n' "$PID" "$LOG" "$OUT" "$DOWNLOAD" "$JOB"
echo 'Read-only Dev comparison launched in background. Download after AUDIT_COMPLETE=True.'
echo 'Progress: tail -n 30 -f "$(cat exp/.latest_dev_audit_log)"'
