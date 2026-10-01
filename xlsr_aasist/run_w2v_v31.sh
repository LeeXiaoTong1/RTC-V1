#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
python -m w2v_aasist.launch --check
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$ROOT/exp/v31_$(date +%Y%m%d_%H%M%S)_$$.log"
export AASIST_PROGRESS_FILE="$LOG.progress.json"
export AASIST_PROGRESS_LOG="$LOG"
nohup python -u -m w2v_v31.workflow "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_v31_log
printf '%s\n' "$PID" > exp/.latest_v31_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
echo 'V3.1: reuse V3 best/cache -> validate baseline -> full/short adaptation -> report'
echo 'Live view: bash watch_w2v_v31.sh; metrics: bash show_w2v_v31.sh'
