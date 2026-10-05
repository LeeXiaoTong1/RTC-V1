#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1
WATCH=1
ARGS=()
for ARG in "$@"; do
  if [[ "$ARG" == --no-watch ]]; then WATCH=0; else ARGS+=("$ARG"); fi
done
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$PWD/exp/v310_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash v310_supervise.sh "$LOG" "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_v310_log
printf '%s\n' "$PID" > exp/.latest_v310_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
printf '%s\n' 'V3.10: verify protected best -> train features + real-only language adversaries -> Dev and independent probes'
printf '%s\n' 'Live view: bash watch_w2v_v310.sh; results: bash show_w2v_v310.sh'
if [[ "$WATCH" == 1 && -t 1 ]]; then exec python -u v310_console.py --log "$LOG" --owner-pid "$PID"; fi
