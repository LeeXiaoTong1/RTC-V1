#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
WATCH=1
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --no-watch ]]; then WATCH=0; else ARGS+=("$arg"); fi
done
python -m w2v_aasist.launch --check
python -c 'from w2v_aasist.full_workflow import ensure_idle; ensure_idle()'
mkdir -p exp
LOG="$ROOT/exp/v35_$(date +%Y%m%d_%H%M%S)_$$.log"
export AASIST_PROGRESS_FILE="$LOG.progress.json"
export AASIST_PROGRESS_LOG="$LOG"
nohup python -u -m w2v_v35.workflow "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$LOG" > exp/.latest_v35_log
printf '%s\n' "$PID" > exp/.latest_v35_pid
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
echo 'V3.5: verify protected reference -> official pretrained encoder -> balanced Online classification'
echo 'One head epoch then up to five joint epochs; rotating two-view noise bank; no separate control training.'
echo 'Live view: bash watch_w2v_v35.sh; metrics: bash show_w2v_v35.sh'
if [[ "$WATCH" == 1 && -t 1 ]]; then
    exec python -u "$ROOT/v35_console.py" --log "$LOG"
fi
