#!/usr/bin/env bash
# A shell parent survives native Python faults and records their exit status.
set -u
cd "$(dirname "$0")"
LOG=$1
shift
export AASIST_PROGRESS_OWNER=$$
export AASIST_PROGRESS_FILE="$LOG.progress.json"
export AASIST_PROGRESS_LOG="$LOG"
python -X faulthandler -u -m w2v_v316.workflow "$@" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true' TERM INT
wait "$CHILD"
STATUS=$?
if kill -0 "$CHILD" 2>/dev/null; then
  wait "$CHILD"
  STATUS=$?
fi
python -u v316_status.py --log "$LOG" --exit-code "$STATUS"
exit "$STATUS"
