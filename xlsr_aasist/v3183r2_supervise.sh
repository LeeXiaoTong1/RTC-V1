#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
LOG=$1
shift
python -X faulthandler -u -m w2v_v3183r2.workflow "$@" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true' TERM INT
wait "$CHILD"
STATUS=$?
if kill -0 "$CHILD" 2>/dev/null; then wait "$CHILD"; STATUS=$?; fi
printf '\n[Exit] V3.18.3-R2 exit_code=%s\n' "$STATUS"
printf '%s\n' "$STATUS" > "$LOG.exit"
exit "$STATUS"
