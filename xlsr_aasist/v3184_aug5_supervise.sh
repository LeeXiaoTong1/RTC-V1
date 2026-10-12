#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
LOG=$1
shift
python -X faulthandler -u -m w2v_v3184_aug5.workflow "$@" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true' TERM INT
wait "$CHILD"
STATUS=$?
if kill -0 "$CHILD" 2>/dev/null; then wait "$CHILD"; STATUS=$?; fi
printf '\n[Exit] V3.18.4 exit_code=%s\n' "$STATUS"
printf '%s\n' "$STATUS" > "$LOG.exit"
exit "$STATUS"
