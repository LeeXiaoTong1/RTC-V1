#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
LOG=${1:-$(cat exp/.latest_v318_log)}
printf 'LOG=%s\n' "$LOG"
if [[ -f "$LOG.exit" ]]; then
  tail -n 160 "$LOG"
else
  PID=$(cat exp/.latest_v318_pid 2>/dev/null || true)
  if [[ "$LOG" == "$(cat exp/.latest_v318_log 2>/dev/null || true)" && "$PID" =~ ^[0-9]+$ ]]; then
    tail -n 160 --pid="$PID" -f "$LOG"
  else
    tail -n 160 -f "$LOG"
  fi
fi
