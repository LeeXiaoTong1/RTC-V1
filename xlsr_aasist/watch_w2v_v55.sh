#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -gt 1 || "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: bash watch_w2v_v55.sh [log-file]\n'
  exit 0
fi
LOG="${1:-}"
if [[ -z "$LOG" ]]; then LOG="$(cat exp/.latest_v55_log)"; fi
[[ -f "$LOG" ]] || { printf 'Missing log: %s\n' "$LOG" >&2; exit 1; }
exec tail -n 50 -F "$LOG"
