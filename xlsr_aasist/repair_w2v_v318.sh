#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SIZE=3b
if [[ $# -gt 0 ]]; then
  if [[ $# != 2 || "$1" != --omni-size || ! "$2" =~ ^(1b|3b)$ ]]; then
    printf '%s\n' 'Usage: bash repair_w2v_v318.sh [--omni-size 1b|3b]' >&2
    exit 2
  fi
  SIZE="$2"
fi
# Setup repairs the active isolated environment and runs every independent
# no-weight check. Stop before a large download if any check failed.
bash setup_w2v_v318.sh
python -m w2v_v318.prepare --omni-size "$SIZE"
python -m w2v_v318.preflight --omni-size "$SIZE" --weights
printf '%s\n' 'V318_REPAIR_COMPLETE=True; ready to start a new training run'
