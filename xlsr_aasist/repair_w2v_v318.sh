#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SIZE=3b
CUDA_PROFILE=auto
while [[ $# -gt 0 ]]; do
  case "$1" in
    --omni-size) [[ $# -ge 2 && "$2" =~ ^(1b|3b)$ ]] || exit 2; SIZE="$2"; shift 2 ;;
    --cuda) [[ $# -ge 2 && "$2" =~ ^(auto|11\.8|12\.6)$ ]] || exit 2; CUDA_PROFILE="$2"; shift 2 ;;
    *) printf '%s\n' 'Usage: bash repair_w2v_v318.sh [--omni-size 1b|3b] [--cuda auto|11.8|12.6]' >&2; exit 2 ;;
  esac
done
# Setup repairs the active isolated environment and runs every independent
# no-weight check. Stop before a large download if any check failed.
bash setup_w2v_v318.sh --cuda "$CUDA_PROFILE"
python -m w2v_v318.prepare --omni-size "$SIZE"
python -m w2v_v318.preflight --omni-size "$SIZE" --weights
printf '%s\n' 'V318_REPAIR_COMPLETE=True; ready to start a new training run'
