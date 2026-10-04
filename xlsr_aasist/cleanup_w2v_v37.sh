#!/usr/bin/env bash
# Preview retirement of V3.6 jobs and safe obsolete checkpoints; apply only explicitly.
set -euo pipefail
cd "$(dirname "$0")"
if (( $# > 1 )) || [[ ${1:-} != '' && ${1:-} != --apply ]]; then
  printf '%s\n' 'Usage: bash cleanup_w2v_v37.sh [--apply]'; exit 2
fi
python -m w2v_v36.stop "$@"
# This maintenance command validates the original/submitted best and checkpoint references.
# It never deletes waveform caches, frozen feature matrices, or runtime model packages.
python -m w2v_v35.maintenance "$@"
