#!/usr/bin/env bash
# Retire the abandoned V3.5 run before V3.6. Dry-run unless explicitly --apply.
set -euo pipefail
cd "$(dirname "$0")"
if (( $# > 1 )) || [[ ${1:-} != '' && ${1:-} != --apply ]]; then
  printf '%s\n' 'Usage: bash cleanup_w2v_v36.sh [--apply]'; exit 2
fi
RUN=${V35_RETIRED_RUN:-exp/w2v_v35_20261003_161443_4fb5}
python retire_w2v_v35.py --run "$RUN" "$@"
python -m w2v_v35.maintenance "$@"
