#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Keep the read-only observer's argv distinct from training jobs. The legacy
# single-GPU guard matches w2v_* run paths in argv, including optional --run.
ARGS=()
while (($#)); do
  case "$1" in
    --run)
      if (($# < 2)); then printf '%s\n' 'Missing value for --run' >&2; exit 2; fi
      export V317_MONITOR_RUN="$2"; shift 2 ;;
    --run=*) export V317_MONITOR_RUN="${1#--run=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec python -u v317_monitor.py "${ARGS[@]}"
