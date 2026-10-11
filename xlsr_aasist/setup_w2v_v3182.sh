#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# The native stack is unchanged; reuse the audited installer and asset directory.
bash setup_w2v_v318.sh "$@"
python -m unittest w2v_v3182.test_core w2v_v3182.test_workflow -v
printf '%s\n' 'V3182_SETUP_COMPLETE=True; shared native stack, separate training runs'
