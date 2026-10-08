#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -u -m w2v_v316_tfcl.cleanup "$@"
