#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -u w2v_v316/cleanup.py --root "$PWD" "$@"
