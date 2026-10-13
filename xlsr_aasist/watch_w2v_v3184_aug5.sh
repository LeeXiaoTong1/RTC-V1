#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python -u v3184_aug5_progress.py "$@"
