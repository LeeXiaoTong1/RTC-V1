#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python -u v3185_progress.py "$@"
