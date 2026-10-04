#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python -u v37_console.py "$@"
