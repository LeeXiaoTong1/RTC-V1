#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
tail -n 160 "${1:-$(cat exp/.latest_v318_log)}"
