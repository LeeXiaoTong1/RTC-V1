#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
tail -n 80 -f -- "$(cat exp/.latest_v4_log)"
