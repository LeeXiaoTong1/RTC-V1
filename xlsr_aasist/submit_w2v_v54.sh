#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python -u -m w2v_v54.delivery export "$@"
