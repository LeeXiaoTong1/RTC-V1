#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# V3.18.3 introduces no dependencies. Preserve the known working parent runtime.
if [[ $# != 0 ]]; then
  printf '%s\n' 'Activate the existing V3.18 environment. This check accepts no installation options.' >&2
  exit 2
fi
python -c 'from w2v_v318.runtime import execution_profile, native_abi; print(native_abi(execution_profile()))'
python -m w2v_v3183.interface_smoke --device cuda:0
printf '%s\n' 'V3183_ENVIRONMENT_CHECK_OK=True; no packages installed or changed'
