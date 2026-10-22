#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RUN=""
if [[ $# -eq 2 && "$1" == --run ]]; then
  RUN="$2"
elif [[ $# -eq 0 ]]; then
  RUN="$(cat exp/.latest_v55_run)"
else
  printf 'Usage: bash stop_w2v_v55.sh [--run V5-run-directory]\n'
  exit 1
fi
python - "$RUN" <<'PY'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1]).expanduser().resolve()
cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
if cfg.get("version") != "5.5":
    raise ValueError("Refusing to stop a non-V5.5 run")
if (run / "completed.json").is_file():
    print("V5.5 run is already complete.")
else:
    (run / "STOP_REQUESTED").write_text("requested\n", encoding="utf-8")
    print("Stop requested; wait for the saved-boundary confirmation before restarting.")
PY
