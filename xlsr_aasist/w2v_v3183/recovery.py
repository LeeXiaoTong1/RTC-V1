"""Keep telemetry consistent with the last atomically committed epoch."""
import json
from pathlib import Path


def trim_step_log(path, cursor):
    path = Path(path)
    if not path.exists():
        return
    tmp = path.with_suffix('.recovery.tmp')
    with path.open(encoding='utf-8') as src, tmp.open('w', encoding='utf-8', newline='\n') as dst:
        for line in src:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A killed write may leave one incomplete trailing record.
            if row['update'] <= cursor:
                dst.write(json.dumps(row, allow_nan=False)+'\n')
    tmp.replace(path)
