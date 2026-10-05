"""Standard-library-only exit record, including SIGSEGV outside Python exceptions."""
import argparse
import json
import os
from pathlib import Path
import signal


def record(log, exit_code):
    status = 'complete' if exit_code == 0 else ('stopped' if exit_code in (130, 143) else 'failed')
    number = exit_code - 128 if 128 < exit_code < 193 else None
    try:
        name = signal.Signals(number).name if number else None
    except ValueError:
        name = 'signal_' + str(number)
    value = dict(status=status, exit_code=exit_code, signal=name,
                 note='Check log; SIGKILL alone does not establish an out-of-memory cause.' if number == 9 else '')
    target = Path(str(log) + '.exit.json')
    temporary = Path(str(target) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, target)
    print(f'[Exit] V3.8 {status}; exit_code={exit_code}' + (f'; signal={name}' if name else ''), flush=True)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', required=True)
    p.add_argument('--exit-code', required=True, type=int)
    args = p.parse_args()
    record(args.log, args.exit_code)
