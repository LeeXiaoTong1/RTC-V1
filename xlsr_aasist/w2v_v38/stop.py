"""Stop only this checkout's V3.8 workflow and verified descendants."""
import argparse
import os
from pathlib import Path
import signal
import time

from .common import ROOT
from w2v_v33.stop import inspect, owned_roots, descendants, same_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Run this command on the Linux training server')
    processes = [p for entry in Path('/proc').iterdir() if entry.name.isdigit()
                 for p in [inspect(int(entry.name))] if p is not None]
    targets = descendants(processes, owned_roots(processes, ROOT.resolve(), 'v38'))
    if any(p['pid'] == os.getpid() for p in targets):
        raise RuntimeError('Refusing to stop the stop command itself')
    if not targets:
        print('No active V3.8 workflow in this checkout.'); return
    for process in targets:
        print(f'PID={process["pid"]} ' + ' '.join(process['args']), flush=True)
    if not args.apply:
        print('Preview only; use --apply to stop these processes.'); return
    for process in targets:
        if same_process(process):
            try:
                os.kill(process['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(same_process(p) for p in targets):
        time.sleep(.25)
    pending = [p['pid'] for p in targets if same_process(p)]
    if pending:
        raise RuntimeError('Still exiting: ' + str(pending) + '; no forced kill issued')
    print('V38_STOPPED=True; completed small fitting stages and all original data retained.')


if __name__ == '__main__':
    main()
