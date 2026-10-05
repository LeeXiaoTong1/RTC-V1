"""Persistent results and explicit child-exit status above one live progress line."""
import argparse
import json
from pathlib import Path
import sys
import time

from live_progress import alive, read_state
from v33_console import Display, LogTail, safe_text

ROOT = Path(__file__).resolve().parent


def watch(log, once=False, owner_pid=None, stream=None, interval=1.):
    display, tail = Display(stream or sys.stdout), LogTail(log)
    if not display.tty and not once:
        raise SystemExit('Use --once when redirecting output')
    display.messages(['V3.10 results remain in the terminal. Ctrl+C closes only this viewer.', 'LOG=' + str(log)])
    ending = 'Viewer closed; background job was not signalled.'
    try:
        while True:
            while True:
                previous = tail.offset
                display.messages([safe_text(line) for line in tail.read()])
                if tail.offset - previous < 2 * 1024**2:
                    break
            exit_state = read_state(str(log) + '.exit.json')
            state = read_state(str(log) + '.progress.json') or dict(label='Waiting for startup', updated_at=time.time())
            if exit_state:
                ending = 'V3.10 ' + exit_state['status'] + '; exit_code=' + str(exit_state['exit_code'])
                if exit_state.get('signal'):
                    ending += '; ' + exit_state['signal'] + '. Inspect the traceback above.'
                # The exit record can precede the last log line by a few milliseconds.
                display.messages([safe_text(line) for line in tail.read()])
                break
            if alive(owner_pid or state.get('owner_pid')) is False:
                ending = 'Background supervisor exited without a final status; inspect this log before resuming.'
                break
            display.progress(state)
            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        display.clear_line()
        display.messages([ending])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path)
    p.add_argument('--owner-pid', type=int)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    log = args.log
    if log is None:
        pointer = ROOT / 'exp' / '.latest_v310_log'
        if not pointer.is_file():
            p.error('Start V3.10 first or pass --log PATH')
        log = Path(pointer.read_text(encoding='utf-8').strip())
        if args.owner_pid is None:
            try:
                args.owner_pid = int((ROOT / 'exp' / '.latest_v310_pid').read_text(encoding='utf-8').strip())
            except (OSError, ValueError):
                pass
    watch(log, args.once, args.owner_pid)


if __name__ == '__main__':
    main()
