"""Keep all V3.7 messages and completed metrics above one live progress line."""
import argparse
import math
from pathlib import Path
import sys
import time
from live_progress import alive, read_state
from v33_console import Display, LogTail, safe_text

ROOT = Path(__file__).resolve().parent


def watch(log, once=False, interval=1., stream=None, owner_pid=None):
    display = Display(stream or sys.stdout)
    if not display.tty and not once:
        raise SystemExit('Use --once for redirected output; interactive viewer requires a terminal')
    tail = LogTail(log)
    display.messages(['V3.7: results remain above the live line. Ctrl+C closes only the viewer.', 'LOG='+str(log)])
    try:
        while True:
            while True:
                previous = tail.offset
                display.messages([safe_text(line) for line in tail.read()])
                if tail.offset-previous < 2*1024*1024:
                    break
            value = read_state(str(log)+'.progress.json') or dict(label='Waiting for startup',
                updated_at=time.time(), owner_pid=owner_pid)
            if value.get('status', 'running') == 'running' and alive(value.get('owner_pid')) is False:
                value = dict(value, status='stopped', label='Background job stopped; inspect log above', total=None)
            display.progress(value)
            if once or value.get('status') in ('complete', 'failed', 'stopped'):
                # Drain the final report/upload lines written before process exit.
                if not once and alive(value.get('owner_pid')) is True:
                    time.sleep(interval)
                    display.messages([safe_text(line) for line in tail.read()])
                    if alive(value.get('owner_pid')) is True:
                        continue
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        display.clear_line()
        display.messages(['Viewer closed; no signal was sent to the background job.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path)
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=float, default=1.)
    p.add_argument('--owner-pid', type=int)
    args = p.parse_args()
    if not math.isfinite(args.interval) or args.interval < .1:
        p.error('interval must be >= 0.1 seconds')
    if args.log is None:
        try:
            latest = (ROOT/'exp'/'.latest_v37_log').read_text(encoding='utf-8').strip()
        except OSError:
            p.error('No V3.7 log found; start a run or pass --log PATH')
        if not latest:
            p.error('The V3.7 log pointer is empty; pass --log PATH')
        log = Path(latest)
    else:
        log = args.log
    owner = args.owner_pid
    if owner is None and args.log is None:
        try:
            owner = int((ROOT/'exp'/'.latest_v37_pid').read_text(encoding='utf-8').strip())
        except (OSError, ValueError):
            pass
    watch(log, args.once, args.interval, owner_pid=owner)


if __name__ == '__main__':
    main()
