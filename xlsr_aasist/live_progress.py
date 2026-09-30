"""Read-only live viewer and lightweight status writer; no ML dependencies."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
_last_write = 0.
_warned = False


def publish(label, *, current=0, total=None, elapsed=0., loss=None,
            status='running', force=False):
    """At most one atomic replacement/second, except phase boundaries."""
    global _last_write, _warned
    target = os.environ.get('AASIST_PROGRESS_FILE')
    if not target:
        return
    now = time.monotonic()
    if not force and now - _last_write < 1.:
        return
    _last_write = now
    value = dict(label=label, current=current, total=total, elapsed=elapsed,
                 loss=loss, status=status, updated_at=time.time(), pid=os.getpid(),
                 owner_pid=int(os.environ.get('AASIST_PROGRESS_OWNER', os.getpid())),
                 log=os.environ.get('AASIST_PROGRESS_LOG', ''))
    temporary = Path(target + '.' + str(os.getpid()) + '.tmp')
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
        os.replace(temporary, target)
    except (OSError, ValueError) as exc:
        # Display failures must never discard a training step or abort training.
        if not _warned:
            print('LIVE_PROGRESS_WARNING=' + str(exc), file=sys.stderr, flush=True)
            _warned = True
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def phase(label):
    publish(label, force=True)


class Phase:
    def __init__(self, label, total):
        self.label, self.total, self.started = label, total, time.monotonic()
        publish(label, total=total, force=True)

    def update(self, current, loss=None):
        publish(self.label, current=current, total=self.total,
                elapsed=time.monotonic() - self.started, loss=loss,
                force=current == 1 or current == self.total)


def read_state(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def alive(pid):
    # Never send signals. Linux /proc also distinguishes unreaped zombies.
    if os.name != 'posix' or not Path('/proc').is_dir() or not pid:
        return None
    try:
        stat = (Path('/proc') / str(int(pid)) / 'stat').read_text()
        return stat.rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False
    except (OSError, ValueError, IndexError):
        return None


def log_state(path):
    """Compatibility for an already-running old job: never invent missing steps."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 131072))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        updated = Path(path).stat().st_mtime
    except OSError:
        return None
    value = dict(label='Waiting for logged progress', current=0, total=None,
                 elapsed=0., updated_at=updated, status='running', legacy=True)
    for line in lines:
        step = re.search(r'STEP (\d+)/(\d+) CE=([\deE.+-]+).*seconds/step=([\d.]+)', line)
        count = re.search(r'^((?:epoch_\d+ )?Dev \w+|Train source headers|Full noisy sources): (\d+)/(\d+)(?: elapsed=([\d.]+)m)?', line)
        if step:
            current, total, loss, rate = step.groups()
            value.update(label='Train (log)', current=int(current), total=int(total),
                         loss=float(loss), elapsed=float(rate)*int(current))
        elif count:
            label, current, total, minutes = count.groups()
            value.update(label=label, current=int(current), total=int(total),
                         loss=None, elapsed=float(minutes or 0)*60)
        elif 'TRAINING_COMPLETE=True' in line or 'PREPARATION_COMPLETE=True' in line:
            value.update(label='Work complete / report export', total=None, loss=None)
        elif line.startswith('TEMP_DOWNLOAD_URL='):
            value.update(label='Complete; download URL is in log', total=None, status='complete')
        elif line.startswith('Traceback (most recent call last)'):
            value.update(label='Error; see log', total=None, status='failed')
    return value


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--:--'
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def render(state, width=120, now=None):
    now = time.time() if now is None else now
    label = re.sub(r'[^\x20-\x7e]', ' ', str(state.get('label', 'Waiting')))
    status = state.get('status', 'running')
    current, total = state.get('current', 0), state.get('total')
    age = max(0., now - state.get('updated_at', now))
    if total:
        fraction = min(1., max(0., current / total))
        elapsed = state.get('elapsed', 0.)
        eta = max(0., elapsed / current * (total-current) - age) if current and elapsed else None
        loss = state.get('loss')
        loss_text = f' loss={loss:.5f}' if loss is not None else ''
        details = f' {fraction:6.2%} {current}/{total}{loss_text} ETA={duration(eta)}'
        available = max(0, width-1-len(details)-3)
        bar_width = min(16, max(4, available//3))
        filled = int(bar_width * fraction)
        bar = '#' * filled + '-' * (bar_width-filled)
        line = f'{label[:max(0, available-bar_width)]} [{bar}]' + details
    else:
        line = label
    if status != 'running':
        line += ' | ' + status.upper()
    elif age > 30:
        line += f' | no update {int(age)}s'
    if state.get('legacy'):
        line += ' | logged steps only'
    return line[:max(1, width-1)]


def watch(path, log=None, pid=None, once=False, interval=1.):
    tty = sys.stdout.isatty()
    if not tty and not once:
        raise SystemExit('Open the viewer directly in a terminal (no pipe). Use --once for a snapshot.')
    previous_width = 0
    try:
        while True:
            value = read_state(path) or (log_state(log) if log else None)
            value = value or dict(label='Waiting for the background job to publish progress', updated_at=time.time())
            owner = value.get('owner_pid') or pid
            if value.get('status', 'running') == 'running' and alive(owner) is False:
                value = dict(value, status='stopped', label='Launcher stopped; check log and child processes', total=None)
            width = shutil.get_terminal_size((120, 24)).columns
            line = render(value, width)
            if tty:
                # Clear the old line, including after a phase change or terminal resize.
                sys.stdout.write('\r' + ' ' * min(previous_width, max(1, width-1)) + '\r' + line)
                sys.stdout.flush()
                previous_width = len(line)
            else:
                print(line)
            if once or value.get('status') in ('complete', 'failed', 'stopped'):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            print('\nViewer closed; no signal was sent to the background job.')


def run_job(kind, arguments):
    os.environ['AASIST_PROGRESS_OWNER'] = str(os.getpid())
    phase('Starting background job')
    module = 'w2v_v3.workflow' if kind == 'v3' else 'w2v_aasist.' + ('full_workflow' if kind == 'full' else 'launch')
    command = [sys.executable, '-u', '-m', module]
    if kind == 'train':
        command.append('--run')
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    try:
        code = subprocess.call(command + arguments, cwd=ROOT)
    except BaseException:
        publish('Launcher interrupted; inspect log and child processes', status='failed', force=True)
        raise
    publish('Complete; results and download URL are in the log' if code == 0 else 'Failed; inspect log',
            status='complete' if code == 0 else 'failed', force=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', choices=('full', 'train', 'v3'))
    parser.add_argument('--state', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.run:
        return run_job(args.run, args.arguments)
    if args.arguments:
        parser.error('Unexpected viewer arguments')
    log = args.log
    pid = None
    if not args.state and not log:
        try:
            log = Path((ROOT/'exp'/'.latest_aasist_log').read_text(encoding='utf-8').strip())
            pid = int((ROOT/'exp'/'.latest_aasist_pid').read_text(encoding='utf-8').strip())
        except (OSError, ValueError):
            if log is None:
                parser.error('No launched job found. Start training, or pass --log /path/to/log')
    state = args.state or Path(str(log) + '.progress.json')
    if not args.once:
        print('Live progress (ETA for current phase). Ctrl+C closes this viewer only.')
        if log:
            print('LOG=' + str(log))
    watch(state, log, pid, args.once)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
