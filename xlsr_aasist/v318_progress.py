"""Read-only progress viewer for already-running V3.18 jobs (stdlib only)."""
import argparse
import json
from pathlib import Path
import re
import sys
import time


class Steps:
    def __init__(self):
        self.offset = 0
        self.pending = b''
        self.latest = None
        self.epoch = None
        self.seconds = 0.
        self.count = 0

    def poll(self, path):
        if not path.is_file():
            return
        if path.stat().st_size < self.offset:
            self.__init__()
        with path.open('rb') as f:
            f.seek(self.offset)
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.offset += len(chunk)
                lines = (self.pending + chunk).split(b'\n')
                self.pending = lines.pop()
                for line in lines:
                    try:
                        row = json.loads(line)
                        epoch, step = int(row['epoch']), int(row['step'])
                        seconds = float(row['compute_seconds']) + float(row['wait_seconds'])
                    except (ValueError, KeyError, TypeError):
                        continue
                    if epoch != self.epoch:
                        self.epoch, self.seconds, self.count = epoch, 0., 0
                    self.seconds += seconds
                    self.count += 1
                    self.latest = row


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def progress(reader, epoch, epochs, total):
    row = reader.latest
    if row is None or row['epoch'] != epoch:
        return f'Epoch {epoch}/{epochs} [--------------------] 0/{total} | waiting for first completed update'
    step = int(row['step'])
    fraction = min(1., step / max(1, total))
    filled = int(fraction * 20)
    speed = reader.seconds / max(1, reader.count)
    return (f'Epoch {epoch}/{epochs} [{"#"*filled}{"-"*(20-filled)}] '
            f'{step}/{total} {100*fraction:.1f}% | loss={float(row["loss"]):.5f} | '
            f'{speed:.2f}s/update | ETA {duration((total-step)*speed)}')


def watch(log):
    reader = Steps()
    run = None
    epoch, epochs, total = 0, 0, 0
    phase = 'initializing'
    offset, pending = 0, b''
    tty = sys.stdout.isatty()
    last_text, last_print = '', 0.

    def clear():
        if tty:
            print('\r\033[2K', end='', flush=True)

    print('LOG='+str(log), flush=True)
    print('Read-only viewer; Ctrl+C closes this viewer, not training.', flush=True)
    while True:
        if log.is_file():
            with log.open('rb') as f:
                f.seek(offset)
                chunk = f.read()
                offset += len(chunk)
            lines = (pending + chunk).split(b'\n')
            pending = lines.pop()
            for raw in lines:
                line = raw.decode('utf-8', errors='replace')
                if line.startswith('V318_RUN='):
                    run = Path(line.split('=', 1)[1].strip())
                m = re.search(r'\[Train\] V3\.18 epoch (\d+)/(\d+).*updates=(\d+)', line)
                if m:
                    epoch, epochs, total = map(int, m.groups())
                    phase = 'train'
                elif line.startswith('[Eval]'):
                    phase = 'eval'
                elif line.startswith(('[Complete]', '[Exit]', 'Traceback')):
                    phase = 'finished'
                clear()
                print(line, flush=True)
        if run is not None:
            reader.poll(run/'steps.jsonl')
        if Path(str(log)+'.exit').is_file():
            # Supervisor writes the log before the exit marker. Drain any final
            # bytes that arrived after this iteration's read before leaving.
            if log.stat().st_size > offset:
                continue
            clear()
            if pending:
                print(pending.decode('utf-8', errors='replace'), flush=True)
            return
        if phase == 'train':
            text = progress(reader, epoch, epochs, total)
        elif phase == 'eval':
            text = f'Epoch {epoch}/{epochs}: evaluating full Dev / Train probe / robustness panel'
        else:
            text = ''
        now = time.monotonic()
        if text and ((tty and text != last_text) or now-last_print >= 30):
            if tty:
                clear()
                print(text, end='', flush=True)
            elif now-last_print >= 30:
                print(text, flush=True)
            last_text, last_print = text, now
        time.sleep(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', nargs='?')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    log = Path(args.log) if args.log else Path((root/'exp'/'.latest_v318_log').read_text().strip())
    try:
        watch(log)
    except KeyboardInterrupt:
        print('\nViewer closed; training was not interrupted.', flush=True)


if __name__ == '__main__':
    main()
