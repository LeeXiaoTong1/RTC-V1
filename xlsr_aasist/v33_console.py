"""Read-only V3.3 console: retain validation results above one live progress line.

This viewer deliberately lives outside the training package. Updating it does not
change the code fingerprints of an already-running training job or its resume.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

from live_progress import alive, log_state, read_state, render

ROOT = Path(__file__).resolve().parent
TAG = re.compile(r'^(?:baseline|epoch_\d+(?:_step_\d+)?)$')
ANSI = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]')
DEV = re.compile(r'^Dev (baseline|epoch_\d+(?:_step_\d+)?)\s+(.*)$')
RECALL = re.compile(r'^(offline/en|online/en|seen/en|heldout/en) recall \[fake,real\]=(.*)$')
GROUPS = ('offline/en', 'online/en', 'seen/en', 'heldout/en')
NOISY_PROGRESS = re.compile(r'^(?:STEP \d+/\d+|(?:(?:baseline|epoch_\d+(?:_step_\d+)?) )?Dev \w+: \d+/\d+|Train source headers: \d+/\d+|Full noisy sources: \d+/\d+)')


def safe_text(value):
    """A log is data: never let escape/control sequences operate the terminal."""
    value = ANSI.sub('', str(value))
    return ''.join(c if c >= ' ' and not '\x7f' <= c <= '\x9f' else ' ' for c in value)


class LogTail:
    """Incremental complete-line reader, including rotation and truncation."""
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.offset = 0
        self.pending = b''
        self.identity = None

    def read(self, limit=2 * 1024 * 1024):
        if self.path is None:
            return []
        try:
            with self.path.open('rb') as stream:
                stat = self.path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.offset, self.pending = 0, b''
                self.identity = identity
                stream.seek(self.offset)
                chunk = stream.read(limit)
                self.offset += len(chunk)
        except OSError:
            return []
        pieces = (self.pending + chunk).split(b'\n')
        self.pending = pieces.pop()
        # An accidental enormous unterminated line must not grow indefinitely.
        if len(self.pending) > 4 * 1024 * 1024:
            self.pending = self.pending[-4 * 1024 * 1024:]
        return [line.decode('utf-8', errors='replace').rstrip('\r') for line in pieces]


def number(value, percent=True):
    try:
        value = float(value)
        return f'{value * (100 if percent else 1):.3f}' if math.isfinite(value) else 'n/a'
    except (ValueError, TypeError):
        return 'n/a'


def format_record(record):
    """Stable, complete validation block; absent data is never invented."""
    tag = safe_text(record.get('tag', 'unknown'))
    dev, decision = record.get('dev', {}), record.get('decision', {})
    title = 'Starting checkpoint / baseline' if tag == 'baseline' else tag
    if '_step_' in tag:
        title += ' (intermediate Dev)'
    elif tag.startswith('epoch_'):
        title += ' (epoch complete)'
    lines = [f'\n[Dev] {safe_text(record.get("arm",""))} {title}  phase={safe_text(record.get("phase", "unknown"))}',
             '  ' + '  '.join(name + '=' + number(dev.get(key)) for name, key in
                 (('Clean', 'clean_f1'), ('Seen', 'seen_f1'), ('Heldout', 'heldout_f1'),
                  ('Noisy', 'noisy_f1'), ('Weighted', 'weighted_f1')))]
    for group in GROUPS:
        recall = dev.get('groups', {}).get(group, {}).get('recall', [])
        fake, real = recall if isinstance(recall, (list, tuple)) and len(recall) == 2 else (None, None)
        lines.append(f'  {group:12s} fake={number(fake)}%  real={number(real)}%')
    promoted = 'best_safe' in decision.get('save', [])
    action = safe_text(decision.get('action', 'unknown'))
    lines.append(f'  Selection: promoted={promoted}  action={action}')
    warnings = decision.get('warnings', [])
    if warnings:
        lines.append('  Selection warnings: ' + ', '.join(safe_text(x) for x in warnings))
    rates = record.get('learning_rates', {})
    if rates:
        lines.append('  LR: ' + ', '.join(safe_text(k) + '=' + safe_text(v) for k, v in rates.items()))
    return lines


class Events:
    """Merge authoritative saved Dev records and backward-compatible log lines."""
    def __init__(self, run=None):
        self.run = Path(run) if run else None
        self.seen = set()
        self.info_seen = set()
        self.last_phase = None
        self.report_signature = None
        self.committed_tags = set()
        self.pending_dev = None
        self.suppress_recalls = False
        self.report_clock = -math.inf

    def _key(self, tag):
        return (str(self.run), tag)

    def record(self, value):
        if not isinstance(value, dict) or not TAG.fullmatch(str(value.get('tag', ''))):
            return []
        tag = value['tag']
        if not isinstance(value.get('dev'), dict) or self._key(tag) in self.seen:
            return []
        self.seen.add(self._key(tag))
        return format_record(value)

    def saved(self, tag):
        if not self.run or not TAG.fullmatch(tag):
            return []
        value = read_state(self.run / (tag + '.json'))
        return self.record(value) if value else []

    def phase(self, label):
        label = safe_text(label)
        if not label or label == self.last_phase:
            return []
        self.last_phase = label
        return ['\n[Phase] ' + label]

    def report(self, now=None, force=False):
        """Only stat one small report every two seconds; never scan score files."""
        now = time.monotonic() if now is None else now
        if not self.run or (not force and now - self.report_clock < 2):
            return []
        self.report_clock = now
        path = self.run / 'report.md'
        try:
            stat = path.stat()
            signature = (str(path), stat.st_mtime_ns, stat.st_size)
            if signature == self.report_signature:
                return []
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            return []
        output = []
        missing = False
        self.committed_tags = set()
        for line in text.splitlines():
            fields = line.split('|')
            tag = fields[1].strip() if len(fields) > 2 else ''
            if TAG.fullmatch(tag):
                self.committed_tags.add(tag)
                output.extend(self.saved(tag))
                missing = missing or self._key(tag) not in self.seen
        # Retry a partial/inconsistent copy rather than permanently losing a row.
        if not missing:
            self.report_signature = signature
        return output

    def _flush_dev(self):
        value, self.pending_dev = self.pending_dev, None
        return self.record(value) if value else []

    def consume(self, raw):
        line = safe_text(raw).strip()
        if not line:
            return []
        if line.startswith(('V32_RUN=', 'V33_ACTIVE_RUN=')):
            value = line.split('=', 1)[1].strip()
            if value and self.run != Path(value):
                self.run = Path(value)
                self.report_signature = None
                self.committed_tags = set()
            return self.report(force=True)
        if line.startswith(('V32_EVENT=', 'V33_EVENT=')):
            try:
                event = json.loads(line.split('=', 1)[1])
            except ValueError:
                return []
            if not isinstance(event, dict):
                return []
            if event.get('kind') == 'phase':
                return self._flush_dev() + self.phase(event.get('label', ''))
            if event.get('kind') == 'validation':
                # New training emits this only after report/checkpoint commit.
                # The report is still authoritative if a copied log is ahead.
                return self._flush_dev() + self.report(force=True)
            return []
        match = DEV.match(line)
        if match:
            output = self._flush_dev()
            tag, tail = match.groups()
            if self.run and self.run.is_dir():
                # Old trainers print Dev before committing last.pt/report.md.
                # Do not announce a promoted checkpoint while it is still saving.
                self.suppress_recalls = True
                return output + self.report(force=True)
            output += self.saved(tag)
            self.suppress_recalls = self._key(tag) in self.seen
            if not self.suppress_recalls:
                try:
                    scores = {key: float(value) / 100 for key, value in
                              re.findall(r'\b(Clean|Seen|Heldout|Noisy|Weighted)=([0-9.eE+-]+)', tail)}
                except ValueError:
                    return output + ['Unparseable Dev log row; see saved report.json or report.md.']
                dev = {key.lower() + '_f1': value for key, value in scores.items()}
                if 'Noisy' not in scores and 'Seen' in scores and 'Heldout' in scores:
                    dev['noisy_f1'] = (scores['Seen'] + scores['Heldout']) / 2
                dev['groups'] = {}
                action = re.search(r'\baction=(\S+)', tail)
                self.pending_dev = {'tag': tag, 'phase': 'joint', 'dev': dev,
                                    'decision': {'save': ['best_safe'] if 'promoted=True' in tail else [],
                                                 'action': action.group(1) if action else 'unknown'}}
            return output
        match = RECALL.match(line)
        if match:
            if self.pending_dev:
                try:
                    recall = json.loads(match.group(2))
                except ValueError:
                    return []
                self.pending_dev['dev']['groups'][match.group(1)] = {'recall': recall}
                if len(self.pending_dev['dev']['groups']) == len(GROUPS):
                    return self._flush_dev()
            return []
        if line.startswith('Selection warnings:') and self.suppress_recalls:
            return []
        if NOISY_PROGRESS.match(line):
            return []
        if line.startswith('BASELINE_SAVED='):
            return self.report(force=True) if self.run and self.run.is_dir() else self.saved('baseline')
        # Keep errors, launch/configuration notices, restoration and download URLs.
        # Dedup on log rotation prevents reprinting the same diagnostics/results.
        if line in self.info_seen:
            return []
        self.info_seen.add(line)
        return self._flush_dev() + [line]


class Display:
    def __init__(self, stream):
        self.stream, self.tty = stream, stream.isatty()
        self.previous_width = 0

    def clear_line(self):
        if self.tty and self.previous_width:
            width = shutil.get_terminal_size((120, 24)).columns
            self.stream.write('\r' + ' ' * min(self.previous_width, max(1, width - 1)) + '\r')
            self.previous_width = 0

    def messages(self, lines):
        if not lines:
            return
        self.clear_line()
        for line in lines:
            # Only our intentional leading blank line may create a new line.
            self.stream.write(('\n' if line.startswith('\n') else '') + safe_text(line.lstrip('\n')) + '\n')
        self.stream.flush()

    def progress(self, value):
        width = shutil.get_terminal_size((120, 24)).columns
        line = render(value, width)
        self.clear_line()
        if self.tty:
            self.stream.write(line)
            self.previous_width = len(line)
        else:
            self.stream.write(line + '\n')
        self.stream.flush()


def watch(log, state, run=None, once=False, interval=1., stream=None, version='V3.3'):
    output = Display(stream or sys.stdout)
    if not output.tty and not once:
        raise SystemExit('Open the viewer in a terminal. For pipes/files use --once.')
    events, tail = Events(run), LogTail(log)
    output.messages([version + ' console: completed Dev results stay above the live line.',
                     'ETA is for the current phase. Ctrl+C closes only this viewer.',
                     'LOG=' + str(log) if log else 'STATE=' + str(state)])
    try:
        while True:
            # Drain retained historical log in bounded chunks, with complete lines.
            while True:
                before = tail.offset
                lines = tail.read()
                for line in lines:
                    output.messages(events.consume(line))
                if tail.offset - before < 2 * 1024 * 1024:
                    break
            output.messages(events.report(force=once))
            value = read_state(state) or (log_state(log) if log else None)
            value = value or {'label': 'Waiting for the background job to publish progress',
                              'updated_at': time.time()}
            if value.get('status', 'running') == 'running' and alive(value.get('owner_pid')) is False:
                value = dict(value, status='stopped', label='Launcher stopped; inspect saved results and log', total=None)
            output.messages(events.phase(value.get('label', '')))
            output.progress(value)
            if once or value.get('status') in ('complete', 'failed', 'stopped'):
                output.messages(events._flush_dev())
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        output.clear_line()
        if output.tty:
            output.messages(['Viewer closed; no signal was sent to the background training job.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=1.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < .1:
        parser.error('--interval must be at least 0.1 seconds')
    if not args.log and not args.state:
        try:
            args.log = Path((ROOT / 'exp' / '.latest_v33_log').read_text(encoding='utf-8').strip())
        except OSError:
            parser.error('No V3.3 job found; pass --log /path/to/log or start training.')
    state = args.state or Path(str(args.log) + '.progress.json')
    watch(args.log, state, args.run_dir, args.once, args.interval)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
