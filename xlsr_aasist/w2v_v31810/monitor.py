"""Append-only epoch metrics and one live progress line; ASCII terminal output."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATTERN = re.compile(r'\[Train\] V3\.18\.10 epoch (\d+)/(\d+).*updates=(\d+)')


def ascii_text(value):
    return ''.join(c if 32 <= ord(c) < 127 else ' ' for c in str(value))


def duration(value):
    value = max(0, int(value))
    return f'{value//3600:02d}:{value//60%60:02d}:{value%60:02d}'


def number(value, percent=False):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return 'NA'
    return f'{100*value:.2f}' if percent else f'{value:.5f}'


def epoch_lines(entry, best_tag=None):
    value=entry.get('metrics',{});best=' *' if entry.get('promoted') or entry.get('tag')==best_tag else ''
    lines=['',f'Epoch {entry["epoch"]} completed'+best,
        'Online  Regular  Heldout  SelectS  TrainLoss',
        '  '.join([number(value.get(k),True) for k in ('online_f1','regular_f1','heldout_f1','score')]+[number(entry.get('training',{}).get('loss'))]),
        'Selection only; fixed threshold 0.5', 'Group                         FakeR  RealR     F1']
    for name,group in sorted(entry.get('dev',{}).get('groups',{}).items()):
        recalls=group.get('recall',[None,None])
        lines.append(f'{ascii_text(name):28s} '+' '.join(f'{number(v,True):>6s}' for v in [*recalls,group.get('macro_f1')]))
    return lines


def print_epoch(entry,state):
    print('\n'.join(epoch_lines(entry,state.get('best_tag'))),flush=True)


def render(run,history,steps):
    from w2v_v317_monitor.render import chart
    out=Path(run)/'diagnostics';out.mkdir(exist_ok=True)
    body=chart('Source-isolated selection (fixed 0.5)',[(key,[(e['epoch'],100*e['metrics'][key]) for e in history])
        for key in ('online_f1','regular_f1','heldout_f1','score')],percent=True)
    body+=chart('Classification objective',[('loss',[(s['update'],s['loss']) for s in steps])],log=True)
    body+=chart('Meta transfer diagnostic',[('delta_B',[(s['update'],s['delta_B']) for s in steps if s.get('delta_B') is not None])])
    body+=chart('CPU data wait fraction',[('wait',[(e['epoch'],100*e['training'].get('data_wait_fraction',0.)) for e in history])],percent=True)
    (out/'curves.html').write_text('<!doctype html><meta charset="utf-8"><title>V3.18.10 diagnostics</title>'
        '<style>body{font:16px system-ui;max-width:1080px;margin:32px auto}svg{width:100%}</style>'
        '<h1>V3.18.10: main model, without target adaptation</h1>'
        '<p>Selection score differs from historical Dev Weighted. Locked audit is evaluated only after best selection.</p>'+body,encoding='utf-8')
    with (out/'epochs.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['epoch','online','regular_noisy','heldout_anlmdn','selection_score','loss'])
        for e in history:writer.writerow([e['epoch'],*[e['metrics'][k] for k in ('online_f1','regular_f1','heldout_f1','score')],e['training']['loss']])


class JsonFiles:
    def __init__(self):
        self.cache = {}

    def read(self, path, default):
        previous = self.cache.get(path)
        try:
            stat = path.stat(); stamp = (stat.st_mtime_ns, stat.st_size)
            if previous and previous[0] == stamp:
                return previous[1]
            value = json.loads(path.read_text(encoding='utf-8'))
            self.cache[path] = (stamp, value)
            return value
        except (OSError, ValueError):
            return previous[1] if previous else default


class Dashboard:
    def __init__(self, log, run=None):
        self.log = Path(log) if log else None
        self.run = Path(run) if run else None
        self.offset = 0; self.pending = b''; self.files = JsonFiles()
        self.history = []; self.links = {}; self.progress_value = {}
        self.epoch = 0; self.epochs = 15; self.total = 0
        self.phase = 'Startup'; self.exit_code = None; self.error = ''

    def line(self, line):
        if line.startswith('V31810_RUN='):
            self.run = Path(line.split('=', 1)[1].strip())
        match = TRAIN_PATTERN.search(line)
        if match:
            self.epoch, self.epochs, self.total = map(int, match.groups()); self.phase = 'Training'
        elif line.startswith('[Eval'):
            self.phase = 'Evaluation'
        elif line.startswith('[Submission]'):
            self.phase = 'Exporting best submission'
        elif line.startswith('[Report]'):
            self.phase = 'Delivering report'
        elif line.startswith('SUBMISSION_TEMP_URL='):
            self.links['submission_url'] = line.split('=', 1)[1].strip()
        elif line.startswith('REPORT_TEMP_URL='):
            self.links['report_url'] = line.split('=', 1)[1].strip()
        elif re.match(r'(?:\w+\.)*\w*(?:Error|Exception):', line):
            self.error = ascii_text(line)

    def poll(self):
        if self.log and self.log.is_file():
            if self.log.stat().st_size < self.offset:
                self.offset = 0; self.pending = b''
            with self.log.open('rb') as stream:
                stream.seek(self.offset)
                for block in iter(lambda: stream.read(65536), b''):
                    self.offset += len(block)
                    lines = (self.pending+block).split(b'\n'); self.pending = lines.pop()
                    for raw in lines:
                        self.line(raw.decode('utf-8', errors='replace').rstrip('\r'))
        if self.run:
            self.history = self.files.read(self.run/'training_history.json', [])
            self.links.update(self.files.read(self.run/'artifact_links.json', {}))
            self.epochs = self.files.read(self.run/'config.json', {}).get('epochs', self.epochs)
            self.progress_value = self.files.read(self.run/'progress.json', {})
            if self.progress_value:
                self.epoch = self.progress_value.get('epoch', self.epoch)
                self.total = self.progress_value.get('steps', self.total)
        marker = Path(str(self.log)+'.exit') if self.log else None
        if marker and marker.is_file() and (not self.log.exists() or self.log.stat().st_size <= self.offset):
            try:
                self.exit_code = int(marker.read_text().strip())
            except (OSError, ValueError):
                return
            if self.pending:
                self.line(self.pending.decode('utf-8', errors='replace')); self.pending = b''

    def progress(self):
        value = self.progress_value
        phase = ascii_text(value.get('phase') or value.get('state') or self.phase)
        if self.exit_code is not None:
            if self.run and (self.run/'paused.json').is_file():
                return 'Paused: resumable checkpoint saved'
            return 'Complete' if self.exit_code == 0 else 'Stopped: '+(self.error or 'see training log')
        epoch = int(value.get('epoch', self.epoch)); epochs = int(value.get('epochs', self.epochs))
        step = int(value.get('step', 0)); total = int(value.get('steps', self.total))
        fraction = min(1., step/max(1, total)); filled = int(12*fraction)
        prefix = f'Epoch {epoch}/{epochs} '
        if value.get('status') in ('complete','evaluating_final'):
            return phase
        if value.get('status')=='exporting':
            return phase+(f' {step}/{total} {100*fraction:.1f}%' if total else '')
        if total:
            prefix += '['+'#'*filled+'-'*(12-filled)+f'] {step}/{total} {100*fraction:.1f}%'
        else:
            prefix += phase
        fields = []
        for label, key in (('loss', 'loss'), ('CE', 'raw_ce')):
            if value.get(key) is not None:
                fields.append(label+'='+number(value[key]))
        eta = value.get('eta_seconds', value.get('eta'))
        if isinstance(eta, (int, float)):
            fields.append('ETA '+duration(eta))
        elif eta:
            fields.append('ETA '+ascii_text(eta))
        if phase.lower() not in ('training', 'train', 'joint', 'warm'):
            fields.append(phase)
        return prefix+(' | '+' | '.join(fields) if fields else '')


class HistoryPrinter:
    """CR overwrites only our single unfinished progress line, never Dev history."""
    def __init__(self, stream, tty=None, width=None):
        self.stream = stream; self.tty = stream.isatty() if tty is None else tty
        self.width = width; self.printed = set(); self.urls = {}; self.live_width = 0
        self.last_progress = ''; self.last_update = 0.; self.selection = None

    def end_live(self):
        if self.live_width:
            self.stream.write('\n'); self.live_width = 0

    def emit(self, state, *, final=False, once=False, now=None):
        now = time.monotonic() if now is None else now
        for entry in state.history:
            identity = entry.get('tag', entry['epoch'])
            if identity not in self.printed:
                self.end_live()
                self.stream.write('\n'.join(epoch_lines(entry))+'\n')
                self.printed.add(identity)
        value=state.progress_value
        best=value.get('best_tag')
        if best and value.get('status') in ('complete','paused') and self.selection!=best:
            origin='v31810_fresh_'+str(state.files.read(state.run/'config.json',{}).get('experiment','E2'))
            self.end_live()
            self.stream.write('BEST='+ascii_text(best)+' ORIGIN='+origin+'\n')
            self.selection=best
        for kind in ('submission', 'report'):
            url = state.links.get(kind+'_url')
            if url and url != self.urls.get(kind):
                self.end_live(); self.stream.write(kind.upper()+'_URL='+ascii_text(url)+'\n'); self.urls[kind] = url
        progress = ascii_text(state.progress())
        if progress != self.last_progress and (self.tty or final or once or now-self.last_update >= 30):
            if self.tty and not (final or once):
                width = self.width or shutil.get_terminal_size((120, 28)).columns
                progress = progress[:max(1, width-1)]
                padding = min(max(0, self.live_width-len(progress)), max(0, width-1-len(progress)))
                self.stream.write('\r'+progress+' '*padding)
                self.live_width = len(progress)
            else:
                self.end_live(); self.stream.write(progress+'\n')
            self.last_progress = progress; self.last_update = now
        self.stream.flush()


def watch(log=None, *, run=None, once=False, stream=None, sleep=time.sleep):
    state = Dashboard(log, run); printer = HistoryPrinter(stream or sys.stdout)
    while True:
        state.poll(); printer.emit(state, final=state.exit_code is not None, once=once)
        if once or state.exit_code is not None:
            printer.end_live(); return state.exit_code or 0
        sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', nargs='?'); parser.add_argument('--run')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    log = args.log
    if log is None and args.run is None:
        log = (ROOT/'exp'/'.latest_v31810_log').read_text().strip()
    try:
        return watch(log, run=args.run, once=args.once)
    except KeyboardInterrupt:
        print('\nViewer closed; training continues.', flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
