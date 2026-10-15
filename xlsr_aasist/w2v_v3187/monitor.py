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
TRAIN_PATTERN = re.compile(r'\[Train\] V3\.18\.7 epoch (\d+)/(\d+).*updates=(\d+)')


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
    value = entry.get('metrics', {})
    selection = entry.get('selection_metrics', {}).get('weighted_f1')
    epoch = entry['epoch']
    best = ' *' if entry.get('promoted') or entry.get('tag') == best_tag else ''
    lines = ['', f'Epoch {epoch} completed{best}',
             'Clean   Noisy   Weighted   Select80   TrainLoss',
             '  '.join([number(value.get(k), True) for k in ('clean_f1', 'noisy_f1', 'weighted_f1')]
                       + [number(selection, True), number(entry.get('training', {}).get('loss'))]),
             f'Full Dev epoch {epoch} (%)',
             'Group         FakeR  RealR     F1    AUC    EER  R@95F']
    for name, group in sorted(entry.get('dev', {}).get('groups', {}).items()):
        recalls = group.get('recall', [None, None])
        values = [*recalls, group.get('macro_f1'), group.get('auc'), group.get('eer'),
                  (group.get('matched_95') or {}).get('real_recall')]
        lines.append(f'{ascii_text(name):12s} ' + ' '.join(f'{number(v, True):>6s}' for v in values))
    return lines


def print_epoch(entry, state):
    print('\n'.join(epoch_lines(entry, state.get('best_tag'))), flush=True)


def panel_rows(cfg):
    from w2v_v3186.monitor import panel_rows as make_panel
    return make_panel(cfg)


def summarize(*args, **kwargs):
    from w2v_v3184.metrics import summarize as fn
    return fn(*args, **kwargs)


def measure(*args, **kwargs):
    from w2v_v3184.metrics import measure as fn
    return fn(*args, **kwargs)


def fit_state(*args, **kwargs):
    from w2v_v3184.metrics import fit_state as fn
    return fn(*args, **kwargs)


def render(run, history, steps):
    """Build CPU-only diagnostics, tolerating deliberately sparse probe panels."""
    from w2v_v317_monitor.render import chart
    from .common import atomic_json
    out = Path(run)/'diagnostics'; out.mkdir(exist_ok=True)
    body = chart('Full Dev', [(key, [(e['epoch'], 100*e['metrics'][key]) for e in history])
        for key in ('clean_f1', 'noisy_f1', 'weighted_f1')], percent=True)
    body += chart('Training objective', [(key, [(s['update'], s[key]) for s in steps if key in s])
        for key in ('loss', 'raw_ce')], log=True)
    body += chart('Selection Dev Weighted', [('selection', [(e['epoch'],
        100*e['selection_metrics']['weighted_f1']) for e in history if e.get('selection_metrics')])], percent=True)
    groups = sorted({name for e in history for name in e.get('dev', {}).get('groups', {})})
    for name in groups:
        body += chart(name+' Dev recall', [(label, [(e['epoch'], 100*e['dev']['groups'][name]['recall'][c])
            for e in history if name in e.get('dev', {}).get('groups', {})])
            for c, label in enumerate(('fake', 'real'))], percent=True)
    text = ('<!doctype html><meta charset="utf-8"><title>V3.18.7 diagnostics</title>'
            '<style>body{font:16px system-ui;max-width:1080px;margin:32px auto}section{padding:20px}'
            'svg{width:100%}pre{white-space:pre-wrap}</style><h1>V3.18.7 training and validation</h1>'
            '<p>Full fixed Dev; optional Train probes and mechanism panels are diagnostics only. '
            'Offline is used for simulation and phoneme references, not classification.</p>'+body)
    if history:
        text += '<pre>'+html.escape(history[-1].get('fit', {}).get('text', ''))+'</pre>'
    text += ('<script>document.querySelectorAll("[data-toggle]").forEach(b=>b.onclick=()=>{'
             'const g=[...b.closest("section").querySelectorAll("[data-series]")].find('
             'g=>g.dataset.series===b.dataset.toggle);if(!g)return;'
             'const hide=g.style.display!=="none";g.style.display=hide?"none":"";'
             'b.style.opacity=hide?"0.4":"1";});</script>')
    (out/'curves.html').write_text(text, encoding='utf-8')
    with (out/'epochs.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'phase', 'clean', 'noisy', 'weighted_full', 'weighted_select', 'loss', 'raw_ce'])
        for entry in history:
            writer.writerow([entry['epoch'], entry.get('phase'),
                *[entry['metrics'].get(k) for k in ('clean_f1', 'noisy_f1', 'weighted_f1')],
                entry.get('selection_metrics', {}).get('weighted_f1'),
                *[entry.get('training', {}).get(k) for k in ('loss', 'raw_ce')]])
    with (out/'groups.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'split', 'condition_language', 'n_fake', 'n_real', 'fake_recall',
                         'real_recall', 'F1', 'AP', 'AUC', 'EER', 'fake_CE', 'real_CE'])
        for entry in history:
            for split in ('train', 'dev', 'panel', 'scenes'):
                for name, group in (entry.get(split) or {}).get('groups', {}).items():
                    writer.writerow([entry['epoch'], split, name, *group['class_counts'], *group['recall'],
                        *[group.get(k) for k in ('macro_f1', 'ap', 'auc', 'eer')], *group['class_ce']])
    with (out/'fusion.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream); writer.writerow(['epoch', 'encoder_layer', 'weight'])
        for entry in history:
            fusion = entry.get('fusion', {}); weights = fusion.get('weights', [])
            for layer, weight in zip(fusion.get('layers', range(1, len(weights)+1)), weights):
                writer.writerow([entry['epoch'], layer, weight])
    atomic_json(out/'fitting.json', history[-1].get('fit', {}) if history else {})


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
        self.epoch = 0; self.epochs = 20; self.total = 0
        self.phase = 'Startup'; self.exit_code = None; self.error = ''

    def line(self, line):
        if line.startswith('V3187_RUN='):
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
        elif re.match(r'\w*(?:Error|Exception):', line):
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
        if total:
            prefix += '['+'#'*filled+'-'*(12-filled)+f'] {step}/{total} {100*fraction:.1f}%'
        else:
            prefix += phase
        fields = []
        for label, key in (('loss', 'loss'), ('CE', 'raw_ce')):
            if value.get(key) is not None:
                fields.append(label+'='+number(value[key]))
        pcl = value.get('pcl_loss', value.get('pcl'))
        if isinstance(pcl, (int, float)):
            fields.append('PCL='+number(pcl))
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
        self.last_progress = ''; self.last_update = 0.

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
        log = (ROOT/'exp'/'.latest_v3187_log').read_text().strip()
    try:
        return watch(log, run=args.run, once=args.once)
    except KeyboardInterrupt:
        print('\nViewer closed; training continues.', flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
