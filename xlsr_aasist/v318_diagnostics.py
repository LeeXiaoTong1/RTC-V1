"""Save live V3.18 diagnostics from existing logs, without loading a model.

This file deliberately lives outside the fingerprinted training package. It
only reads config/history/steps and writes diagnostics/live; no training hook,
checkpoint, RNG, GPU, data-loader or optimizer interaction is involved.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


SCALARS = ('loss', 'raw_ce_sum', 'views', 'grad_norm', 'max_example_ce',
           'risk_coefficient', 'compute_seconds', 'wait_seconds', 'unique_original_sources')
COLORS = ('#2563eb', '#dc2626', '#059669', '#9333ea', '#d97706', '#0891b2', '#475569', '#db2777')
NOTES = (
    'Train metrics are fixed probes, not full Train. Dev metrics use the full fixed manifest; '
    'checkpoint selection uses the separate 80% source partition. Scores here are raw, uncalibrated. '
    'Training objective includes smoothing/risk and domain weights; raw CE is per view. '
    'Neither training-mode curve is directly comparable to eval-mode probe/Dev CE. '
    'Noisy probe A/B and Dev Seen/Heldout have different distributions. '
    'Gradient norm is global before clipping, not a per-layer LoRA update measurement. '
    'Graphs average at most 100 updates within each epoch; CSV retains every logged update. '
    'AUC/EER and both class recalls must be considered together; these curves do not prove a cause.'
)


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def atomic_text(path, text):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class StepLog:
    """Incremental reader; exclude partial records and superseded retry suffixes."""
    def __init__(self):
        self.offset = 0
        self.pending = b''
        self.identity = None
        self.rows = {}
        self.last = 0

    def poll(self, path):
        if not path.is_file():
            return False
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity != identity or stat.st_size < self.offset:
            self.__init__()
            self.identity = identity
        changed = False
        with path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                self.offset += len(chunk)
                lines = (self.pending + chunk).split(b'\n')
                self.pending = lines.pop()
                for line in lines:
                    row = json.loads(line)
                    cursor = int(row['update'])
                    if cursor < 1 or int(row['epoch']) < 1 or int(row['step']) < 1:
                        raise ValueError('Invalid step cursor in steps.jsonl')
                    if cursor <= self.last:
                        self.rows = {k: v for k, v in self.rows.items() if k < cursor}
                    compact = {k: row[k] for k in ('update', 'epoch', 'step', *SCALARS) if k in row}
                    compact.update(lrs=row.get('lrs', {}), cells=row.get('cells', {}))
                    self.rows[cursor] = compact
                    self.last = cursor
                    changed = True
        return changed

    def values(self):
        return [self.rows[k] for k in sorted(self.rows)]


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def blocks(rows, width=100):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row['epoch'], (row['step'] - 1) // width)].append(row)
    result = []
    for _, items in sorted(buckets.items()):
        out = dict(update=items[-1]['update'], epoch=items[-1]['epoch'], lrs=items[-1]['lrs'])
        for key in ('loss', 'grad_norm', 'risk_coefficient', 'compute_seconds', 'wait_seconds'):
            out[key] = mean(r.get(key) for r in items)
        out['max_example_ce'] = max((r['max_example_ce'] for r in items if 'max_example_ce' in r), default=None)
        out['raw_ce'] = ratio(sum(r.get('raw_ce_sum', 0) for r in items), sum(r.get('views', 0) for r in items))
        out['cells'] = {}
        for cell in sorted({k for r in items for k in r['cells']}):
            parts = [r['cells'][cell] for r in items if cell in r['cells']]
            out['cells'][cell] = ratio(sum(p.get('raw_ce_sum', 0) for p in parts), sum(p.get('views', 0) for p in parts))
        result.append(out)
    return result


def chart(title, series, log=False, percent=False, x_label='Epoch'):
    cleaned = []
    for label, points in series:
        valid = [(float(x), float(y)) for x, y in points
                 if y is not None and math.isfinite(float(y)) and (not log or y > 0)]
        if valid:
            cleaned.append((label, valid))
    title = html.escape(title)
    if not cleaned:
        return f'<section><h2>{title}</h2><p>Waiting for recorded values.</p></section>'
    transform = math.log10 if log else lambda y: y
    xy = [(x, transform(y)) for _, points in cleaned for x, y in points]
    x0, x1 = min(x for x, _ in xy), max(x for x, _ in xy)
    y0, y1 = min(y for _, y in xy), max(y for _, y in xy)
    if x0 == x1:
        x0, x1 = x0 - .5, x1 + .5
    pad = max(y1 - y0, 1 if percent else .02 if log else max(abs(y1), 1e-3) * .2) * .1
    y0, y1 = y0 - pad, y1 + pad
    if percent:
        y0, y1 = max(0, y0), min(100, y1)
    def position(x, y):
        return 70 + 760 * (x - x0) / (x1 - x0), 240 - 210 * (transform(y) - y0) / (y1 - y0)
    svg = [f'<svg viewBox="0 0 865 300" role="img" aria-label="{title}">']
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        py = 240 - 210 * i / 4
        value = 10 ** y if log else y
        svg.append(f'<line x1="70" x2="830" y1="{py}" y2="{py}" stroke="#e2e8f0"/>'
                   f'<text x="62" y="{py+4}" text-anchor="end">{value:.4g}</text>')
        x = x0 + (x1 - x0) * i / 4
        svg.append(f'<text x="{70+760*i/4}" y="263" text-anchor="middle">{x:.5g}</text>')
    legend = []
    for index, (label, points) in enumerate(cleaned):
        color = COLORS[index % len(COLORS)]
        coords = ' '.join(f'{px:.2f},{py:.2f}' for px, py in (position(x, y) for x, y in points))
        svg.append(f'<g data-series="{index}"><polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            px, py = position(x, y)
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.7" fill="{color}">'
                       f'<title>{html.escape(label)}: x={x:g}, y={y:.7g}</title></circle>')
        svg.append('</g>')
        legend.append(f'<button data-toggle="{index}" style="color:{color}">{html.escape(label)}</button>')
    svg.append(f'<text x="450" y="289" text-anchor="middle">{html.escape(x_label)}</text></svg>')
    return f'<section><h2>{title}</h2>' + ''.join(legend + svg) + '</section>'


def metric(epoch, split, group, key, index=None):
    value = epoch.get(split, {}).get('groups', {}).get(group, {}).get(key)
    return value[index] if value is not None and index is not None else value


def write_csv(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def render(run, reader, history, cfg, out, status='running_or_unknown'):
    rows, epochs = reader.values(), history
    binned = blocks(rows)
    committed = max((int(e['tag'].rsplit('_', 1)[1]) * e['epoch'] for e in epochs), default=0)
    lr_names = sorted({key for r in rows for key in r['lrs']})
    loss_rows = []
    cell_rows = []
    for r in rows:
        loss_rows.append(dict(r, raw_ce=ratio(r.get('raw_ce_sum', 0), r.get('views', 0)),
                             checkpoint_committed=r['update'] <= committed,
                             **{name + '_lr': r['lrs'].get(name) for name in lr_names}))
        for cell, values in r['cells'].items():
            cell_rows.append(dict(update=r['update'], epoch=r['epoch'], step=r['step'], cell=cell,
                                  **values, raw_ce=ratio(values.get('raw_ce_sum', 0), values.get('views', 0)),
                                  recall=ratio(values.get('correct', 0), values.get('views', 0))))
    write_csv(out/'loss_steps.csv', ['update', 'epoch', 'step', *SCALARS, 'raw_ce',
              *[name+'_lr' for name in lr_names], 'checkpoint_committed'], loss_rows)
    write_csv(out/'training_cells.csv', ['update', 'epoch', 'step', 'cell', 'objective', 'raw_ce_sum',
              'views', 'correct', 'raw_ce', 'recall'], cell_rows)
    epoch_rows, group_rows = [], []
    for e in epochs:
        item = dict(epoch=e['epoch'], tag=e['tag'], phase=e.get('phase'), promoted=e.get('promoted'),
                    **{k: e.get('metrics', {}).get(k) for k in ('clean_f1', 'noisy_f1', 'weighted_f1')},
                    weighted_select=e.get('selection_metrics', {}).get('weighted_f1'))
        item.update({k: e.get('training', {}).get(k) for k in ('loss', 'raw_ce', 'compute_seconds', 'wait_seconds')})
        epoch_rows.append(item)
        for split in ('train', 'dev', 'panel'):
            for group, m in e.get(split, {}).get('groups', {}).items():
                counts, recall, ce = (m.get(k, [None, None]) for k in ('class_counts', 'recall', 'class_ce'))
                group_rows.append(dict(epoch=e['epoch'], tag=e['tag'], split=split, group=group,
                    n_fake=counts[0], n_real=counts[1], fake_recall=recall[0], real_recall=recall[1],
                    fake_ce=ce[0], real_ce=ce[1],
                    **{k: m.get(k) for k in ('macro_f1', 'ap', 'auc', 'eer', 'balanced_ce')}))
    write_csv(out/'epochs.csv', ['epoch', 'tag', 'phase', 'promoted', 'clean_f1', 'noisy_f1',
              'weighted_f1', 'weighted_select', 'loss', 'raw_ce', 'compute_seconds', 'wait_seconds'], epoch_rows)
    write_csv(out/'groups.csv', ['epoch', 'tag', 'split', 'group', 'n_fake', 'n_real', 'fake_recall',
              'real_recall', 'macro_f1', 'ap', 'auc', 'eer', 'balanced_ce', 'fake_ce', 'real_ce'], group_rows)
    series = lambda key: [(r['update'], r.get(key)) for r in binned]
    charts = [chart('Training objective and raw CE (log scale)', [(k, series(k)) for k in ('loss', 'raw_ce')], log=True, x_label='Update'),
              chart('Global pre-clip gradient norm / worst example CE', [(k, series(k)) for k in ('grad_norm', 'max_example_ce')], log=True, x_label='Update'),
              chart('Learning rates (log scale; disabled zero rates omitted)',
                    [(k, [(r['update'], r['lrs'].get(k)) for r in binned]) for k in lr_names], log=True, x_label='Update'),
              chart('Risk coefficient', [('risk', series('risk_coefficient'))], x_label='Update'),
              chart('Compute and data wait (seconds/update)', [(k, series(k)) for k in ('compute_seconds', 'wait_seconds')], x_label='Update'),
              chart('Raw Dev F1 (%)', [(k, [(r['epoch'], None if r.get(k) is None else 100*r[k]) for r in epoch_rows])
                    for k in ('clean_f1', 'noisy_f1', 'weighted_f1', 'weighted_select')], percent=True)]
    for lang in ('en', 'zh'):
        curves = []
        for split in ('train', 'dev'):
            for c, label in enumerate(('fake', 'real')):
                curves.append((split+'/'+label, [(e['epoch'], metric(e, split, 'online/'+lang, 'class_ce', c)) for e in epochs]))
        charts.append(chart(lang.upper()+' Online: fixed Train probe / full Dev class CE', curves))
        cells = sorted({k for r in binned for k in r['cells'] if '/'+lang+'/' in k})
        charts.append(chart(lang.upper()+' training-mode class CE (changing augmented samples)',
                      [(k, [(r['update'], r['cells'].get(k)) for r in binned]) for k in cells], x_label='Update'))
        for split in ('train', 'dev'):
            groups = sorted({g for e in epochs for g in e.get(split, {}).get('groups', {}) if g.endswith('/'+lang)})
            scope = 'fixed Train probe' if split == 'train' else 'full Dev'
            for key, title, classes in (('recall', 'Recall (%)', True), ('class_ce', 'class CE', True), ('auc', 'AUC (%)', False), ('eer', 'EER (%)', False)):
                curves = []
                for group in groups:
                    for c in ((0, 1) if classes else (None,)):
                        points = [(e['epoch'], metric(e, split, group, key, c)) for e in epochs]
                        percentage = key != 'class_ce'
                        if percentage:
                            points = [(x, None if y is None else 100*y) for x, y in points]
                        curves.append((group + (('/fake', '/real')[c] if c is not None else ''), points))
                charts.append(chart(lang.upper()+' '+scope+' '+title, curves, percent=key != 'class_ce'))
    updated = datetime.now(timezone.utc).isoformat()
    latest = rows[-1] if rows else None
    report = dict(version='3.18', run=str(run), updated_utc=updated, status=status,
                  committed_update=committed, logged_update=reader.last, recorded_updates=len(rows),
                  completed_epochs=len(epochs), latest_step=latest,
                  fit=epochs[-1].get('fit', {}) if epochs else {}, notes=NOTES,
                  warm_epochs=cfg.get('warm_epochs'), planned_epochs=cfg.get('epochs'),
                  config_sha256=hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),
                  saved_dev_scores=sorted(p.name for p in run.glob('dev_scores_*.npz')),
                  unmeasured=['per-layer LoRA update magnitude', 'post-processing SNR', 'calibrated selection metrics'])
    summary = (f"RUN={run}\nSTATUS={status}\nUPDATED_UTC={updated}\nCOMPLETED_EPOCHS={len(epochs)}\n"
               f"LOGGED_UPDATE={reader.last}\nCOMMITTED_UPDATE={committed}\n"
               f"FIT={report['fit'].get('text', 'Waiting for completed epoch evaluation.')}\n\n{NOTES}\n")
    atomic_text(out/'summary.txt', summary)
    atomic_text(out/'observations.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    intro = f'<h1>V3.18 training diagnostics</h1><p>{html.escape(str(run))}</p><p>{html.escape(updated)} | {html.escape(status)}</p>'
    intro += f'<p>Logged update {reader.last}; checkpoint committed through {committed}. Warmup epochs: {cfg.get("warm_epochs", "?")}.</p>'
    intro += '<p>'+html.escape(NOTES)+'</p><p><strong>'+html.escape(report['fit'].get('text', 'Waiting for evaluation.'))+'</strong></p>'
    intro += '<p>'+' | '.join(f'<a href="{n}">{n}</a>' for n in ('loss_steps.csv', 'training_cells.csv', 'epochs.csv', 'groups.csv', 'observations.json'))+'</p>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V3.18 live diagnostics</title><style>
body{font:15px system-ui,sans-serif;color:#172033;background:#f1f5f9;max-width:1500px;margin:auto;padding:24px}
header,section{background:white;padding:20px;border-radius:10px;margin-bottom:16px}p{line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}h2{font-size:17px}
svg{width:100%;font-size:12px}button{background:white;border:1px solid #cbd5e1;border-radius:4px;margin:3px;padding:4px;cursor:pointer}
@media(max-width:650px){body{padding:8px}.grid{grid-template-columns:1fr}}
</style>'''
    page += '<header>'+intro+'</header><main class="grid">'+''.join(charts)+'</main>'
    page += '''<script>document.querySelectorAll('[data-toggle]').forEach(b=>b.onclick=()=>{
const g=b.closest('section').querySelector('g[data-series="'+b.dataset.toggle+'"]');
const hide=g.style.display!=='none';g.style.display=hide?'none':'';b.style.opacity=hide?'.4':'1';});</script></html>'''
    atomic_text(out/'curves.html', page)
    return report


def matching_log(root, run):
    run = run.resolve()
    pointer = root/'exp'/'.latest_v318_log'
    if not pointer.is_file():
        return None
    log = Path(pointer.read_text(encoding='utf-8').strip()).expanduser()
    if not log.is_absolute():
        log = root/log
    if not log.is_file():
        return None
    with log.open(encoding='utf-8', errors='replace') as stream:
        prefix = stream.read(131072)
    for value in re.findall(r'^V318_RUN=(.+)$', prefix, re.MULTILINE):
        if Path(value.strip()).resolve() == run:
            return log.resolve()
    return None


def finished(run, log):
    done = read_json(run/'completed.json', {})
    if done.get('status') == 'complete':
        return 'complete'
    if log is not None and Path(str(log)+'.exit').is_file():
        return 'training_exit_' + Path(str(log)+'.exit').read_text().strip()
    return None


@contextmanager
def exclusive_writer(out):
    # OS-held lock disappears on process death; never trust or kill a stale PID.
    stream = (out/'monitor.lock').open('a+b')
    try:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        stream.close()
        raise RuntimeError('A diagnostic writer already owns this run; see monitor.log') from exc
    try:
        yield
    finally:
        stream.close()


def follow(run, interval, watch=False, root=None):
    run = run.resolve()
    cfg = read_json(run/'config.json', {})
    if cfg.get('version') != '3.18':
        raise ValueError('Expected a V3.18 run with config.json')
    out = run/'diagnostics'/'live'
    out.mkdir(parents=True, exist_ok=True)
    root = root or Path(__file__).resolve().parent
    log = matching_log(root, run)
    reader = StepLog()
    previous = None
    with exclusive_writer(out):
        atomic_text(out/'monitor.pid', str(os.getpid())+'\n')
        while True:
            status = finished(run, log)
            reader.poll(run/'steps.jsonl')
            history = read_json(run/'training_history.json', [])
            if not isinstance(history, list):
                raise ValueError('Invalid training history')
            state = (reader.offset, len(history), status)
            if previous != state:
                result = render(run, reader, history, cfg, out, status or 'running_or_unknown')
                print(f'DIAGNOSTICS={out / "curves.html"}; update={reader.last}; epochs={len(history)}; status={result["status"]}', flush=True)
                previous = state
            if not watch or status:
                return
            time.sleep(interval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', help='Default: pin the current .latest_v318_run once at startup')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--watch', action='store_true', help='Refresh until training completes/exits; Ctrl+C stops diagnostics only')
    mode.add_argument('--background', action='store_true', help='Linux: detach diagnostics and return to the same terminal')
    p.add_argument('--interval', type=float, default=60)
    args = p.parse_args()
    if not math.isfinite(args.interval) or args.interval < 5:
        p.error('--interval must be finite and at least 5 seconds')
    root = Path(__file__).resolve().parent
    raw = args.run or (root/'exp'/'.latest_v318_run').read_text(encoding='utf-8').strip()
    run = Path(raw).expanduser().resolve()
    if read_json(run/'config.json', {}).get('version') != '3.18':
        p.error('Not a V3.18 run: '+str(run))
    if args.background:
        if os.name != 'posix':
            p.error('--background requires Linux; use one-shot mode on other platforms')
        out = run/'diagnostics'/'live'
        out.mkdir(parents=True, exist_ok=True)
        with (out/'monitor.log').open('ab', buffering=0) as stream:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--run', str(run), '--watch', '--interval', str(args.interval)], cwd=root,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        time.sleep(.3)
        if child.poll() not in (None, 0):
            raise SystemExit('Diagnostics did not start; inspect '+str(out/'monitor.log'))
        print(f'MONITOR_PID={child.pid}\nMONITOR_LOG={out / "monitor.log"}\nDIAGNOSTICS={out / "curves.html"}', flush=True)
        return
    try:
        follow(run, args.interval, args.watch, root)
    except KeyboardInterrupt:
        print('\nDiagnostics stopped; training was not signaled.', flush=True)


if __name__ == '__main__':
    main()
