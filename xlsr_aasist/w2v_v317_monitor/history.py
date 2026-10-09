"""Read committed scalar results and retry-aware update logs, without a GPU."""
import json
from pathlib import Path
import re
import numpy as np
from w2v_v39.common import read_json
from .metrics import summarize, fit_state, table_lines


def read_steps(path):
    rows = {}
    previous = -1
    path = Path(path)
    if not path.exists(): return []
    # A killed/retried epoch can repeat cursors. Discard its stale suffix when
    # the new attempt starts, rather than joining observations from two attempts.
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'): break  # Writer may be mid-record.
            value = json.loads(line)
            cursor = int(value['cursor'])
            if cursor < 1: raise ValueError('Invalid logged update cursor')
            if cursor <= previous:
                rows = {k:v for k,v in rows.items() if k < cursor}
            rows[cursor] = value; previous = cursor
    return [rows[k] for k in sorted(rows)]


def score_summary(path, rows):
    with np.load(path, allow_pickle=False) as data:
        return summarize(rows, data['logits'])


def collect(run):
    run = Path(run).resolve()
    cfg = read_json(run/'config.json')
    if cfg.get('version') != '3.17': raise ValueError('Expected a V3.17 run')
    history = read_json(run/'training_history.json') if (run/'training_history.json').exists() else []
    result = dict(run=str(run), epochs=[], steps=read_steps(run/'training_steps.jsonl'),
                  warnings=[], completed=(run/'completed.json').exists())
    row_sets = {}
    for split, name in (('train','train_probe_rows.json'),('dev','dev_rows.json')):
        if (run/name).exists(): row_sets[split] = read_json(run/name)
    for item in history:
        if not item.get('committed'): continue
        match = re.fullmatch(r'epoch_(\d+)_step_(\d+)', item['tag'])
        if not match: raise ValueError('Invalid V3.17 epoch tag')
        entry = dict(tag=item['tag'], epoch=int(match[1]), cursor=item['cursor'],
                     metrics=item['metrics'], segment_training=item.get('segment_training', {}))
        for split, prefix in (('train','train_probe_scores_'), ('dev','dev_scores_')):
            path = run/(prefix+item['tag']+'.npz')
            if split in row_sets and path.exists():
                entry[split] = score_summary(path, row_sets[split])
            else:
                result['warnings'].append(f'{item["tag"]}: {split} score/row file missing; no invented metrics')
        panel = run/'diagnostics'/('panel_'+item['tag']+'.json')
        if panel.exists(): entry['panel'] = read_json(panel)
        failure=run/'diagnostics'/('failure_'+item['tag']+'.log')
        if failure.exists() and not entry.get('panel'):
            result['warnings'].append(f'{item["tag"]}: full Train diagnostic failed; inspect {failure.name}')
        result['epochs'].append(entry)
    fit_epochs=result['epochs'];fit_scope='original fixed Train Online probe'
    if fit_epochs and fit_epochs[-1].get('panel'):
        signature=fit_epochs[-1]['panel']['panel_signature']
        fit_epochs=[dict(e,train=e['panel']['metrics']) for e in fit_epochs
                    if e.get('panel',{}).get('panel_signature')==signature]
        fit_scope='full Train Online; same fixed definition across compared epochs'
    result['fit'] = dict(fit_state(fit_epochs),scope=fit_scope)
    result['committed_cursor'] = max((e['cursor'] for e in result['epochs']), default=0)
    return result


def console_summary(report, all_epochs=False):
    lines = []
    epochs = report['epochs'] if all_epochs else report['epochs'][-1:]
    if not epochs: lines.append('尚无完整 epoch；loss 曲线展示已记录更新，分组指标等待首次验证。')
    for epoch in epochs:
        lines.append('\n[Diagnostics] '+epoch['tag'])
        if epoch.get('train') and not epoch.get('panel'):
            lines += table_lines('[Train fixed Online probe — sample, not full Train]', epoch['train'])
        if epoch.get('panel'):
            panel=epoch['panel']
            lines += table_lines('[Train FULL — '+panel['scope']+']', panel['metrics'])
        else:
            lines.append('  Train offline/noisy_train: not measured for this checkpoint; not replaced by Dev Seen.')
        if epoch.get('dev'): lines += table_lines('[Dev — full fixed Online / Seen / Heldout]', epoch['dev'])
        m = epoch['metrics']
        lines.append('  Dev Clean/Noisy/Weighted='+'/'.join(f'{100*m[k]:.3f}' for k in ('clean_f1','noisy_f1','weighted_f1')))
        s=epoch['segment_training']; n=s.get('updates',0)
        if n:
            lines.append('  Epoch mean TOTAL/CE/weighted-time/weighted-CKA='+'/'.join(
                f'{s[k]/n:.6f}' if k in s else 'NA' for k in ('total_loss','classification_loss','weighted_time_loss','weighted_structure_loss')))
    lines.append('[Fit] '+report['fit']['text']+' ('+report['fit']['scope']+')')
    if 'recall_gap_pp' in report['fit']:
        lines.append(f'  Fixed Online Train–Dev recall gap={report["fit"]["recall_gap_pp"]:.2f} pp; '
                     f'Dev–Train CE gap={report["fit"]["ce_gap"]:.4f}')
    lines += ['[Notice] '+w for w in report['warnings']]
    return '\n'.join(lines)
