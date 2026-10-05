"""Evaluate serialized deployment modules once; diagnostics never alter their scores."""
import numpy as np
import torch

from w2v_v36.fit import _save_dev_scores
from .metrics import measure, selection, change_audit
from .model import ResidualClassifier


def replay(spec, x, device='cpu', chunk=8192):
    module = ResidualClassifier.restore(spec).to(device)
    output = np.empty((len(x), 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), chunk):
            value = torch.from_numpy(np.array(x[start:start+chunk], dtype=np.float32, copy=True)).to(device)
            output[start:start+chunk] = module(value).cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError('Nonfinite deployment scores')
    return output


def evaluate_candidates(result, bundle, baseline_logits, weight, bias, cfg, out):
    rows, x = bundle['rows'], bundle['x']
    result['baseline'] = measure(rows, baseline_logits, target=cfg['matched_fake_recall'])
    _save_dev_scores(out, 'baseline', rows, baseline_logits)
    for item in result['candidates']:
        if item['status'] != 'fitted':
            continue
        scores = replay(item['spec'], x, cfg['device'], cfg['batch_rows'])
        reference = replay(item['spec'], x, 'cpu', cfg['batch_rows']) if cfg['device'] != 'cpu' else scores
        error = float(np.max(np.abs(scores.astype(np.float64) - reference)))
        changed = int(np.count_nonzero(scores.argmax(1) != reference.argmax(1)))
        item['replay'] = dict(max_cpu_device_logit_difference=error, decision_differences=changed,
                              actual_module=True, device=cfg['device'])
        if changed or not np.allclose(scores, reference, rtol=3e-5, atol=1e-3):
            item.update(status='rejected', reason='deployment_cpu_device_replay_mismatch')
            continue
        item['metrics'] = measure(rows, scores, target=cfg['matched_fake_recall'])
        item['decision_changes'] = change_audit(rows, baseline_logits, scores)
        base_margin = baseline_logits[:, 0] - baseline_logits[:, 1]
        margin = scores[:, 0] - scores[:, 1]
        delta = margin - base_margin
        wrong = baseline_logits.argmax(1) != np.asarray([r['label'] for r in rows])
        item['correction_audit'] = dict(
            decisions_changed=int(np.count_nonzero(scores.argmax(1) != baseline_logits.argmax(1))),
            mean_absolute_delta=float(np.mean(np.abs(delta))),
            max_absolute_delta=float(np.max(np.abs(delta))),
            baseline_wrong_rows=int(wrong.sum()),
            baseline_wrong_mean_absolute_delta=float(np.mean(np.abs(delta[wrong]))) if wrong.any() else 0.)
        item['dev_score_file'] = _save_dev_scores(out, item['name'], rows, scores)
        print(f'[Replay] {item["name"]} max_logit_delta={error:.7g} decision_changes={changed}', flush=True)
    selected = selection(result['baseline'], result['candidates'], cfg)
    spec = next((c['spec'] for c in result['candidates'] if c['name'] == selected),
                ResidualClassifier(weight, bias).spec())
    result['selected'] = selected
    return result, spec


def report(run, result):
    from .common import atomic_json
    from pathlib import Path
    clean = dict(result, candidates=[{k: v for k, v in c.items() if k != 'spec'} for c in result['candidates']])
    atomic_json(Path(run) / 'fit_report.json', clean)
    lines = ['# V3.9 adaptive correction on frozen features', '',
        'Local fixed Dev proxy. Clean = Online only; Weighted = 0.3 Clean + 0.7 mean(Seen, Heldout).',
        'Optimization and epoch/regularization selection use official Train only; no Dev refitting.',
        'Repair range grows with the original margin; correct-decision protection replaces fixed-logit anchoring.',
        'Train-only hard-example weighting preserves language/class/condition budgets.',
        'Train holdout separates source groups, but the frozen encoder previously saw Train.',
        'Matched-fake-recall thresholds are diagnostic only; exported scores always use threshold 0.5.', '',
        '| Model | Clean | Noisy | Weighted |', '|---|---:|---:|---:|']
    entries = [('baseline', result['baseline'])] + [(c['name'], c['metrics']) for c in result['candidates'] if c.get('metrics')]
    for name, value in entries:
        numbers = [value[k] * 100 for k in ('clean_f1', 'noisy_f1', 'weighted_f1')]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.3f}' for v in numbers) + ' |')
        print('[Dev] V3.9 ' + name + ' Clean=%.3f Noisy=%.3f Weighted=%.3f' % tuple(numbers), flush=True)
        for group, scores in value['groups'].items():
            if '/' in group:
                print(f'  {group} fake={scores["recall"][0]*100:.3f}% real={scores["recall"][1]*100:.3f}% AUC={scores["auc"]*100:.3f}%', flush=True)
    for item in result['candidates']:
        message = item['name'] + ': eligible=' + str(item['eligible']) + ' reasons=' + str(item['guardrails'])
        print('[Selection] ' + message, flush=True)
        lines += ['', message]
        if item.get('correction_audit'):
            audit = item['correction_audit']
            message = (f"[Correction] {item['name']} changed={audit['decisions_changed']} "
                       f"mean_abs_delta={audit['mean_absolute_delta']:.4f} "
                       f"max_abs_delta={audit['max_absolute_delta']:.4f}")
            print(message, flush=True)
            lines += ['', message]
        if item['name'] == 'language_residual':
            message = ('Language-specific evidence=' + str(item.get('language_specific_evidence', False))
                       + '; reasons=' + str(item.get('language_attribution_reasons', [])))
            print('[Attribution] ' + message, flush=True)
            lines += ['', message]
    lines += ['', 'Selected: ' + result['selected'],
        'Language student qualified: ' + str(result['language']['qualified']),
        'Teacher mean is the reconstruction null control; raw cosine is not language accuracy.',
        'Promotion uses performance and safety guards. Language attribution is reported separately, never invented from a score win.',
        'No guarantee of official Weighted 97. No old best, cache, or training script was deleted.']
    Path(run, 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
