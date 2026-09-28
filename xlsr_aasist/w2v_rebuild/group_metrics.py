"""Reuse existing predictions for language diagnostics and a weights-free report ZIP."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

GROUPS = ('en-fake', 'en-real', 'zh-fake', 'zh-real')
CONDITIONS = ('online', 'offline', 'seen', 'heldout')


class GroupMetrics:
    """Unweighted diagnostics; no scalar reads or host transfers in update().

    The caller validates labels/languages on CPU. Counts include repeated and
    augmented training views, so these are not independent-source counts.
    """
    def __init__(self, device='cpu'):
        self.sums = torch.zeros((4, 4), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, logits, labels, languages):
        if logits.shape != (len(labels), 2) or languages.shape != labels.shape or labels.ndim != 1:
            raise ValueError('Group metric shapes differ')
        if not len(labels):
            return
        z = logits.detach().float().to(self.sums.device)
        y = labels.detach().long().to(self.sums.device)
        language = languages.detach().long().to(self.sums.device)
        index = 2*language + y
        p = z.softmax(1)[:, 0]
        predicted = (p < .5).long()  # Same tie rule as official fixed-threshold metrics.
        values = torch.stack((torch.ones_like(p), (predicted == y).float(),
                              F.cross_entropy(z, y, reduction='none'), p), dim=1).double()
        self.sums.scatter_add_(0, index[:, None].expand(-1, 4), values)

    def result(self):
        rows = self.sums.detach().cpu().tolist()
        if not all(math.isfinite(v) for row in rows for v in row):
            raise FloatingPointError('Non-finite language metrics')
        return {name: {'count': int(row[0]),
                       'recall': row[1]/row[0] if row[0] else None,
                       'mean_ce': row[2]/row[0] if row[0] else None,
                       'mean_fake_score': row[3]/row[0] if row[0] else None}
                for name, row in zip(GROUPS, rows)}


def auc_fake(labels, margins):
    """Fake-positive AUC on raw margins, with average ranks for tied scores."""
    labels, margins = np.asarray(labels), np.asarray(margins, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != margins.shape:
        raise ValueError('AUC array shapes differ')
    if not np.isfinite(margins).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError('AUC requires finite margins and binary labels')
    positive = labels == 0
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(margins, kind='stable')
    ordered = margins[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], len(labels)]
    ranks = np.empty(len(labels), dtype=np.float64)
    ranks[order] = np.repeat((starts + stops + 1)/2, stops - starts)
    return float((ranks[positive].sum() - npos*(npos+1)/2)/(npos*nneg))


class DevGroupRecorder:
    """CPU validation predictions, optional atomically committed per-view JSONL.

    Use as a context manager. A failed validation removes its temporary output;
    it never replaces a previously completed score file. AUC over noisy bands is
    pooled diagnostic AUC, not an official four-band average or independent test.
    """
    def __init__(self, output_path=None):
        self.output_path = Path(output_path) if output_path is not None else None
        self.meters = {name: GroupMetrics() for name in CONDITIONS}
        self.band_meters = {name: [GroupMetrics() for _ in range(4)] for name in ('seen', 'heldout')}
        self.scores = {name: {lang: ([], []) for lang in ('en', 'zh')} for name in CONDITIONS}
        self._stream, self._tmp, self._entered = None, None, False

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Recorder contexts cannot be re-entered')
        self._entered = True
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_path.exists():
                raise FileExistsError(f'Dev scores already exist: {self.output_path}')
            self._stream = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                                       dir=self.output_path.parent,
                                                       prefix=self.output_path.name+'.', suffix='.tmp', delete=False)
            self._tmp = Path(self._stream.name)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._stream is not None:
                self._stream.close()
            if self._tmp is not None and exc_type is None:
                if self.output_path.exists():
                    raise FileExistsError(f'Dev scores already exist: {self.output_path}')
                self._tmp.replace(self.output_path)
        finally:
            if self._tmp is not None and self._tmp.exists():
                self._tmp.unlink()
        return False

    @torch.no_grad()
    def update(self, condition, logits, labels, ids, bands=None):
        from .language import language_id
        if not self._entered:
            raise RuntimeError('Use DevGroupRecorder as a context manager')
        if condition not in CONDITIONS:
            raise ValueError('Unknown validation condition')
        if logits.device.type != 'cpu' or labels.device.type != 'cpu':
            raise ValueError('Pass already-transferred CPU validation predictions')
        if logits.shape != (len(labels), 2) or len(ids) != len(labels):
            raise ValueError('Prediction and source counts differ')
        if not bool(torch.isfinite(logits).all()) or not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError('Invalid validation predictions or labels')
        if bands is None:
            if condition in self.band_meters:
                raise ValueError('Noisy validation requires per-view bands')
            band_values = [None]*len(labels)
        else:
            band_values = list(bands)
            if len(band_values) != len(labels) or any(int(b) != b or not 0 <= b < 4 for b in band_values):
                raise ValueError('Expected noisy bands 0..3')
            if condition not in self.band_meters:
                raise ValueError('Clean validation must not carry noisy bands')
            band_values = [int(b) for b in band_values]
        language_values = [language_id(str(source)) for source in ids]
        languages = torch.tensor(language_values, dtype=torch.long)
        self.meters[condition].update(logits, labels, languages)
        if condition in self.band_meters:
            band_tensor = torch.tensor(band_values, dtype=torch.long)
            for b, meter in enumerate(self.band_meters[condition]):
                mask = band_tensor == b
                meter.update(logits[mask], labels[mask], languages[mask])
        z = logits.detach().float()
        probabilities = z.softmax(1)[:, 0].tolist()
        margins = (z[:, 0] - z[:, 1]).tolist()
        for source, label, lang, band, prob, margin in zip(ids, labels.tolist(), language_values,
                                                        band_values, probabilities, margins):
            language = ('en', 'zh')[lang]
            ys, ms = self.scores[condition][language]
            ys.append(int(label))
            ms.append(margin)
            if self._stream is not None:
                self._stream.write(json.dumps({'condition': condition, 'source_id': str(source),
                                               'label': int(label), 'language': language,
                                               'p_fake': prob, 'margin': margin, 'band': band},
                                              ensure_ascii=False, allow_nan=False)+'\n')

    def result(self):
        return {'conditions': {
                    condition: {'groups': self.meters[condition].result(),
                                'auc_by_language': {language: auc_fake(*values)
                                                    for language, values in self.scores[condition].items()}}
                    for condition in CONDITIONS},
                'noisy_bands': {condition: {str(i): meter.result() for i, meter in enumerate(meters)}
                                for condition, meters in self.band_meters.items()},
                'auc_note': 'Fake-positive raw-margin AUC; seen/heldout AUC pools correlated band views. '
                            'Diagnostic only, not official per-band mean or an independent test.',
                'train_note': 'Training group metrics count augmented/repeated views; they are unweighted '
                              'diagnostics, not unique-source counts or clean Train evaluation.'}


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def _write_atomic(path, text):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(text)
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _pct(value):
    return '-' if value is None else f'{100*value:.3f}'


def _dev_row(name, dev):
    return f'| {name} | '+ ' | '.join(_pct(dev.get(k, {}).get('macro_f1')) for k in ('online', 'seen', 'heldout')) + \
           f" | {_pct(dev.get('robust_f1'))} | {dev.get('robust_ce', '-')} |"


def tradeoff_metrics(dev):
    """Use the original per-band averages, not pooled score-view statistics."""
    def mean_band_recall(label):
        values = []
        for condition in ('seen', 'heldout'):
            bands = dev.get(condition, {}).get('bands', [])
            if len(bands) != 4:
                return None
            for band in bands:
                recalls = band.get('recall', [])
                if len(recalls) != 2 or recalls[label] is None:
                    return None
                values.append(recalls[label])
        return sum(values)/len(values)
    online_recall = dev.get('online', {}).get('recall', [])
    noisy_f1 = [dev.get(condition, {}).get('macro_f1') for condition in ('seen', 'heldout')]
    return {'online_real_recall': online_recall[1] if len(online_recall) == 2 else None,
            'noisy_fake_recall': mean_band_recall(0), 'noisy_real_recall': mean_band_recall(1),
            'noisy_f1': sum(noisy_f1)/2 if all(value is not None for value in noisy_f1) else None}


def comparable_reference(reference, current_config):
    """Recheck saved reference hashes against this run's freshly recorded Dev inputs.

    Path text is normalized without resolving it on the reporting machine: an
    exported Linux run may be reviewed on Windows. Training hashes do not affect
    Dev comparability. Missing proof makes the reference diagnostic-only.
    """
    if not reference:
        return reference
    result = copy.deepcopy(reference)
    if not result.get('available', False):
        return result
    try:
        if not isinstance(current_config, dict):
            raise ValueError('Current run config is missing')
        reference_inputs = result.get('dev_inputs')
        if not isinstance(reference_inputs, dict):
            raise ValueError('Reference Dev input paths are missing')
        def normalized(path):
            return str(path).replace('\\', '/').rstrip('/')
        def fingerprints(value, label):
            if not isinstance(value, dict) or not value:
                raise ValueError(f'{label} data_fingerprints are missing')
            items = {normalized(key): digest for key, digest in value.items()}
            if len(items) != len(value):
                raise ValueError(f'{label} fingerprint paths are ambiguous')
            return items
        old_hashes = fingerprints(result.get('data_fingerprints'), 'Reference')
        new_hashes = fingerprints(current_config.get('data_fingerprints'), 'Current')
        for key in ('eval_microbatch', 'eval_batch'):
            old, new = result.get(key), current_config.get(key)
            if type(old) is not int or type(new) is not int or old < 1 or new < 1 or old != new:
                raise ValueError(f'Reference inference setting differs or is missing: {key}')
        if (not reference_inputs.get('dev_data_path') or not current_config.get('dev_data_path')
                or normalized(reference_inputs['dev_data_path']) != normalized(current_config['dev_data_path'])):
            raise ValueError('Reference clean Dev data path differs or is missing')
        roles = [('dev_protocol', None), ('dev_noisy_cache', 'config.json'),
                 ('dev_noisy_cache', 'manifest.jsonl'), ('dev_heldout_cache', 'config.json'),
                 ('dev_heldout_cache', 'manifest.jsonl'), ('ssl_path', 'preprocessor_config.json')]
        checked = {}
        for key, suffix in roles:
            if not reference_inputs.get(key) or not current_config.get(key):
                raise ValueError(f'Reference/current Dev input setting is missing: {key}')
            old_path, new_path = normalized(reference_inputs[key]), normalized(current_config[key])
            if suffix:
                old_path += '/'+suffix
                new_path += '/'+suffix
            role = key + ('/'+suffix if suffix else '')
            old, new = old_hashes.get(old_path), new_hashes.get(new_path)
            if not isinstance(old, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', old):
                raise ValueError(f'Reference Dev fingerprint missing or invalid: {role}')
            if not isinstance(new, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', new):
                raise ValueError(f'Current Dev fingerprint missing or invalid: {role}')
            if old.lower() != new.lower():
                raise ValueError(f'Reference Dev fingerprint differs from current run: {role}')
            checked[role] = new.lower()
        result['comparability'] = {'verified': True, 'checked_dev_fingerprints': checked,
                                   'eval_microbatch': result['eval_microbatch'],
                                   'eval_batch': result['eval_batch']}
    except (TypeError, ValueError) as exc:
        result.update(available=False, reason=str(exc), comparability={'verified': False})
    return result


def export_language_report(stage_dir, reference_metrics=None, download_dir=None):
    """Export only an explicit diagnostic allowlist; never weights/audio/features.

    Returns report, summary, archive and download_archive path strings. A missing
    completed.json or failed launcher status is prominently marked incomplete.
    """
    stage = Path(stage_dir).expanduser().resolve(strict=True)
    if not stage.is_dir():
        raise ValueError('stage_dir must be an existing directory')
    if isinstance(reference_metrics, (str, Path)):
        reference_metrics = _read_json(Path(reference_metrics).expanduser().resolve(strict=True))
    current_config = _read_json(stage/'config.json')
    reference_metrics = comparable_reference(reference_metrics, current_config)
    baseline = _read_json(stage/'baseline_dev.json')
    epochs = [_read_json(path) for path in sorted(stage.glob('epoch_*_evaluation.json'))
              if re.fullmatch(r'epoch_\d+_evaluation\.json', path.name)]
    completed, launcher = _read_json(stage/'completed.json'), _read_json(stage/'launcher_status.json')
    failed = bool(launcher and (launcher.get('failed') or launcher.get('status') == 'failed'
                               or launcher.get('returncode') not in (None, 0)))
    status = 'failed' if failed else ('complete' if completed else 'incomplete')
    summary = {'schema': 'w2v_language_report_v1', 'status': status, 'stage_dir': str(stage),
               'baseline': baseline, 'epochs': epochs, 'completed': completed, 'launcher': launcher,
               'config': current_config,
               'language_budget': _read_json(stage/'language_budget.json'),
               'reference': reference_metrics,
               'notes': ['Metrics are fixed-threshold metrics on the same Dev conditions.',
                         'Noisy language AUC pools correlated band views; it is diagnostic only.',
                         'Train group metrics use augmented/repeated training views, not clean independent sources.',
                         'Reference E1 is copied from prior metrics; it is not reevaluated and may lack group metrics.',
                         'The package contains diagnostic text only, no checkpoints, features or audio.']}
    coverage_run = bool(current_config and current_config.get('coverage_training')) or (stage.parent/'coverage_plan.json').is_file()
    structure_run = bool(current_config and current_config.get('local_structure_weight')) or (stage.parent/'structure_plan.json').is_file()
    title = ('Local-structure robustness report' if structure_run else
             ('Coverage-repair fine-tuning report' if coverage_run else 'English-weighted fine-tuning report'))
    lines = ['# '+title, '', f'**Run status: {status.upper()}**', '',
             'F1/recall values below are percentages; AUC is on the 0-1 scale. Missing values are shown as `-`.', '',
             '| Model | Online F1 | Seen F1 | Heldout F1 | Robust F1 | Robust CE |',
             '|---|---:|---:|---:|---:|---:|']
    devs = []
    if baseline:
        devs.append(('Original best (before training)', baseline))
    for epoch in epochs:
        if isinstance(epoch, dict) and isinstance(epoch.get('dev'), dict):
            devs.append((f"Epoch {epoch.get('epoch', '?')}", epoch['dev']))
    if reference_metrics and reference_metrics.get('available', True):
        reference = reference_metrics.get('dev') or reference_metrics.get('metrics', {})
        if 'dev' in reference:
            reference = reference['dev']
        if reference:
            devs.append((f"Prior reference E{reference_metrics.get('epoch', '?')} (saved metrics)", reference))
    lines.extend(_dev_row(name, dev) for name, dev in devs)
    if reference_metrics and not reference_metrics.get('available', False):
        lines += ['', '**Prior reference excluded from comparable-model tables:** '+
                  reference_metrics.get('reason', 'Comparability could not be verified')+
                  '. Its original record is retained in summary.json.', '']
    summary['tradeoff_comparison'] = [{'model': name, **tradeoff_metrics(dev)} for name, dev in devs]
    lines += ['', '## Real/fake tradeoffs on fixed Dev', '',
              'Noisy recall is the equal average over all four seen and four heldout bands; noisy F1 is the mean of seen and heldout F1.', '',
              '| Model | Online real recall | Mean noisy fake recall | Mean noisy real recall | Mean noisy F1 |',
              '|---|---:|---:|---:|---:|']
    for row in summary['tradeoff_comparison']:
        lines.append(f"| {row['model']} | "+' | '.join(_pct(row[key]) for key in
                     ('online_real_recall', 'noisy_fake_recall', 'noisy_real_recall', 'noisy_f1'))+' |')
    lines += ['', '## Language diagnostics', '',
              '| Model | Condition | en fake recall | en real recall | zh fake recall | zh real recall | en AUC | zh AUC |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, dev in devs:
        for condition, values in dev.get('language_groups', {}).get('conditions', {}).items():
            groups, auc = values.get('groups', {}), values.get('auc_by_language', {})
            if not any(g.get('count', 0) for g in groups.values()):
                continue
            auc_text = [f'{auc[k]:.6f}' if auc.get(k) is not None else '-' for k in ('en', 'zh')]
            lines.append(f'| {name} | {condition} | '+ ' | '.join(_pct(groups.get(g, {}).get('recall'))
                         for g in GROUPS)+' | '+' | '.join(auc_text)+' |')
    lines += ['', '## Training groups (augmented views)', '',
              '| Epoch | Branch | Group | Views | Recall | Mean unweighted CE |',
              '|---|---|---|---:|---:|---:|']
    for epoch in epochs:
        for branch, values in epoch.get('train_groups', {}).items():
            for group, metric in values.items():
                ce = '-' if metric.get('mean_ce') is None else f"{metric['mean_ce']:.6f}"
                lines.append(f"| {epoch.get('epoch', '?')} | {branch} | {group} | {metric.get('count', 0)} | "
                             f"{_pct(metric.get('recall'))} | {ce} |")
    if coverage_run:
        summary['coverage'] = {p.name: _read_json(p) for p in sorted(stage.glob('coverage_*_epoch_*.json'))
                               if re.fullmatch(r'coverage_(plan|actual)_epoch_\d+\.json', p.name)}
        lines += ['', '## Consumed coverage (completed optimizer steps only)', '',
                  'Ordinary uses one window per file. Crop groups are label|language|kind; fake/en=0, real/zh=1.',
                  'The noisy sampler implements the English target shares; noisy CE language multipliers are one.', '',
                  '| Epoch | Ordinary views | Unique files | Random accepted | Prefix/fallback/short | Noisy processed views |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, value in summary['coverage'].items():
            if not name.startswith('coverage_actual'):
                continue
            random_count = sum(v for k, v in value['crop_counts'].items() if k.endswith('|random'))
            lines.append(f"| {name} | {value['ordinary_views']} | {value['ordinary_unique_files']} | {random_count} | "
                         f"{value['ordinary_views']-random_count} | {value['noisy_processed_views']} |")
        lines += ['', 'Each cell is bank|family|SNR-band|label|language. See coverage plan/actual JSON for counts and distinct sources.',
                  'Energy screening is not speech recognition. Random crops are a hypothesis, not verified accuracy gains.', '']
    if structure_run:
        summary['local_structure_config'] = (current_config or {}).get('local_structure_config')
        summary['structure_plan'] = _read_json(stage.parent/'structure_plan.json')
        metrics_path = stage/'metrics.jsonl'
        structure_epochs = []
        if metrics_path.is_file():
            for raw in metrics_path.read_text(encoding='utf-8').splitlines():
                row = json.loads(raw)
                values = {k: v for k, v in row.get('mean_batch', {}).items() if k.startswith('structure_')}
                structure_epochs.append({'epoch': row['epoch'], **values})
        summary['structure_training'] = structure_epochs
        lines += ['', '## Local structure objective', '',
                  'Only same-recording noisy pairs are matched. Low-dynamic/ambiguous latent blocks are rejected; this is not a VAD or phoneme recognizer.',
                  'The local objective replaces the noisy global contrastive contribution. Classification and real RTC supervision remain active.',
                  'Structure loss and accepted match fractions are diagnostic, not evidence of accuracy by themselves.',
                  'Selection prioritizes noisy F1; promotion also requires improved weighted F1 and no clean Online F1 decrease.', '',
                  'Each group cell gives usable-pair percentage / accepted-bin percentage. Training dropout can make these differ from frozen audit results.', '',
                  '| Epoch | Structure loss | en-fake usable/accepted | en-real usable/accepted | zh-fake usable/accepted | zh-real usable/accepted |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in structure_epochs:
            fractions = []
            for group in ('en_fake', 'en_real', 'zh_fake', 'zh_real'):
                denominator = row.get('structure_'+group+'_pairs', 0)
                value = row.get('structure_'+group+'_accepted_sum', 0)/denominator if denominator else None
                usable = row.get('structure_'+group+'_usable_sum', 0)/denominator if denominator else None
                fractions.append(_pct(usable)+' / '+_pct(value))
            lines.append('| '+str(row['epoch'])+' | '+str(row.get('structure_loss', '-'))+' | '+' | '.join(fractions)+' |')
    lines += ['', '## Interpretation and limits', ''] + ['- '+note for note in summary['notes']]
    lines += ['- A real-recall gain alone is not evidence of better discrimination; inspect fake recall, noisy F1 and AUC together.',
              '- Group weighting changes loss coefficients. It does not guarantee the same shares of actual gradient or loss.',
              '- Existing best selection guards remain authoritative. This report does not promote or overwrite checkpoints.',
              '- See summary.json for noisy bands, exact counts, loss budgets, configuration and selection/completion details.', '']
    if status != 'complete':
        lines += ['**This is a partial/failed-run diagnostic package, not a successfully completed training result.**', '']
    output = stage/'language_report'
    output.mkdir(exist_ok=True)
    if output.is_symlink():
        raise ValueError('Report output must not be a symbolic link')
    report, summary_path = output/'report.md', output/'summary.json'
    _write_atomic(summary_path, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    _write_atomic(report, '\n'.join(lines))
    allowed = {'baseline_dev.json', 'config.json', 'completed.json', 'language_budget.json',
               'launcher_status.json', 'metrics.jsonl', 'baseline_scores.jsonl', 'preflight.json'}
    sources = [(report, 'report.md'), (summary_path, 'summary.json')]
    for path in sorted(stage.iterdir()):
        if (path.name in allowed or re.fullmatch(r'epoch_\d+_(evaluation\.json|scores\.jsonl)', path.name)
                or re.fullmatch(r'coverage_(plan|actual)_epoch_\d+\.json', path.name)):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Diagnostic input must be a regular file: {path}')
            sources.append((path, 'stage3/'+path.name))
    for name in ('en_plan.json', 'en_execution.log', 'coverage_plan.json', 'coverage_execution.log',
                 'structure_plan.json', 'structure_execution.log', 'reviewed_audit_manifest.json', 'reviewed_audit_summary.json'):
        path = stage.parent/name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Diagnostic input must be a regular file: {path}')
            sources.append((path, name))
    archive = output/(stage.parent.name+'_language_report_'+uuid.uuid4().hex[:8]+'.zip')
    temporary_archive = Path(str(archive)+'.tmp')
    try:
        with zipfile.ZipFile(temporary_archive, 'x', zipfile.ZIP_DEFLATED) as package:
            for path, name in sources:
                package.write(path, name)
        with zipfile.ZipFile(temporary_archive) as package:
            if package.testzip() is not None:
                raise IOError('Diagnostic archive verification failed')
        temporary_archive.replace(archive)
    finally:
        temporary_archive.unlink(missing_ok=True)
    downloaded = None
    if download_dir is not None:
        destination = Path(download_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        downloaded = destination/archive.name
        with archive.open('rb') as source, downloaded.open('xb') as target:
            shutil.copyfileobj(source, target)
    return {'report': str(report), 'summary': str(summary_path), 'archive': str(archive),
            'download_archive': str(downloaded) if downloaded else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage_dir', required=True)
    parser.add_argument('--download_dir')
    parser.add_argument('--reference_metrics')
    args = parser.parse_args()
    print(json.dumps(export_language_report(args.stage_dir, args.reference_metrics, args.download_dir), indent=2))


if __name__ == '__main__':
    main()
