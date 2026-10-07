"""Official Offline is diagnostic only and never changes the Online objective."""
from pathlib import Path

import numpy as np

from w2v_aasist.data import read_protocol
from w2v_v36.data import _audio
from w2v_v36.metrics import evaluate
from w2v_v39.common import digest, read_json
from w2v_v312.replay import delta


def offline_rows(cfg, dev_rows, train_rows):
    source = cfg.get('dev_config', {})
    if not source.get('dev_protocol') or not source.get('dev_data_path'):
        return [], 'Official Offline protocol is absent from the recorded source configuration'
    protocol = Path(source['dev_protocol']).resolve()
    if cfg['data_fingerprints'].get(str(protocol)) != digest(protocol):
        raise ValueError('Offline diagnostic protocol is not pinned to the existing Dev')
    originals = {r['source_id']: r for r in dev_rows if r['condition'] == 'seen'}
    output = []
    train_hashes = {r['group_id'] for r in train_rows}
    for r in read_protocol(protocol, source['dev_data_path']):
        if r['domain'] != 'offline':
            continue
        previous = originals.get(r['id'])
        if previous is None or any(r[k] != previous[k] for k in ('label', 'language')):
            raise ValueError('Offline diagnostic labels/source differ from the fixed Noisy Dev')
        meta = _audio(r['audio'], previous['source_sha256'])
        if meta['audio_sha256'] in train_hashes:
            raise ValueError('Offline diagnostic shares an original Train waveform')
        output.append(dict(r, **meta, source_id=r['id'], group_id=meta['audio_sha256'],
            source_sha256=meta['audio_sha256'], condition='offline', split='dev', view='full', full_length=True))
    if {r['id'] for r in output} != set(originals):
        raise ValueError('Official Offline diagnostic inventory is incomplete')
    return output, 'Baseline and whole-epoch diagnostics only; excluded from Weighted and all gradients'


def source_replay(cfg, rows, logits):
    source = Path(cfg['source_run'])
    scores = source/('dev_scores_'+cfg['starting_tag']+'.npz')
    if not scores.is_file() or not (source/'dev_rows.json').is_file():
        return dict(status='scores_unavailable', note='Weight identity verified; historical score replay unavailable')
    old = read_json(source/'dev_rows.json')
    keys = ('id', 'source_id', 'group_id', 'condition', 'language', 'label')
    if [{k:r[k] for k in keys} for r in rows] != [{k:r[k] for k in keys} for r in old]:
        raise ValueError('Starting LAST Dev row identity/order differs')
    with np.load(scores, allow_pickle=False) as data:
        result = delta(data['logits'], logits)
    if not result['allclose'] or result['decision_changes']:
        raise ValueError('Starting V3.12 LAST no longer reproduces its committed validation')
    return dict(result, status='passed')


def offline_metrics(rows, logits):
    return dict(groups=evaluate(rows, logits)['groups'], included_in_weighted=False)
