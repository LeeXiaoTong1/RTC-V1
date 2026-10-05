"""Borrow completed V3.7 vectors, verify their identities, never regenerate audio."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .common import digest, read_json, verify_files


def cache_path(source, split):
    source = Path(source)
    local = source / 'features' / split
    if (local / 'complete.json').is_file():
        return local
    reuse = read_json(source / 'feature_reuse.json')
    borrowed = reuse.get(split, {}).get('path')
    if not borrowed or not (Path(borrowed) / 'complete.json').is_file():
        raise FileNotFoundError(f'Completed {split} vectors missing. V3.9 will not regenerate them.')
    return Path(borrowed).resolve()


def verify_owner(path, split, cfg):
    from w2v_v36.features import MODE
    owner = read_json(Path(path) / 'owner.json')
    identity = owner['identity']
    if (owner.get('mode') != MODE or identity.get('split') != split
            or identity.get('base_checkpoint_sha256') != cfg['base_checkpoint_sha256']
            or identity.get('data_fingerprints') != cfg['data_fingerprints']):
        raise ValueError('Cached vectors belong to a different base, split, or dataset')
    if owner['preprocessing_sha256'] != digest(Path(cfg['ssl_path']) / 'preprocessor_config.json'):
        raise ValueError('Cached preprocessing differs from the protected best')
    recorded = identity.get('code_fingerprints', {})
    for folder, name in (('w2v_v36', 'features.py'), ('w2v_v3', 'model.py'),
                         ('w2v_v3', 'data.py'), ('w2v_v3', 'step.py'),
                         ('w2v_aasist', 'model.py'), ('w2v_aasist', 'data.py')):
        matches = {p: h for p, h in recorded.items() if Path(p).parent.name == folder and Path(p).name == name}
        if len(matches) != 1:
            raise ValueError('Missing feature producer identity: ' + folder + '/' + name)
        verify_files(matches)


def base_weights(cfg):
    """Read only the last layer of the pinned checkpoint, without building the SSL model."""
    from w2v_v37.patch import load_selected
    source = Path(cfg['v37_run'])
    verify_files({str(source / 'completed.json'): cfg['v37_completed_sha256'],
                  str(source / 'best_patch.pt'): cfg['v37_patch_sha256'],
                  cfg['base_checkpoint']: cfg['base_checkpoint_sha256']})
    patch, done = load_selected(source)
    if done['selected'] != 'baseline':
        raise ValueError('Source must be the unchanged V3.7 baseline')
    state = torch.load(cfg['base_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    if state.get('kind') != 'weights' or state.get('tag') != cfg['base_tag']:
        raise ValueError('Protected base checkpoint kind/tag differs')
    weight = state['model']['head.classifier.2.weight'].detach().clone()
    bias = state['model']['head.classifier.2.bias'].detach().clone()
    if weight.shape != (2, 512) or bias.shape != (2,):
        raise ValueError('Expected the submitted MultiConv final Linear(512,2)')
    if not torch.equal(weight, patch['weight']) or not torch.equal(bias, patch['bias']):
        raise ValueError('V3.7 baseline patch is not identical to the original final layer')
    return weight, bias


def validate_rows(train, dev, language):
    if train != language:
        raise ValueError('Teacher and detector Train row identity/order differs')
    if any(r.get('split') != 'train' for r in train) or any(r.get('split') != 'dev' for r in dev):
        raise ValueError('Official Train and fixed Dev split metadata are required')
    train_groups = {r.get('group_id', r['source_id']) for r in train}
    dev_groups = {r.get('group_id', r['source_id']) for r in dev}
    if train_groups & dev_groups:
        raise ValueError('Train/Dev original-source overlap; refusing leakage')
    if any(r['condition'] not in ('online', 'seen', 'heldout') for r in dev):
        raise ValueError('Dev must use Online Clean and both fixed noisy pools')


@contextmanager
def borrowed_bundles(cfg):
    from w2v_v36.features import load_cache
    from w2v_v37.language_cache import load_language_cache
    source = Path(cfg['v37_run'])
    bundles, paths = {}, {}
    try:
        verify_files(cfg['data_fingerprints'])
        for split in ('train', 'dev'):
            path = cache_path(source, split)
            verify_owner(path, split, cfg)
            bundles[split] = load_cache(path)
            paths[split] = str(path.resolve())
        language_path = source / 'language' / 'train'
        bundles['language'] = load_language_cache(language_path)
        identity = bundles['language']['manifest']['identity']
        if (identity.get('base_checkpoint_sha256') != cfg['base_checkpoint_sha256']
                or identity.get('split') != 'train'
                or identity.get('data_fingerprints') != cfg['data_fingerprints']):
            raise ValueError('Teacher vectors belong to another Train/base identity')
        validate_rows(bundles['train']['rows'], bundles['dev']['rows'], bundles['language']['rows'])
        bundles['train']['lid'] = bundles['language']['lid']
        paths['language'] = str(language_path.resolve())
        yield bundles['train'], bundles['dev'], paths
    finally:
        closed = set()
        for bundle in bundles.values():
            for array in bundle.values():
                mapped = getattr(array, '_mmap', None)
                if mapped is not None and id(mapped) not in closed:
                    mapped.close()
                    closed.add(id(mapped))


def replay_base(bundle, weight, bias, device, chunk=8192):
    measured = np.empty((len(bundle['rows']), 2), dtype=np.float32)
    weight, bias = weight.to(device), bias.to(device)
    with torch.inference_mode():
        for start in range(0, len(measured), chunk):
            x = torch.from_numpy(np.array(bundle['x'][start:start+chunk], copy=True)).to(device)
            measured[start:start+chunk] = torch.nn.functional.linear(x, weight, bias).cpu().numpy()
    if not np.allclose(measured, bundle['logits'], rtol=3e-5, atol=1e-3):
        raise ValueError('Cached final-layer inputs do not replay the protected base logits')
    if not np.array_equal(measured.argmax(1), np.asarray(bundle['logits']).argmax(1)):
        raise ValueError('Baseline decisions changed during replay; inspect numeric execution before fitting')
    return measured
