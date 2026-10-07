"""Compact a completed V3.14 run; preserve all three export choices and sources."""
import argparse
from pathlib import Path
import shutil

import torch

from w2v_v39.common import ROOT, atomic_json, digest, read_json, verify_files
from .state import KINDS, load_selected, atomic_save, tensor_bytes


def cleanup(run, apply=False, remove_features=False):
    requested = Path(run).expanduser()
    if requested.is_symlink():
        raise ValueError('Refusing linked run')
    run = requested.resolve()
    _, done = load_selected(run)
    cfg = read_json(run/'config.json')
    protected = dict(cfg['source_fingerprints'], **cfg['data_fingerprints'])
    protected.update({cfg['base_checkpoint']:cfg['base_checkpoint_sha256'], cfg['starting_checkpoint']:cfg['starting_checkpoint_sha256']})
    verify_files(protected)
    protected_paths = {Path(p).resolve() for p in protected}
    path = Path(done['state_file'])
    if (run/path).resolve() in protected_paths:
        raise ValueError('Cannot compact a protected input')
    targets = [run/'last.pt', run/'inference.pt', run/'features']
    if any(p.is_symlink() or p.resolve().parent != run for p in targets):
        raise ValueError('Owned output paths must remain inside the completed run')
    if any(p.resolve() in protected_paths for p in targets[:2]):
        raise ValueError('Cannot replace or remove a protected checkpoint')
    features = run/'features'
    if remove_features and features.exists():
        from .features import FORMAT
        from .state import identity
        if set(p.name for p in features.iterdir()) != {'train', 'dev'}:
            raise ValueError('Refusing unexpected feature directory contents')
        for split in ('train', 'dev'):
            allowed = {'owner.json', 'complete.json', 'cursor.json', 'rows.json', 'x.npy', 'logits.npy', '.extract.lock'}
            if any(p.name not in allowed or not p.is_file() for p in (features/split).iterdir()):
                raise ValueError('Refusing unexpected files inside owned feature cache')
            owner = read_json(features/split/'owner.json')
            if owner.get('format') != FORMAT or owner.get('identity', {}).get('config_identity') != identity(cfg):
                raise ValueError('Feature cache not owned by this completed V3.14 run')
        for child in features.rglob('*'):
            resolved = child.resolve()
            if child.is_symlink() or not resolved.is_relative_to(features.resolve()) or resolved in protected_paths:
                raise ValueError('Refusing linked/external/protected feature content')
    state = torch.load(run/path, map_location='cpu', weights_only=True)
    slim = {k:v for k,v in state.items() if k not in ('optimizer', 'rng')}
    slim['inference_only'] = True
    estimate = max(0, (run/path).stat().st_size-int(1.1*tensor_bytes(slim)))
    if remove_features and features.exists():
        estimate += sum(p.stat().st_size for p in features.rglob('*') if p.is_file())
    print(f'V314_CLEANUP apply={apply}; approximate_reclaim_GiB={estimate/1024**3:.2f}; all export choices retained', flush=True)
    if apply:
        # Write a second file, then atomically redirect the manifest. A crash
        # before or after either commit leaves at least one valid export state.
        if done['state_file'] != 'inference.pt':
            atomic_save(run/'inference.pt', slim, cfg['disk_margin_bytes'])
            done.update(state_file='inference.pt', checkpoint_sha256=digest(run/'inference.pt'), inference_only=True)
            atomic_json(run/'completed.json', done)
        for kind in KINDS:
            load_selected(run, kind)
            alias = read_json(run/(kind+'.json'))
            alias['checkpoint'] = 'inference.pt'
            atomic_json(run/(kind+'.json'), alias)
        (run/'last.pt').unlink(missing_ok=True)
        if remove_features and features.exists():
            shutil.rmtree(features)
        verify_files(protected)
    return estimate


def main():
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--remove-features', action='store_true')
    args = p.parse_args()
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        cleanup(args.run, args.apply, args.remove_features)


if __name__ == '__main__':
    main()
