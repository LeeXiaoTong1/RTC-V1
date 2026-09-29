"""Retain a few verified best checkpoints. Never delete experiment weights or audio."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import os
import pickle
import torch
from .launch import BASELINE, BASELINE_SHA256, ROOT, run_lock
from .runtime import atomic_json, sha256


def describe(path):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    schemas = {'rtc_w2v_rebuild_v1': 'aasist', 'rtc_w2v_aasist_full_v1': 'aasist',
               'rtc_w2v_multiconv_v1': 'multiconv'}
    family = schemas.get(state.get('schema'))
    dev = state.get('dev', {})
    score = dev.get('weighted_f1', dev.get('robust_f1'))
    if family is None or not isinstance(score, (float, int)) or not 0 <= score <= 1:
        return None
    return {'source': str(Path(path).resolve()), 'family': family,
            'epoch': state.get('epoch'), 'weighted_dev_proxy': float(score),
            'bytes': Path(path).stat().st_size, 'schema': state['schema']}


def retain(baseline, roots, destination, apply=False, expected_sha=BASELINE_SHA256):
    baseline, destination = Path(baseline).resolve(), Path(destination).resolve()
    if sha256(baseline) != expected_sha:
        raise ValueError('The original 91.68 checkpoint identity differs; retention stopped')
    rows, skipped = [], []
    paths = {baseline}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for name in ('best_model.pt', 'candidate_best.pt', 'best_noisy.pt'):
            paths.update(p.resolve() for p in root.rglob(name)
                         if destination not in p.resolve().parents)
    for path in sorted(paths):
        try:
            row = describe(path)
            if row:
                rows.append(row)
        except (ValueError, RuntimeError, OSError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            skipped.append({'path': str(path), 'error': type(exc).__name__ + ': ' + str(exc)[:160]})
    initial = next((r for r in rows if r['source'] == str(baseline)), None)
    if initial is None:
        raise ValueError('Cannot read the original best metadata')
    initial = {**initial, 'tag': 'original_91_68', 'sha256': expected_sha}
    selected, digests = [initial], {expected_sha}
    for family, count in (('multiconv', 1), ('aasist', 2)):
        choices = sorted([r for r in rows if r['family'] == family],
                         key=lambda r: (-r['weighted_dev_proxy'], r['source']))
        accepted = 0
        for row in choices:
            print('Checking retained candidate: ' + row['source'], flush=True)
            digest = sha256(row['source'])
            if digest in digests:
                continue
            digests.add(digest)
            accepted += 1
            selected.append({**row, 'sha256': digest, 'tag': f'{family}_candidate_{accepted}'})
            if accepted >= count:
                break
    if apply:
        destination.mkdir(parents=True, exist_ok=True)
        with run_lock(destination / '.retention.lock'):
            for row in selected:
                target = destination / (row['tag'] + '_' + row['sha256'][:16] + '.pt')
                if target.exists():
                    if sha256(target) != row['sha256']:
                        raise ValueError('Existing retained copy has a different hash')
                else:
                    if shutil.disk_usage(destination).free < row['bytes'] + 1024**3:
                        raise OSError('Insufficient room to preserve checkpoint; no deletion is attempted')
                    temp = target.with_suffix('.pt.tmp')
                    if temp.exists():
                        raise FileExistsError('Inspect interrupted retention copy: ' + str(temp))
                    print('Preserving ' + row['tag'], flush=True)
                    try:
                        shutil.copyfile(row['source'], temp)
                        if sha256(temp) != row['sha256']:
                            raise RuntimeError('Source changed during copy')
                        os.replace(temp, target)
                    finally:
                        if temp.exists():
                            temp.unlink()
                row['retained_copy'] = str(target)
            atomic_json(destination / 'catalog.json', {
                'created': datetime.now().isoformat(), 'protected': selected, 'skipped': skipped,
                'selection': 'Original platform 91.68 pinned; other candidates ranked WITHIN architecture by recorded Dev proxy',
                'deleted_checkpoints': [], 'source_checkpoints_unchanged': True})
    print(json.dumps({'apply': apply, 'retained': selected, 'skipped_count': len(skipped)}, indent=2), flush=True)
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default=BASELINE)
    p.add_argument('--exp-root', action='append')
    p.add_argument('--destination', default=str(ROOT / 'checkpoints' / 'retained'))
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    roots = args.exp_root or [ROOT / 'exp', Path(args.baseline).parents[2]]
    retain(args.baseline, roots, args.destination, args.apply)


if __name__ == '__main__':
    main()
