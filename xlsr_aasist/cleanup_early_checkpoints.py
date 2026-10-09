"""Retire ONLY two reviewed old resume checkpoints; keep every best and cache.

Run in the original training environment. --apply performs live dependency and
standalone-best checks first. No globs are ever used as deletion targets.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickletools
import zipfile

from audit_w2v_storage import load, path_values
from w2v_v316.cleanup import idle_lock, safe, sha


TARGETS = {
    'w2v_v34_20261003_035410_4f34': 'rtc_w2v_multiconv_v34',
    'w2v_v32_20261001_172251_00cd': 'rtc_w2v_multiconv_v32',
}
# These are inventories, not runtime model-selection records. All best/selection
# JSON and config files are still checked, including those in nested stage dirs.
INVENTORIES = ('cache_retirement_', 'checkpoint_cleanup_', 'cleanup_',
               'storage_audit_', 'maintenance_', 'retirement_')
ROW_LISTS = {'rows.json', 'train_rows.json', 'dev_rows.json', 'records.json'}
WEIGHTS = {'.pt', '.pth', '.ckpt'}


def stamp(path):
    s = path.stat()
    return (s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev)


def checkpoint_strings(path):
    """Read only pickle metadata, never execute pickle or read tensor payloads."""
    if not zipfile.is_zipfile(path):
        raise ValueError('Cannot inspect legacy checkpoint format: '+str(path))
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl') or n == 'data.pkl']
        if len(names) != 1: raise ValueError('Ambiguous checkpoint metadata: '+str(path))
        if archive.getinfo(names[0]).file_size > 64*1024**2:
            raise ValueError('Checkpoint metadata too large: '+str(path))
        for op, value, _ in pickletools.genops(archive.read(names[0])):
            if isinstance(value, str) and op.name in ('UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8'):
                yield value


def metadata_files(root):
    """Walk experiment metadata, never audio trees, feature arrays or pair caches."""
    for current, dirs, names in os.walk(root/'exp', followlinks=False):
        parent = Path(current)
        dirs[:] = [n for n in dirs if n not in ('features','rolling_pairs','__pycache__','.git')
                   and not (parent/n).is_symlink()]
        for name in names:
            p = parent/name
            if p.is_symlink(): raise ValueError('Linked experiment metadata needs review: '+str(p))
            if p.suffix in WEIGHTS: yield p, 'weight'
            elif p.suffix == '.json' and not name.startswith(INVENTORIES) and name not in ROW_LISTS:
                yield p, 'json'


def references(root, targets):
    found = {p: set() for p in targets}
    checked = {}
    for path, kind in metadata_files(root):
        if path in targets: continue  # discarded resume state's own config is not a consumer
        before = stamp(path)
        values = checkpoint_strings(path) if kind == 'weight' else [load(path)]
        for value in values:
            for candidate in path_values(value, root, path.parent):
                for target in targets:
                    # Explicit file references always protect. A run-directory
                    # reference in ANOTHER run may select its last implicitly.
                    if candidate == target or (candidate == target.parent and path.parent != target.parent):
                        found[target].add(str(path))
        if stamp(path) != before: raise ValueError('Metadata changed during check: '+str(path))
        checked[str(path)] = before
    return found, checked


def validate_best(path, schema):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (not isinstance(state, dict) or state.get('schema') != schema or state.get('kind') != 'weights'
            or not all(k in state for k in ('model','model_config','head_config'))):
        raise ValueError('Best is not an independent weights checkpoint: '+str(path))
    weights = state['model']
    if (not isinstance(weights, dict) or not weights or
            not all(isinstance(v, torch.Tensor) for v in weights.values()) or
            not any(k.startswith('backbone.') for k in weights) or
            not any(k.startswith('head.') for k in weights)):
        raise ValueError('Best lacks encoder or head tensors: '+str(path))
    # Exact key/shape check against the architecture without allocating another
    # full encoder, accessing the GPU, or downloading pretrained parameters.
    from w2v_v3.model import Detector
    with torch.device('meta'):
        expected = Detector.from_config(state['model_config'],state['head_config'],checkpointing=False).state_dict()
    if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in expected):
        raise ValueError('Best tensors do not match its complete architecture: '+str(path))
    return {'tag': state.get('tag'), 'schema': schema, 'sha256': sha(path)}


def plan(root, best_validator=validate_best):
    root = Path(root).resolve()
    targets = []
    missing = []
    bests = {}
    for run, schema in TARGETS.items():
        target = safe(root/'exp'/run/'last.pt', root)
        if not target.exists(): missing.append(str(target)); continue
        if not target.is_file() or target.stat().st_nlink != 1:
            raise ValueError('Candidate is not an unlinked regular checkpoint: '+str(target))
        best = safe(target.parent/'best_model.pt', root)
        if not best.is_file(): raise ValueError('Separate best_model.pt is missing: '+str(best))
        if list(target.parent.glob('*.previous')):
            raise ValueError('Uncommitted checkpoint promotion must be recovered first: '+str(target.parent))
        print('Checking standalone best: '+str(best), flush=True)
        before = stamp(best)
        bests[str(best)] = best_validator(best, schema)
        if stamp(best) != before: raise ValueError('Best changed during validation')
        targets.append(target)
    print('Checking experiment JSON and checkpoint metadata (no tensor payload scan)...',flush=True)
    refs, checked = references(root, targets) if targets else ({}, {})
    candidates = []
    retained = []
    for target in targets:
        if refs[target]: retained.append({'path':str(target),'referenced_by':sorted(refs[target])})
        else: candidates.append({'path':str(target),'identity':stamp(target)})
    return dict(root=str(root),candidates=candidates,retained=retained,already_absent=missing,
                checked_files=checked,verified_best=bests,
                total_bytes=sum(item['identity'][0] for item in candidates))


def apply(result):
    root = Path(result['root']).resolve()
    allowed = {root/'exp'/run/'last.pt' for run in TARGETS}
    for value, identity in result['checked_files'].items():
        if stamp(Path(value)) != tuple(identity): raise ValueError('Dependency changed; cancelled: '+value)
    for value, info in result['verified_best'].items():
        if sha(value) != info['sha256']: raise ValueError('Best changed; cancelled: '+value)
    # Validate every target before the first unlink; there is no recursive delete.
    for item in result['candidates']:
        path = safe(item['path'], root)
        if path not in allowed or stamp(path) != tuple(item['identity']):
            raise ValueError('Candidate changed or is outside the two-file allowlist')
    removed = []
    for item in result['candidates']:
        Path(item['path']).unlink()
        removed.append(item['path'])
        print('DELETED='+item['path'],flush=True)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    def work():
        result = plan(root)
        report = root/'exp'/('cleanup_early_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        with report.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
        print(f'CLEANUP_REPORT={report}\nSELECTED_GiB={result["total_bytes"]/1024**3:.3f}',flush=True)
        for item in result['retained']: print('KEPT_DEPENDENCY='+item['path'],flush=True)
        if args.apply:
            removed = apply(result)
            result['removed'] = removed
            report.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(f'FILES_DELETED={len(removed)}\nFREED_GiB={result["total_bytes"]/1024**3:.3f}',flush=True)
    if args.apply:
        with idle_lock(root): work()
    else: work()


if __name__ == '__main__': main()
