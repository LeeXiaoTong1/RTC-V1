"""Retire pre-V3.12 resume/intermediate weights, preserving all best models.

Only reviewed full-detector formats and completed V3.10/11 baseline fallbacks
are eligible. No audio/cache traversal or recursive deletion. Run with the old
working training environment (sdd); no GPU or new OmniASR packages are needed.
"""
import argparse
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import re
import shutil

import cleanup_early_checkpoints as early
from audit_w2v_storage import load, path_values
from w2v_v316.cleanup import idle_lock, safe, sha


FULL = {n: 'rtc_w2v_multiconv_v3' + (str(n) if n else '') for n in range(6)}
EARLY_AASIST = 'w2v_aasist_full_20260930_094306_c614'
INTERMEDIATE = re.compile(r'(?:last|latest|checkpoint_last|(?:epoch|checkpoint|ckpt|snapshot|model_epoch)[_-]?\d+(?:[_-]step[_-]?\d+)?)\.(?:pt|pth|ckpt)$')
# These directory-valued fields select the preserved best/metadata in the
# reviewed pre-V3.12 source loaders, not last.pt. An explicit last.pt anywhere
# (including fingerprint dict keys) still pins that file. Unknown fields pin it.
BEST_RUN_FIELDS = {'source_run', 'v37_run', 'run', 'out', 'output_dir'}


def minor(run_name):
    m = re.fullmatch(r'w2v_v3(\d*)_\d{8}_\d{6}_[0-9a-f]+', run_name)
    return int(m[1] or '0') if m else None


def owner(path, root):
    rel = path.relative_to(root/'exp')
    return rel.parts[0]


def allowed(path, root):
    rel = path.relative_to(root/'exp')
    if len(rel.parts) < 2 or any('best' in s.lower() for s in rel.parts[1:]):
        return False
    n = minor(rel.parts[0])
    if not ((n is not None and n < 12) or rel.parts[0] == EARLY_AASIST):
        return False
    # Do not touch arbitrary nested stage payloads or cached tensors.
    if len(rel.parts) != 2 and not (len(rel.parts) == 3 and n == 3 and rel.parts[1] in ('control', 'candidate')):
        return False
    return bool(INTERMEDIATE.fullmatch(path.name))


def identity(path):
    s = path.stat()
    return (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino, s.st_dev, s.st_nlink)


def text_leaves(obj, key=''):
    if isinstance(obj, str):
        yield key, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield '__key__', k
                yield from text_leaves(v, k)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from text_leaves(v, key)


def read_metadata(path, kind):
    if kind == 'json':
        return list(text_leaves(load(path)))
    import torch
    try:
        # mmap reads tensor descriptors without pulling multi-GB tensors into
        # RAM. Never use weights_only=False for a metadata/dependency scan.
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        return list(text_leaves(state))
    except Exception:
        # Old NumPy RNG encodings can be rejected by weights_only. The fallback
        # executes nothing; opaque directory references conservatively pin.
        return [('__opaque__', s) for s in early.checkpoint_strings(path)]


def verified_best(target, root):
    n = minor(owner(target, root))
    if list(target.parent.glob('*.previous')) or list(target.parent.glob('*.tmp')):
        raise ValueError('Unfinished checkpoint transaction; keep resume')
    import torch
    state = torch.load(target, map_location='cpu', weights_only=True, mmap=True)
    if n in FULL or owner(target, root) == EARLY_AASIST:
        schema = FULL.get(n, 'rtc_w2v_aasist_full_v1')
        if state.get('schema') != schema:
            raise ValueError('Unrecognized resume schema')
        # An embedded best must not be removed merely because its file is last.
        if any(k in state for k in ('best_model', 'best_states', 'selected_states')):
            raise ValueError('Embedded best state requires retention')
        best = safe(target.parent/'best_model.pt', root)
        if n in FULL:
            info = early.validate_best(best, schema)
        else:
            from cleanup_for_v318 import validate_selected
            info = validate_selected(best, schema)
        return {str(best): info['sha256']}
    if n in (10, 11):
        module = importlib.import_module('w2v_v3'+str(n)+'.state')
        best, done = module.load_selected(target.parent)
        # These historical runs exported a baseline fallback. A trained partial
        # winner is intentionally not removed by this narrowly reviewed route.
        if (best.get('model') is not None or not done.get('baseline_fallback') or
                state.get('schema') != module.SCHEMA or
                state.get('identity') != best.get('identity') or
                state.get('selected') != best.get('selected') or
                state.get('best_model') is not None):
            raise ValueError('Selected partial model is not an exported baseline fallback')
        cfg = best['config']
        base = Path(cfg['base_checkpoint']).resolve()
        if sha(base) != cfg['base_checkpoint_sha256']:
            raise ValueError('Preserved best base checkpoint identity differs')
        return {str(target.parent/'best.pt'): sha(target.parent/'best.pt'),
                str(target.parent/'completed.json'): sha(target.parent/'completed.json'),
                str(base): cfg['base_checkpoint_sha256']}
    raise ValueError('Compact/unknown format retained; no reviewed independent best')


def dependencies(root, files, candidates):
    found = {p: set() for p in candidates}
    checked = {}
    names = {p.name for p in candidates}
    parents = {p.parent for p in candidates}
    run_names = {owner(p, root) for p in candidates}
    for path, kind in files:
        if path in candidates:
            continue
        before = identity(path)
        for key, value in read_metadata(path, kind):
            tail = value.replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]
            if tail not in names and tail not in ('.', '..', '') and not any(n in value for n in run_names):
                continue
            for ref in path_values(value, root, path.parent):
                if ref in found:
                    found[ref].add(str(path))
                if ref in parents:
                    # A known run's source_run resolves through best loaders.
                    # No such assumption is made for custom/unknown consumers.
                    known_owner = bool(re.fullmatch(r'w2v_v3\d*(?:_tfcl)?_\d{8}_\d{6}_[0-9a-f]+', owner(path, root)))
                    if key not in BEST_RUN_FIELDS or not known_owner:
                        for p in candidates:
                            if p.parent == ref and p.parent != path.parent:
                                found[p].add(str(path)+' [directory reference: '+key+']')
        if identity(path) != before:
            raise ValueError('Metadata changed while checking: '+str(path))
        checked[str(path)] = before
    return found, checked


def plan(root, validator=verified_best):
    root = Path(root).resolve()
    files = sorted(early.metadata_files(root))
    candidates, kept, hashes, checked = [], [], {}, {}
    for path, kind in files:
        if kind != 'weight' or not allowed(path, root):
            continue
        before = identity(path)
        try:
            safe(path, root)
            if not path.is_file() or path.stat().st_nlink != 1:
                raise ValueError('Linked/nonregular checkpoint')
            print('Checking old checkpoint: '+str(path.relative_to(root)), flush=True)
            protected = validator(path, root)
            if identity(path) != before:
                raise ValueError('Checkpoint changed during best verification')
            hashes.update(protected)
            candidates.append(path)
        except Exception as exc:
            kept.append(dict(path=str(path), reason=str(exc)))
    # Fixed point: when a candidate is retained, its own base references become
    # live, too. This prevents deleting dependencies of a retained last.pt.
    while candidates:
        print('Checking retained checkpoint/JSON dependencies...', flush=True)
        refs, seen = dependencies(root, files, candidates)
        checked.update(seen)
        pinned = {p for p in candidates if refs[p] or str(p) in hashes}
        if not pinned:
            break
        kept.extend(dict(path=str(p), reason='Required by retained checkpoint/metadata',
                         references=sorted(refs[p])) for p in sorted(pinned))
        candidates = [p for p in candidates if p not in pinned]
    items = [dict(path=str(p), identity=identity(p)) for p in candidates]
    return dict(root=str(root), candidates=items, retained=kept, protected_hashes=hashes,
                checked_files=checked, metadata_files=[str(p) for p, _ in files],
                selected_bytes=sum(x['identity'][0] for x in items), removed=[])


def write_report(path, result):
    # Report is not model-selection metadata and is excluded from future scans.
    with path.open('w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())


def apply(result, report=None):
    root = Path(result['root']).resolve()
    if [str(p) for p, _ in sorted(early.metadata_files(root))] != result['metadata_files']:
        raise ValueError('Experiment metadata inventory changed; cancelled')
    for value, old in result['checked_files'].items():
        if identity(safe(value, root)) != tuple(old):
            raise ValueError('Dependency changed; cancelled: '+value)
    for value, digest in result['protected_hashes'].items():
        if sha(value) != digest:
            raise ValueError('Best/base changed; cancelled: '+value)
    for item in result['candidates']:
        path = safe(item['path'], root)
        if (not allowed(path, root) or identity(path) != tuple(item['identity']) or
                path.stat().st_nlink != 1 or str(path) in result['protected_hashes']):
            raise ValueError('Deletion scope/identity changed; cancelled')
    free_before = shutil.disk_usage(root).free
    for item in result['candidates']:
        path = safe(item['path'], root)
        if identity(path) != tuple(item['identity']):
            raise ValueError('Candidate changed immediately before unlink')
        path.unlink()
        result['removed'].append(item)
        if report:
            write_report(report, result)
        print('DELETED='+str(path), flush=True)
    result['deleted_file_bytes'] = sum(x['identity'][0] for x in result['removed'])
    result['free_bytes_change'] = shutil.disk_usage(root).free - free_before
    if report:
        write_report(report, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(Path(__file__).resolve().parent))
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    root = Path(args.root).resolve()
    def work():
        result = plan(root)
        report = root/'exp'/('cleanup_before_v312_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        write_report(report, result)
        print(f'CLEANUP_REPORT={report}\nSELECTED_GiB={result["selected_bytes"]/2**30:.3f}', flush=True)
        for item in result['retained']:
            print('KEPT='+item['path']+'; '+item['reason'], flush=True)
        if args.apply:
            apply(result, report)
            print(f'FILES_DELETED={len(result["removed"])}\nDELETED_FILE_GiB={result["deleted_file_bytes"]/2**30:.3f}\nFILESYSTEM_FREE_CHANGE_GiB={result["free_bytes_change"]/2**30:.3f}', flush=True)
    if args.apply:
        with idle_lock(root):
            work()
    else:
        work()


if __name__ == '__main__':
    main()
