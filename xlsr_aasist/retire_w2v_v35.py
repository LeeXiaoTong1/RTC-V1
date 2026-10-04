"""Retire one stopped V3.5 run's resume state and owned training audio caches."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

import torch
from w2v_aasist.launch import ROOT, run_lock
from w2v_v35 import SCHEMA
from w2v_v35.cache import FORMAT, _root
from w2v_v35.maintenance import active_jobs, atomic_json, identity, json_read, sha256
from rtc_noisy_v2.plan import digest_json

GIB = 1024**3


def safe(path, root):
    """Reject traversal, every symlink/junction ancestor, and resolved escapes."""
    root, path = Path(root).absolute(), Path(path).absolute()
    relative = path.relative_to(root)
    if '..' in relative.parts:
        raise ValueError('Traversal is not allowed')
    for entry in [path, *path.parents]:
        if entry.is_symlink() or (entry.exists() and
                getattr(entry.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('Symlink/junction is not allowed: ' + str(entry))
    path.resolve().relative_to(root.resolve())
    return path


def guard(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Required protection file is missing: ' + str(path))
    # Read-only HF snapshot metadata may legitimately link to cache blobs.
    # Deletion candidates, cache trees and best weights use safe() separately.
    return dict(path=str(path), resolved=str(path.resolve(strict=True)),
                identity=identity(path), sha256=sha256(path))


def read_inventory(path):
    if Path(path).stat().st_size > 256*1024**2:
        raise ValueError('Unexpectedly large source inventory')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def idle(check):
    jobs = check()
    if jobs:
        raise RuntimeError('Training/evaluation/cache job is active; nothing removed: ' + json.dumps(jobs))


def check_weights(path, cfg):
    """Validate a standalone, complete MultiConv state without allocating a model."""
    from w2v_v33.model import Detector
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (not isinstance(state, dict) or state.get('schema') != SCHEMA
            or state.get('kind') != 'weights' or state.get('config') != cfg
            or state.get('reference_checkpoint_sha256') != cfg['reference_checkpoint_sha256']):
        raise ValueError('Standalone best checkpoint identity differs: ' + str(path))
    if not state.get('model_config') or not state.get('head_config') or not state.get('model'):
        raise ValueError('Best checkpoint lacks architecture or model weights')
    with torch.device('meta'):
        expected = Detector.from_config(state['model_config'], state['head_config'], checkpointing=False).state_dict()
    if set(expected) != set(state['model']) or any(
            not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
            for name, value in state['model'].items()):
        raise ValueError('Best checkpoint is not a complete model state')
    if any(k in state for k in ('optimizer', 'ema')):
        raise ValueError('Expected standalone weights, not training state')
    return state


def cache_inventory(cfg, root):
    # Validate the lexical path before cache._root resolves it.
    raw = Path(cfg['train_cache_root']).expanduser()
    cache = safe(raw if raw.is_absolute() else root/raw, root/'data')
    if cache != _root(cfg, 'train'):
        raise ValueError('Unexpected training cache root')
    if cache == root/'data':
        raise ValueError('The data root cannot be a cache retirement target')
    owner = json_read(cache/'owner.json')
    if owner != dict(format=FORMAT, role='train', root=str(cache)):
        raise ValueError('Unrecognized cache owner')
    recipe = json_read(cache/'recipe.json')
    # Real Train inventories exceed the small-config reader's 16 MiB limit.
    sources = read_inventory(cache/'sources.json')
    if (recipe.get('format') != FORMAT or recipe.get('role') != 'train'
            or recipe.get('source_digest') != digest_json(sources)
            or recipe.get('protocol_sha256') != sha256(cfg['train_protocol'])):
        raise ValueError('Cache source metadata or recipe differs')
    keys = {hashlib.sha256(key.encode()).hexdigest(): key
            for key, row in sources.items() if row['domain'] == 'offline'}
    files, folders = [], []
    recipe_sha = sha256(cache/'recipe.json')
    for folder in sorted(cache.iterdir()):
        safe(folder, cache)
        if folder.name in {'owner.json', 'recipe.json', 'sources.json', '.prepare.lock'}:
            continue
        if not re.fullmatch(r'epoch_\d{3,}', folder.name) or not folder.is_dir():
            raise ValueError('Unknown cache entry; nothing removed: ' + str(folder))
        marker = json_read(folder/'generation.json')
        epoch = marker.get('epoch')
        if (type(epoch) is not int or folder.name != f'epoch_{epoch:03d}' or marker !=
                dict(format=FORMAT, role='train', epoch=epoch, recipe_sha256=recipe_sha, root=str(cache))):
            raise ValueError('Unrecognized cache generation')
        directories = [folder]
        for parent, dirs, names in os.walk(folder, followlinks=False):
            for name in dirs:
                child = safe(Path(parent)/name, folder)
                rel = child.relative_to(folder).parts
                if not (rel == ('audio',) or len(rel) == 2 and rel[0] == 'audio' and re.fullmatch('[0-9a-f]{2}', rel[1])):
                    raise ValueError('Unknown cache directory: ' + str(child))
                directories.append(child)
            for name in names:
                child = safe(Path(parent)/name, folder)
                rel = child.relative_to(folder).parts
                if len(rel) == 1 and name in {'generation.json', '.active.lock'}:
                    pass
                elif len(rel) == 3 and rel[0] == 'audio':
                    match = re.fullmatch(r'([0-9a-f]{64})(?:\.lock|_noisy_[ab]\.(?:wav|tmp\.wav|json|json\.tmp))', name)
                    if not match or match[1] not in keys or rel[1] != match[1][:2]:
                        raise ValueError('Unknown cache file: ' + str(child))
                    if name.endswith('.json'):
                        saved = json_read(child)
                        source = keys[match[1]]
                        if any(saved.get(k) != v for k, v in dict(format=FORMAT, role='train', epoch=epoch,
                                source=source, source_sha256=sources[source]['sha256'], recipe_sha256=recipe_sha).items()):
                            raise ValueError('Cache sidecar ownership differs')
                else:
                    raise ValueError('Unknown cache file: ' + str(child))
                if not child.is_file():
                    raise ValueError('Cache entry is not a regular file')
                files.append(dict(path=str(child), identity=identity(child), kind='train_cache'))
        folders.extend(str(p) for p in sorted(directories, key=lambda p: len(p.parts), reverse=True))
    return cache, files, folders


def build_plan(root, run, idle_check):
    root = Path(root).absolute()
    run = safe(run, root/'exp')
    if run.parent != root/'exp' or not re.fullmatch(r'w2v_v35_[A-Za-z0-9_]+', run.name):
        raise ValueError('Select one direct exp/w2v_v35_* run')
    idle(idle_check)
    if list(run.glob('*.previous')):
        raise ValueError('Uncommitted winner transaction (.previous) exists; preserve it and resolve recovery first')
    cfg = json_read(safe(run/'config.json', run))
    if cfg.get('version') != '3.5' or Path(cfg.get('run_dir', run)).resolve() != run.resolve():
        raise ValueError('Run configuration identity differs')
    protected = dict(cfg.get('source_provenance', {}).get('file_fingerprints', {}))
    for key in ('baseline', 'reference_checkpoint'):
        protected[cfg[key]] = cfg[key+'_sha256']
    guards = [guard(run/'config.json')]
    for path, digest in protected.items():
        item = guard(path)
        if item['sha256'] != digest:
            raise ValueError('Protected original/submitted checkpoint or metadata differs: ' + str(path))
        guards.append(item)
    best = sorted(run.glob('best*.pt'))
    if not (run/'best_model.pt').is_file() or not (run/'best_candidate.pt').is_file():
        raise ValueError('Both selected best and learned best_candidate must exist before retirement')
    data = None
    best_summary = []
    for path in best:
        safe(path, run)
        state = check_weights(path, cfg)
        fingerprints = state.get('data_fingerprints')
        if not fingerprints or data is not None and fingerprints != data:
            raise ValueError('Best checkpoints disagree on source metadata')
        data = fingerprints
        best_summary.append(dict(path=str(path), tag=state.get('tag'), epoch=state.get('epoch')))
        guards.append(guard(path))
        del state
    cache, files, folders = cache_inventory(cfg, root)
    required = {str(cache/name) for name in ('owner.json', 'recipe.json', 'sources.json')}
    if not required <= set(data):
        raise ValueError('Best checkpoint does not identify the training cache metadata')
    for path, digest in data.items():
        item = guard(path)
        if item['sha256'] != digest:
            raise ValueError('Saved source metadata changed: ' + str(path))
        guards.append(item)
    last = safe(run/'last.pt', run)
    if last.exists():
        if not last.is_file():
            raise ValueError('last.pt is not a regular file')
        files.insert(0, dict(path=str(last), identity=identity(last), kind='resume_state'))
    protected_paths = {Path(g['path']).resolve() for g in guards}
    if any(Path(item['path']).resolve() in protected_paths for item in files):
        raise ValueError('A proposed deletion is a protected reference')
    return dict(root=str(root), run=str(run), cache=str(cache), files=files, folders=folders,
                guards=guards, retained_weights=best_summary, logical_bytes=sum(f['identity'][2] for f in files))


def validate(plan):
    root = Path(plan['root'])
    if list(Path(plan['run']).glob('*.previous')):
        raise ValueError('Winner transaction appeared during planning')
    for item in plan['guards']:
        if (str(Path(item['path']).resolve(strict=True)) != item['resolved']
                or identity(item['path']) != item['identity'] or sha256(item['path']) != item['sha256']):
            raise RuntimeError('Protected file changed; no deletion started')
    for item in plan['files']:
        path = safe(item['path'], root)
        if identity(path) != item['identity']:
            raise RuntimeError('Deletion target changed; rebuild the plan')


def apply_plan(plan, idle_check, report):
    idle(idle_check)
    validate(plan)
    root, run = Path(plan['root']), Path(plan['run'])
    free_before = shutil.disk_usage(root).free
    removed = []
    atomic_json(report, dict(plan, status='planned', removed=[]))
    idle(idle_check)
    atomic_json(run/'retired.json', dict(status='retiring', resume_allowed=False,
        note='last.pt and derived Train generations are retired. Start a NEW run from retained weights; do not --resume.',
        retained_weights=plan['retained_weights'], manifest=str(report)))
    try:
        for index, item in enumerate(plan['files']):
            if index % 1000 == 0:
                idle(idle_check)
            path = safe(item['path'], root)
            if identity(path) != item['identity']:
                raise RuntimeError('Target changed during deletion')
            path.unlink()
            removed.append(str(path))
        # Explicit files only: a new/unknown entry causes rmdir to fail, never recursive deletion.
        for folder in plan['folders']:
            safe(folder, root).rmdir()
        status = 'complete'
    finally:
        status = locals().get('status', 'interrupted')
        after = shutil.disk_usage(root).free
        result = dict(plan, status=status, removed=removed, free_before_bytes=free_before,
                      free_after_bytes=after, measured_free_change_bytes=after-free_before)
        atomic_json(report, result)
        atomic_json(run/'retired.json', dict(status=status, resume_allowed=False,
            note='No exact resume remains. Retained best weights can initialize a NEW experiment.',
            retained_weights=plan['retained_weights'], manifest=str(report)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix':
        raise RuntimeError('Production cleanup must run on the Linux training server')
    root = ROOT.resolve()
    run = args.run if args.run.is_absolute() else root/args.run
    with ExitStack() as locks:
        locks.enter_context(run_lock(root/'exp'/'.aasist-launch.lock'))
        idle(lambda: active_jobs(root))
        print('Checking retained best weights, source metadata and owned Train caches; large inventories may take a few minutes...', flush=True)
        plan = build_plan(root, run, lambda: active_jobs(root))
        locks.enter_context(run_lock(Path(plan['cache'])/'.prepare.lock'))
        for folder in plan['folders']:
            if Path(folder).name.startswith('epoch_'):
                locks.enter_context(run_lock(Path(folder)/'.active.lock'))
        # Lock files may have been created: rebuild a complete exact-file manifest.
        plan = build_plan(root, run, lambda: active_jobs(root))
        report = run/('retirement_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.json')
        print(f"FILES={len(plan['files'])} LOGICAL_GiB={plan['logical_bytes']/GIB:.2f}")
        print('DELETE: this run last.pt and owned V3.5 Train epoch caches. KEEP: all best weights, metadata, scores, original audio and fixed Dev.')
        if not args.apply:
            atomic_json(report, dict(plan, status='preview'))
            print('PREVIEW_ONLY=True; add --apply to abandon exact resume and delete the listed files.')
        else:
            result = apply_plan(plan, lambda: active_jobs(root), report)
            print(f"RETIRED=True FREE_GiB={result['free_after_bytes']/GIB:.2f} MEASURED_FREE_CHANGE_GiB={result['measured_free_change_bytes']/GIB:.2f}")
        print('MANIFEST='+str(report))


if __name__ == '__main__':
    main()
