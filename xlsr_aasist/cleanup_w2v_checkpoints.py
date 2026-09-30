"""Remove recognizable intermediate checkpoints; preserve every named best and candidate."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re

from w2v_aasist.launch import BASELINE, BASELINE_SHA256, ROOT, run_lock
from w2v_aasist.runtime import sha256, atomic_json


def running_training():
    """Conservative Linux check for this user's Python training/launcher processes."""
    found = []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if int(entry.name) == os.getpid() or entry.stat().st_uid != os.getuid():
                continue
            parts = (entry/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if not parts or not Path(parts[0]).name.startswith('python'):
                continue
            if any(('train' in x.lower() or 'launch' in x.lower()) for x in parts[1:]):
                found.append({'pid': int(entry.name), 'command': ' '.join(parts)})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def intermediate(path):
    name = path.name.lower()
    if any('best' in x.lower() or 'candidate' in x.lower() for x in path.parts):
        return False
    if path.suffix.lower() not in {'.pt', '.pth', '.ckpt'}:
        return False
    stem = path.stem.lower()
    return stem in {'last', 'latest', 'last_model', 'latest_model', 'checkpoint_last'} or bool(
        re.match(r'^(?:epoch|checkpoint|ckpt|snapshot|model_epoch)[_-]?\d', stem))


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def plan(roots, pinned):
    pinned = {Path(p).resolve() for p in pinned}
    protected_ids = {tuple(identity(p)[:2]) for p in pinned if p.is_file()}
    candidates, seen, skipped = [], set(), []
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        # Require a nonempty named best in the SAME experiment before deleting its intermediates.
        paths = []
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(folder)/d).is_symlink()
                       and d not in {'retained', 'data', 'dataset', 'pretrained'}]
            for name in names:
                p = Path(folder)/name
                if p.is_symlink() or not p.is_file():
                    continue
                p.resolve().relative_to(root)
                paths.append(p)
        best_runs = {p.relative_to(root).parts[0] for p in paths
                     if 'best' in p.name.lower() and p.suffix.lower() in {'.pt','.pth','.ckpt'}
                     and p.stat().st_size > 0 and len(p.relative_to(root).parts) > 1}
        for p in paths:
            if not intermediate(p) or p.resolve() in pinned or p.resolve() in seen:
                continue
            info = identity(p)
            if tuple(info[:2]) in protected_ids:
                continue
            if p.relative_to(root).parts[0] not in best_runs:
                skipped.append(str(p))
                continue
            seen.add(p.resolve())
            candidates.append({'path': str(p.resolve()), 'root': str(root), 'identity': info})
    return candidates, skipped


def remove(planned):
    # Check every final target before deleting any; never recurse over directories.
    for item in planned:
        p = Path(item['path'])
        p.resolve().relative_to(Path(item['root']))
        if p.is_symlink() or not intermediate(p) or identity(p) != item['identity']:
            raise RuntimeError('Target changed; cleanup stopped: ' + str(p))
    for item in planned:
        p = Path(item['path'])
        if p.is_symlink() or identity(p) != item['identity']:
            raise RuntimeError('Target changed during cleanup: ' + str(p))
        p.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Delete the listed intermediate weights')
    args = parser.parse_args()
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Run this server cleanup on Linux')
    baseline = Path(BASELINE)
    print('Verifying protected original 91.68 checkpoint...', flush=True)
    if sha256(baseline) != BASELINE_SHA256:
        raise RuntimeError('Original best identity differs; nothing deleted')
    roots = [ROOT/'exp', baseline.parents[2]]
    pinned = [baseline]
    catalog = ROOT/'checkpoints'/'retained'/'catalog.json'
    if catalog.is_file():
        for row in json.loads(catalog.read_text(encoding='utf-8'))['protected']:
            pinned.extend(Path(row[k]) for k in ('source','retained_copy') if row.get(k))
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        active = running_training()
        if active:
            raise RuntimeError('Training/launcher is running; nothing deleted: ' + json.dumps(active))
        planned, skipped = plan(roots, pinned)
        total = sum(x['identity'][2] for x in planned)
        for row in planned:
            print(f'{row["identity"][2]/1024**3:.2f} GiB  {row["path"]}', flush=True)
        print(f'PLANNED_FILES={len(planned)} LOGICAL_GiB={total/1024**3:.2f} SKIPPED_WITHOUT_NAMED_BEST={len(skipped)}', flush=True)
        if not args.apply:
            print('PREVIEW_ONLY=True; pass --apply to remove these files.')
            return
        log = ROOT/'exp'/('checkpoint_cleanup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        report = {'status':'planned', 'baseline_sha256':BASELINE_SHA256, 'files':planned,
                  'skipped_without_named_best':skipped, 'audio_or_cache_deleted':False}
        atomic_json(log, report)
        if running_training():
            raise RuntimeError('Training started; nothing deleted')
        remove(planned)
        unchanged = sha256(baseline) == BASELINE_SHA256
        atomic_json(log, dict(report, status='complete', original_best_unchanged=unchanged))
        print(f'DELETED_FILES={len(planned)} ORIGINAL_BEST_UNCHANGED={unchanged}\nREPORT={log}')
        import shutil
        print(f'FREE_GiB={shutil.disk_usage(ROOT).free/1024**3:.2f}')
        if not unchanged:
            raise RuntimeError('Original best changed externally; inspect report')


if __name__ == '__main__':
    main()
