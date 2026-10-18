"""Preview/remove unreferenced intermediate checkpoints; preserve best and last.

Only recognized intermediate filenames below explicitly selected exp roots are
eligible. No model loading, recursive directory deletion, or checkpoint rewrite.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat

WEIGHTS = {'.pt', '.pth', '.ckpt'}
INTERMEDIATE = re.compile(r'(?:epoch|step|checkpoint|ckpt|snapshot|model_epoch)[_-]?\d.*', re.I)
EXCLUDED = {'data', 'dataset', 'wav', 'audio', 'features', 'pretrained', 'assets', '.git', '__pycache__'}


def linked(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def stamp(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from values(item)
    elif isinstance(value, list):
        for item in value:
            yield from values(item)


def named_keep(path, root):
    parts = [part.lower() for part in path.relative_to(root).parts]
    return any(any(token in part for token in ('best', 'last', 'latest', 'retained', 'protected'))
               for part in parts) or path.name.lower() == 'inference.pt'


def check_path(path, root):
    path.relative_to(root)
    path.resolve().relative_to(root)
    for node in (path, *path.parents):
        if node == root:
            break
        if linked(node):
            raise RuntimeError('Linked path; nothing deleted: ' + str(node))
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('Not a regular file: ' + str(path))


def plan(roots):
    roots = [Path(root).expanduser().absolute() for root in roots]
    if len(set(roots)) != len(roots):
        raise ValueError('Duplicate exp roots')
    for root in roots:
        if root.name != 'exp' or not root.is_dir() or root.resolve() != root or linked(root):
            raise ValueError('Use an existing, unlinked absolute exp directory: ' + str(root))
        if any(other != root and other in root.parents for other in roots):
            raise ValueError('Overlapping exp roots')
    records, observed, links, json_paths = [], {}, [], []
    for root in roots:
        for folder, dirs, names in os.walk(root, followlinks=False):
            parent = Path(folder)
            for name in list(dirs):
                node = parent / name
                if linked(node):
                    # A best/last directory alias may point at a numbered epoch.
                    links.append(node.resolve())
                    dirs.remove(name)
                elif name in EXCLUDED:
                    dirs.remove(name)
            for name in names:
                path = parent / name
                if linked(path):
                    links.append(path.resolve())
                    continue
                if path.suffix.lower() in WEIGHTS or path.suffix.lower() == '.json':
                    check_path(path, root)
                    observed[str(path)] = stamp(path)
                if path.suffix.lower() == '.json' and not name.startswith('cleanup_keep_best_last_'):
                    json_paths.append(path)
                elif path.suffix.lower() in WEIGHTS:
                    records.append((path, root))
    referenced_names, tags = set(), set()
    for path in json_paths:
        if path.stat().st_size > 256 * 1024**2:
            raise RuntimeError('Oversized metadata needs review before cleanup: ' + str(path))
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError) as exc:
            raise RuntimeError('Unreadable metadata; nothing deleted: ' + str(path)) from exc
        # Retain matching filenames even when an old absolute path has moved.
        for text in values(data):
            tail = text.replace('\\', '/').rsplit('/', 1)[-1]
            if Path(tail).suffix.lower() in WEIGHTS:
                referenced_names.add(tail.lower())
        def selected_tags(item):
            if isinstance(item, dict):
                for key, value in item.items():
                    if any(word in key.lower() for word in ('best', 'last', 'selected', 'selections')):
                        for text in values(value):
                            tags.add(Path(text.replace('\\', '/')).stem.lower())
                    elif isinstance(value, (dict, list)):
                        selected_tags(value)
            elif isinstance(item, list):
                for value in item:
                    selected_tags(value)
        selected_tags(data)
        if any(word in path.stem.lower() for word in ('best', 'last', 'selection')):
            tags.update(Path(text.replace('\\', '/')).stem.lower() for text in values(data))
    protected, candidates, skipped = [], [], []
    for path, root in records:
        reason = None
        if named_keep(path, root): reason = 'best/last/latest/inference or protected path'
        elif path.name.lower() in referenced_names: reason = 'checkpoint filename referenced by JSON metadata'
        elif path.stem.lower() in tags: reason = 'selected best/last tag'
        elif any(path == target or target in path.parents for target in links): reason = 'symlink/junction target'
        elif path.stat().st_nlink > 1: reason = 'hard-linked checkpoint'
        elif not INTERMEDIATE.fullmatch(path.stem): reason = 'unrecognized filename; retained'
        entry = dict(path=str(path), root=str(root), identity=observed[str(path)])
        if reason:
            protected.append(dict(entry, reason=reason))
        else:
            candidates.append(entry)
    # Never discard numbered snapshots without a surviving nonempty named or
    # referenced checkpoint in the same directory. Unknown files are not proof.
    kept_dirs = {Path(row['path']).parent for row in protected
                 if row['identity'][2] > 0 and row['reason'] != 'unrecognized filename; retained'}
    approved = []
    for row in candidates:
        if Path(row['path']).parent in kept_dirs:
            approved.append(row)
        else:
            skipped.append(dict(row, reason='no surviving checkpoint in this directory'))
    return dict(roots=[str(root) for root in roots], candidates=approved,
        protected=protected, skipped=skipped, observed=observed,
        total_bytes=sum(row['identity'][2] for row in approved))


def apply(result):
    # Complete validation before the first unlink; abort on any changed input.
    for name, identity in result['observed'].items():
        path = Path(name)
        if linked(path) or stamp(path) != identity:
            raise RuntimeError('Input changed; rerun preview: ' + name)
    for row in result['candidates']:
        path, root = Path(row['path']), Path(row['root'])
        check_path(path, root)
        if named_keep(path, root) or not INTERMEDIATE.fullmatch(path.stem):
            raise RuntimeError('Invalid cleanup candidate: ' + str(path))
    removed = []
    for row in result['candidates']:
        path = Path(row['path'])
        check_path(path, Path(row['root']))
        if stamp(path) != row['identity'] or path.stat().st_nlink != 1:
            raise RuntimeError('Candidate changed during cleanup: ' + str(path))
        path.unlink()
        removed.append(str(path))
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-root', action='append', help='Repeat for additional exp directories; default ./exp')
    parser.add_argument('--apply', action='store_true', help='Delete the listed intermediate checkpoints')
    args = parser.parse_args()
    roots = args.exp_root or [str(Path.cwd() / 'exp')]
    if os.name != 'posix':
        raise RuntimeError('Run this command on the Linux training server')
    from w2v_v31810.processes import ensure_idle
    import fcntl
    with ExitStack() as stack:
        # Use the training launch locks, so a new workflow cannot start mid-scan.
        for root in sorted(set(roots)):
            folder = Path(root).expanduser().absolute()
            if folder.name != 'exp' or folder.resolve() != folder or not folder.is_dir():
                raise ValueError('Expected an existing, unlinked exp directory: ' + str(folder))
            lock_path = folder / '.aasist-launch.lock'
            if linked(lock_path):
                raise ValueError('Refusing linked launch lock: ' + str(lock_path))
            lock = stack.enter_context(lock_path.open('a+'))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle()
        result = plan(roots)
        for row in result['candidates']:
            print(f'{row["identity"][2]/1024**3:.3f} GiB  {row["path"]}', flush=True)
        print(f'FILES={len(result["candidates"])}  LOGICAL_GiB={result["total_bytes"]/1024**3:.3f}  '
              f'RETAINED={len(result["protected"])+len(result["skipped"])}', flush=True)
        log = Path(result['roots'][0]) / ('cleanup_keep_best_last_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
        result['mode'] = 'apply' if args.apply else 'preview'
        log.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('REPORT=' + str(log), flush=True)
        if not args.apply:
            print('PREVIEW ONLY. Add --apply to delete these files.', flush=True)
            return
        ensure_idle()
        result['removed'] = apply(result)
        result['status'] = 'complete'
        log.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('DELETED=' + str(len(result['removed'])) + '; BEST/LAST RETAINED', flush=True)


if __name__ == '__main__':
    main()
