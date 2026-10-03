"""Conservative, manifest-based retirement of inactive old checkpoints and launchers.

Uses only the standard library. No waveform cache, model package, selected best,
current V3.5 run or latest-version resume state is ever a removal candidate.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path('/home/ubuntu/LXT/RTC/xlsr_aasist/exp/w2v_rebuild_20260920_093548/stage3/best_model.pt')
BASELINE_SHA256 = 'db3f8167742bf2fe41cfad028dec962d56f6c61295442870620421d7f3a9bbee'
SUBMISSION_META = Path('/home/ubuntu/LXT/temp/w2v_v33_20261002_004018_198a_submission/submission_meta.json')
SOURCE_RUN = ROOT/'exp'/'w2v_v33_20261002_004018_198a'
OBSOLETE_SCRIPTS = {
    # Retire only byte-identified, superseded launch/setup wrappers. The packages,
    # exporters, status tools and V3.3/V3.4 deployment entry points remain intact.
    'run_w2v_v31.sh': 'fa8d098635c7ced3c350b1e35324e614f94059d3019dfad5e6aac2b9193b1007',
    'setup_w2v_v31.sh': '8ecb6eb29c3b3680ccb7d355d013c405bae64c6a9a6a511c26c49237933dbf9a',
    'run_w2v_v32.sh': 'a447b397cbc03cbd80408609b7eec7d0a5ae0f56d1cd91959d261616b07a17b7',
    'setup_w2v_v32.sh': 'b9e8fce0f5e179f8d2e7a17e943af7ab3fcfd51d508e9d83961b72f453c96d14',
}
CHECKPOINT_NAME = re.compile(r'^(?:last|latest|last_model|latest_model|checkpoint_last|(?:epoch|checkpoint|ckpt|snapshot|model_epoch)[_-]?\d+(?:[_-][a-zA-Z0-9_.-]+)?)\.(?:pt|pth|ckpt)$')
REFERENCE_NAMES = {'config.json', 'source.json', 'source_provenance.json', 'comparison.json',
                   'completed.json', 'stopped.json', 'status.json', 'submission_meta.json', 'catalog.json', 'selection.json'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def safe_path(path, root):
    """Validate lexical containment and every parent, including directory symlinks."""
    root = Path(root).absolute()
    path = Path(path).absolute()
    relative = path.relative_to(root)
    if '..' in relative.parts:
        raise ValueError('Traversal in cleanup path: ' + str(path))
    current = root
    if current.is_symlink():
        raise ValueError('Cleanup root cannot be a symlink')
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Symlink is never a cleanup target: ' + str(current))
    path.resolve().relative_to(root.resolve())
    return path


def safe_files(folder):
    if not Path(folder).is_dir() or Path(folder).is_symlink():
        return
    for parent, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(parent)/name).is_symlink()
                   and name not in {'maintenance_archive', 'retained', '.git', 'data', 'dataset', 'pretrained'}]
        for name in files:
            path = Path(parent)/name
            if path.is_file() and not path.is_symlink():
                yield path


def json_read(path):
    if Path(path).stat().st_size > 16 * 1024**2:
        raise ValueError('Unexpectedly large protection metadata: ' + str(path))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checkpoint_references(value, metadata, root):
    """Protect both cwd-relative and metadata-relative interpretations."""
    values = []
    if isinstance(value, dict):
        for key, item in value.items():
            values.extend(checkpoint_references(key, metadata, root))
            values.extend(checkpoint_references(item, metadata, root))
    elif isinstance(value, list):
        for item in value:
            values.extend(checkpoint_references(item, metadata, root))
    elif isinstance(value, str) and re.search(r'\.(?:pt|pth|ckpt)(?:\.previous)?$', value):
        p = Path(value).expanduser()
        values.extend([p.resolve()] if p.is_absolute() else [(root/p).resolve(), (metadata.parent/p).resolve()])
    return values


def active_jobs(root):
    """All model/data jobs in this checkout block cleanup, including evaluators."""
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Run production cleanup on the Linux training server')
    root = Path(root).resolve()
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(proc.name)
            if pid == os.getpid() or proc.stat().st_uid != os.getuid():
                continue
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            args = (proc/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            cwd = (proc/'cwd').resolve(strict=True)
            owned = cwd == root or root in cwd.parents or any(str(root) in arg for arg in args[1:])
            if not owned or not args or not Path(args[0]).name.startswith(('python', 'torchrun')):
                continue
            ignored = ('v32_console.py', 'v33_console.py', 'v34_console.py', 'v35_console.py',
                       'w2v_v35.maintenance', 'w2v_v35.stop')
            if any(arg in ignored or Path(arg).name in ignored for arg in args[1:]):
                continue
            if any(any(token in arg.lower() for token in ('w2v_', 'rtc_noisy', 'main_train', 'main_eval', 'prepare_rtc')) for arg in args[1:]):
                found.append({'pid': pid, 'command': ' '.join(args)})
        except (OSError, ValueError, IndexError):
            continue
    return found


def terminal_run(run):
    # A failed/interrupted run remains resumable unless explicitly stopped.
    for name in ('completed.json', 'stopped.json'):
        path = run/name
        if path.is_file() and not path.is_symlink():
            data = json_read(path)
            if isinstance(data, dict) and data.get('success') is not False:
                return True
    path = run/'status.json'
    if path.is_file() and not path.is_symlink():
        return json_read(path).get('status') in {'complete', 'completed', 'stopped'}
    # V3.3 has completion records inside its arms.
    arms = [run/name for name in ('control', 'candidate') if (run/name).is_dir()]
    return bool(arms) and all(terminal_run(arm) for arm in arms)


def build_plan(root, pins=(), metadata_files=()):
    root = Path(root).resolve()
    exp = safe_path(root/'exp', root)
    files = list(safe_files(exp))
    protected = {Path(path).resolve() for path in pins}
    manifests = [path for path in files if path.name in REFERENCE_NAMES]
    manifests += [Path(path).resolve() for path in metadata_files]
    catalog = root/'checkpoints'/'retained'/'catalog.json'
    if catalog.is_file():
        manifests.append(catalog)
    # A malformed configuration prevents deletion: it may contain a necessary pin.
    for parent, dirs, names in os.walk(exp, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent)/d).is_symlink() and d != 'maintenance_archive']
        if any((Path(parent)/name).is_symlink() for name in names if name in REFERENCE_NAMES):
            raise ValueError('Symlink protection metadata requires manual inspection')
    metadata_guards = []
    for path in manifests:
        protected.update(checkpoint_references(json_read(path), path, root))
        metadata_guards.append({'path': str(path), 'identity': identity(path), 'sha256': sha256(path)})
    latest = set()
    for pointer in exp.glob('.latest*_run'):
        if pointer.is_symlink():
            raise ValueError('Unexpected symlink resume pointer: ' + str(pointer))
        value = Path(pointer.read_text(encoding='utf-8').strip()).expanduser()
        latest.add((value if value.is_absolute() else root/value).resolve())
        metadata_guards.append({'path': str(pointer), 'identity': identity(pointer), 'sha256': sha256(pointer)})
    for path in files:
        parts = path.relative_to(exp).parts
        if (any('best' in part.lower() or 'candidate' in part.lower() for part in parts)
                or path.name.endswith('.previous') or 'ema' in path.name.lower()
                or any(part.startswith('w2v_v35_') for part in parts)):
            protected.add(path.resolve())
        if any(path == run or run in path.parents for run in latest) and path.stem in {
                'last', 'latest', 'last_model', 'latest_model', 'checkpoint_last'}:
            protected.add(path.resolve())
    inodes = {tuple(identity(p)[:2]) for p in protected if p.is_file()}
    checkpoints, skipped = [], []
    for path in files:
        if not CHECKPOINT_NAME.fullmatch(path.name):
            continue
        relative = path.relative_to(exp)
        if len(relative.parts) < 2:
            continue
        run = exp/relative.parts[0]
        # Scope to our known experiment namespace only.
        if not run.name.startswith('w2v_'):
            continue
        reason = None
        if path.resolve() in protected or tuple(identity(path)[:2]) in inodes:
            reason = 'protected_best_reference_or_resume'
        elif not terminal_run(run):
            reason = 'not_confirmed_complete_or_stopped'
        elif not any(p.name.startswith('best') and p.suffix in {'.pt', '.pth', '.ckpt'}
                     and p.stat().st_size > 0 for p in files if run in p.parents):
            reason = 'no_saved_best_in_experiment'
        if reason:
            skipped.append({'path': str(path), 'reason': reason})
        else:
            safe_path(path, exp)
            checkpoints.append({'path': str(path), 'identity': identity(path), 'kind': 'checkpoint'})
    scripts = []
    for name, expected in OBSOLETE_SCRIPTS.items():
        # Never turn a manifest value into an arbitrary deletion path.
        if Path(name).name != name or not name.endswith('.sh'):
            raise ValueError('Invalid obsolete script allowlist')
        path = safe_path(root/name, root)
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if digest != expected:
            skipped.append({'path': str(path), 'reason': 'modified_obsolete_script_retained'})
            continue
        scripts.append({'path': str(path), 'identity': identity(path), 'sha256': sha256(path), 'kind': 'script'})
    return {'root': str(root), 'checkpoints': checkpoints, 'scripts': scripts,
            'protected': sorted(map(str, protected)), 'skipped': skipped,
            'metadata_guards': metadata_guards,
            'planned_bytes': sum(row['identity'][2] for row in checkpoints + scripts),
            'audio_cache_deleted': False, 'runtime_packages_deleted': False}


def validate_plan(plan):
    root = Path(plan['root']).resolve()
    for item in plan.get('metadata_guards', []):
        if identity(item['path']) != item['identity'] or sha256(item['path']) != item['sha256']:
            raise RuntimeError('Protection metadata changed; rebuild cleanup plan')
    for item in plan['checkpoints'] + plan['scripts']:
        path = safe_path(item['path'], root/'exp' if item['kind'] == 'checkpoint' else root)
        if item['kind'] == 'checkpoint' and not CHECKPOINT_NAME.fullmatch(path.name):
            raise ValueError('Not a recognized intermediate checkpoint: ' + str(path))
        if item['kind'] == 'script' and path.name not in OBSOLETE_SCRIPTS:
            raise ValueError('Not an allowlisted obsolete launcher: ' + str(path))
        if identity(path) != item['identity']:
            raise RuntimeError('Cleanup target changed; no deletion started: ' + str(path))
        if item['kind'] == 'script' and sha256(path) != item['sha256']:
            raise RuntimeError('Obsolete launcher changed')


def apply_plan(plan, report, idle_check):
    active = idle_check()
    if active:
        raise RuntimeError('Active training/evaluation/cache job; nothing removed: ' + json.dumps(active))
    validate_plan(plan)
    atomic_json(report, dict(plan, status='planned', removed=[]))
    archive = None
    if plan['scripts']:
        root = Path(plan['root'])
        archive_dir = safe_path(root/'exp'/'maintenance_archive', root)
        archive_dir.mkdir(exist_ok=True)
        archive = archive_dir/(Path(report).stem + '_scripts.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as handle:
            for item in plan['scripts']:
                handle.write(item['path'], arcname=Path(item['path']).name)
        with zipfile.ZipFile(archive) as handle:
            for item in plan['scripts']:
                if hashlib.sha256(handle.read(Path(item['path']).name)).hexdigest() != item['sha256']:
                    raise RuntimeError('Script archive verification failed; nothing removed')
    validate_plan(plan)
    active = idle_check()
    if active:
        raise RuntimeError('Job started during cleanup planning; nothing removed: ' + json.dumps(active))
    removed = []
    try:
        for item in plan['checkpoints'] + plan['scripts']:
            validate_plan(dict(plan, checkpoints=[item] if item['kind'] == 'checkpoint' else [],
                               scripts=[item] if item['kind'] == 'script' else []))
            Path(item['path']).unlink()  # Explicit files only; no recursive deletion.
            removed.append(item['path'])
    finally:
        atomic_json(report, dict(plan, status='complete' if len(removed) == len(plan['checkpoints']) + len(plan['scripts']) else 'interrupted',
                                removed=removed, script_archive=str(archive) if archive else None))
    return removed


@contextmanager
def cleanup_lock(root):
    # Same lock as every production launcher; no Torch import required.
    import fcntl
    path = Path(root)/'exp'/'.aasist-launch.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Training/workflow holds the launch lock; nothing removed') from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def protected_submission(metadata, source_run=SOURCE_RUN):
    data = json_read(metadata)
    # V3.3's exporter records SHA/tag, but not the checkpoint path. Bind that
    # metadata to its completed comparison instead of inventing a best filename.
    if data.get('checkpoint'):
        path = Path(data['checkpoint']).expanduser()
    else:
        source_run = Path(source_run).expanduser().resolve()
        comparison = json_read(source_run/'comparison.json')
        arm = comparison.get('selected_arm')
        if comparison.get('status') != 'complete' or arm not in {'control', 'candidate'}:
            raise ValueError('Submission requires a completed V3.3 selected arm')
        path = Path(comparison['checkpoint']).expanduser().resolve()
        expected = (source_run/arm/'best_model.pt').resolve()
        expected.relative_to(source_run)
        tag = comparison.get('arms', {}).get(arm, {}).get('selected', {}).get('tag')
        if path != expected or not tag or data.get('checkpoint_tag') != tag:
            raise ValueError('Submission metadata differs from the completed selected checkpoint')
    digest = data['checkpoint_sha256']
    if not path.is_absolute():
        path = ROOT/path
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest) or sha256(path) != digest:
        raise RuntimeError('Submitted checkpoint identity differs; nothing removed')
    return path.resolve(), digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply the printed exact-file cleanup manifest')
    parser.add_argument('--submission-meta', type=Path, default=SUBMISSION_META)
    parser.add_argument('--source-run', type=Path, default=SOURCE_RUN,
                        help='V3.3 completed comparison for metadata without a checkpoint path')
    args = parser.parse_args()
    if os.name != 'posix':
        raise RuntimeError('Run production cleanup on the Linux training server')
    with cleanup_lock(ROOT):
        active = active_jobs(ROOT)
        if active:
            raise RuntimeError('Training/evaluation/cache job is active; nothing removed: ' + json.dumps(active))
        print('Verifying original 91.68 and submitted 93.3941 checkpoint identities...', flush=True)
        if sha256(BASELINE) != BASELINE_SHA256:
            raise RuntimeError('Protected original checkpoint differs; nothing removed')
        submitted, digest = protected_submission(args.submission_meta, args.source_run)
        extra_metadata = [args.submission_meta]
        if (args.source_run/'comparison.json').is_file():
            extra_metadata.append(args.source_run/'comparison.json')
        plan = build_plan(ROOT, [BASELINE, submitted], extra_metadata)
        for item in plan['checkpoints'] + plan['scripts']:
            print(f"{item['kind']} {item['identity'][2]/1024**3:.3f} GiB {item['path']}", flush=True)
        print(f"PLANNED_FILES={len(plan['checkpoints'])+len(plan['scripts'])} LOGICAL_GiB={plan['planned_bytes']/1024**3:.3f}", flush=True)
        report = ROOT/'exp'/('maintenance_v35_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        if not args.apply:
            atomic_json(report, dict(plan, status='preview'))
            print('PREVIEW_ONLY=True; add --apply to delete only these files.\nREPORT='+str(report))
            return
        removed = apply_plan(plan, report, lambda: active_jobs(ROOT))
        if sha256(BASELINE) != BASELINE_SHA256 or sha256(submitted) != digest:
            raise RuntimeError('Protected checkpoint changed externally during maintenance')
        print(f'DELETED_FILES={len(removed)} ORIGINAL_AND_SUBMITTED_BEST_PRESERVED=True\nREPORT={report}')
        print(f'FREE_GiB={shutil.disk_usage(ROOT).free/1024**3:.2f}')


if __name__ == '__main__':
    main()
