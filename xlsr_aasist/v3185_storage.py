"""Preview or retire completed noise-preparation inputs before V3.18.5.

Only the payload whitelist in v3184_storage is eligible. Prepared noise,
manifests, original/fixed-Dev audio, checkpoints, environments, model caches,
and unknown files are retained. Apply requires an idle Linux training server.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import sys

from v3184_storage import ROOT, SOURCES, GIB, noise_plan, apply_plan, checked_path
from w2v_v3184.prepare_noise import preparation_lock


def emit(message):
    """Keep paths/errors printable even on an ASCII-only terminal."""
    print(str(message).encode('ascii', 'backslashreplace').decode('ascii'), flush=True)


def canonical_directory(path):
    path = Path(path).expanduser().absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('Refuse directory link/junction: ' + str(part))
    if not path.is_dir():
        raise ValueError('Expected an existing directory: ' + str(path))
    return path.resolve()


def is_producer(args):
    """Match executable arguments, never read an unbounded process environment."""
    if not args:
        return False
    executable = Path(args[0]).name.lower()
    if executable.startswith('python') or executable in ('torchrun', 'accelerate'):
        # Status viewers can carry the run name; retain the conservative legacy
        # behavior for those processes instead of incorrectly declaring idle.
        return any(any(tag in item.lower() for tag in
                       ('w2v_', 'rtc_noisy', 'main_train', 'noise_v3184',
                        'v3184_aug5_supervise', 'v3185_supervise')) for item in args[1:])
    if executable in ('bash', 'sh', 'dash', 'zsh'):
        # Direct launcher/supervisor commands cover the short interval before
        # their Python child starts. The cleanup wrapper itself is not a job.
        return any(re.fullmatch(r'(?:run_w2v_.*|prepare_noise_.*|v318[0-9a-z_]*_supervise)\.sh',
                                Path(item).name) for item in args[1:])
    return False


def active_jobs(proc_root=Path('/proc'), uid=None, own_pid=None):
    proc_root = Path(proc_root)
    if not proc_root.is_dir():
        raise RuntimeError('Cannot verify running processes: /proc is unavailable')
    uid = os.getuid() if uid is None else uid
    own_pid = os.getpid() if own_pid is None else own_pid
    jobs = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) == own_pid:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if is_producer(args):
                jobs.append(dict(pid=int(entry.name), command=' '.join(args)))
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process that has exited cannot use these files.
        except PermissionError as exc:
            raise RuntimeError('Cannot inspect own process; no cleanup: PID ' + entry.name) from exc
    return sorted(jobs, key=lambda row: row['pid'])


def require_idle():
    jobs = active_jobs()
    if jobs:
        raise RuntimeError('Active training/evaluation/preparation/viewer; finish it before cleanup: '
                           + json.dumps(jobs, ensure_ascii=True))


@contextmanager
def idle_launch_lock(root):
    if os.name != 'posix':
        raise RuntimeError('Apply on the Linux training server; preview is available here')
    import fcntl
    require_idle()
    exp = canonical_directory(root / 'exp')
    lock = checked_path(exp, '.aasist-launch.lock', missing=True)
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Training workflow owns the launch lock; no cleanup') from exc
        try:
            require_idle()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def summarize(plan, progress=emit):
    progress('NOISE_BANK=' + plan['bank'])
    progress('RAW_CANDIDATES=' + str(len(plan['candidates'])))
    progress('RECLAIMABLE_BYTES=' + str(plan['reclaimable_bytes']))
    progress(f'RECLAIMABLE_GIB={plan["reclaimable_bytes"] / GIB:.3f}')
    progress('KEEP=prepared_noise,manifests,original_audio,fixed_dev,all_checkpoints,environments,model_caches,unknown_files')


def retire_verified(plan, guard=require_idle, progress=emit):
    """Caller holds the launch and preparation locks for the entire operation."""
    guard()
    bank = canonical_directory(plan['bank'])
    before = shutil.disk_usage(bank).free

    def verified_progress(message):
        guard()
        progress(message)

    # The existing helper authenticates retained prepared audio (in noise_plan),
    # completion metadata, raw hashes and file identities before deletion; it
    # rechecks each payload's identity immediately before unlinking that file.
    result = apply_plan(plan, verified_progress)
    after = shutil.disk_usage(bank).free
    result.update(freed_bytes=max(0, after - before), free_bytes_before=before,
                  free_bytes_after=after,
                  freed_bytes_note='Observed positive filesystem free-space change; other processes and receipt allocation can affect it.')
    progress('REMOVED_FILES=' + str(result['removed_files']))
    progress('REMOVED_PAYLOAD_BYTES=' + str(result['removed_bytes']))
    progress('FREED_BYTES=' + str(result['freed_bytes']))
    progress(f'FILESYSTEM_FREE_GIB={after / GIB:.3f}')
    progress('FREED_BYTES_IS_OBSERVED_FILESYSTEM_DELTA=True')
    return result


def execute(root, bank, apply=False, sources_path=SOURCES, progress=emit):
    root = canonical_directory(root)
    bank = canonical_directory(bank)
    progress('MODE=' + ('APPLY_VERIFIED_NOISE_RAW_ONLY' if apply else 'PREVIEW_NO_DELETION'))
    progress(f'FILESYSTEM_FREE_GIB={shutil.disk_usage(bank).free / GIB:.3f}')

    def work():
        checked_path(bank, '.prepare.lock', missing=True)
        with preparation_lock(bank / '.prepare.lock'):
            progress('VERIFY=completed_noise_metadata_and_retained_audio; this may take time')
            plan = noise_plan(bank, sources_path)
            summarize(plan, progress)
            if apply:
                return retire_verified(plan, progress=progress)
            progress('PREVIEW_COMPLETE=True; deletion requires --apply after all jobs finish')
            return plan

    if apply:
        with idle_launch_lock(root):
            return work()
    return work()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--noise-bank', help='Default: ROOT/assets/noise_v3184_rnnoise')
    parser.add_argument('--apply', action='store_true', help='Delete verified raw inputs only, after all jobs finish')
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().absolute()
    bank = Path(args.noise_bank).expanduser().absolute() if args.noise_bank else root / 'assets/noise_v3184_rnnoise'
    try:
        execute(root, bank, args.apply)
    except (OSError, ValueError, RuntimeError) as exc:
        emit('CLEANUP_REFUSED=' + str(exc))
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
