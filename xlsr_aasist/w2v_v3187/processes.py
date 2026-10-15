"""Keep read-only viewers separate from training/evaluation concurrency checks."""
import json
import os
from pathlib import Path

READ_ONLY_MODULES = frozenset({'w2v_v3187.monitor'})


def is_read_only_viewer(argv):
    """Match the executed Python module, never text in a script's arguments."""
    if not argv or not Path(argv[0]).name.startswith('python'):
        return False
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == '-m':
            return index + 1 < len(argv) and argv[index + 1] in READ_ONLY_MODULES
        if token in ('-X', '-W'):
            if index + 1 >= len(argv):
                return False
            index += 2
        elif token in ('-u', '-B', '-E', '-I', '-O', '-OO', '-s', '-S', '-q', '-v', '-b', '-bb', '-d'):
            index += 1
        elif token.startswith(('-X', '-W')) and len(token) > 2:
            index += 1
        else:
            # -c, stdin, scripts and unknown interpreter options do not grant
            # an exemption even if a later argument mentions the viewer.
            return False
    return False


def ensure_idle(*, proc_root=None):
    """Block other model/cache jobs while allowing our read-only dashboard.

    All production callers also use the existing launch lock. proc_root is a
    test seam for process-table fixtures, never a CLI bypass.
    """
    if proc_root is None:
        if os.name != 'posix' or not Path('/proc').is_dir():
            raise RuntimeError('Run the production workflow on the Linux training server')
        proc_root = Path('/proc')
    else:
        proc_root = Path(proc_root)
    uid, pid, parent = os.getuid(), os.getpid(), os.getppid()
    active = []
    for entry in proc_root.glob('[0-9]*'):
        try:
            if int(entry.name) == pid or entry.stat().st_uid != uid:
                continue
            argv = (entry/'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            if not argv or not Path(argv[0]).name.startswith('python'):
                continue
            if is_read_only_viewer(argv):
                continue
            # Preserve the older authenticated progress-owner exception.
            if (entry.name == os.environ.get('AASIST_PROGRESS_OWNER') and int(entry.name) == parent
                    and any(Path(arg).name == 'live_progress.py' for arg in argv[1:3])):
                continue
            if any(any(tag in arg.lower() for tag in ('w2v_', 'rtc_noisy', 'main_train')) for arg in argv[1:]):
                active.append(dict(pid=int(entry.name), command=' '.join(argv)))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if active:
        raise RuntimeError('Finish/stop the existing training, evaluation or cache job before launching: '
                           + json.dumps(active))


if __name__ == '__main__':
    ensure_idle()
