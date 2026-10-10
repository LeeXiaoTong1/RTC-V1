"""Launch the V3.18 supervisor in its own session, separate from the viewer."""
import os
from pathlib import Path
import subprocess
import sys


def launch(log, args, root=None, popen=subprocess.Popen):
    if os.name != 'posix':
        raise RuntimeError('Production training launch requires Linux')
    root = Path(root or Path(__file__).resolve().parent).resolve()
    log = Path(log).resolve()
    with log.open('ab', buffering=0) as output:
        child = popen(['bash', str(root/'v318_supervise.sh'), str(log), *args],
                      cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                      stderr=subprocess.STDOUT, start_new_session=True,
                      close_fds=True)
    return child.pid


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python v318_detach.py LOG [training arguments]')
    print(launch(sys.argv[1], sys.argv[2:]), flush=True)
