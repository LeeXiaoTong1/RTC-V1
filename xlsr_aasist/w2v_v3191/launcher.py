"""Launch V3.19.1 in an isolated session; the read-only viewer can be closed."""
import os
from pathlib import Path
import subprocess
import sys


def launch(log, args, root=None, popen=subprocess.Popen):
    if os.name != 'posix':
        raise RuntimeError('Production training launch requires Linux')
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    log = Path(log).resolve()
    if log.parent != root/'exp' or not log.name.startswith('v3191_') or log.suffix != '.log':
        raise ValueError('Expected an exp/v3191_*.log output')
    with log.open('ab', buffering=0) as output:
        child = popen(['bash', str(root/'v3191_supervise.sh'), str(log), *args], cwd=root,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    return child.pid


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m w2v_v3191.launcher LOG [training arguments]')
    print(launch(sys.argv[1], sys.argv[2:]), flush=True)
