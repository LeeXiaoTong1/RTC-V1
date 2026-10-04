"""Stop only this checkout's V3.7 workflow and verified descendants."""
import argparse
import os
from pathlib import Path
import signal
import time
ROOT = Path(__file__).resolve().parent.parent
from w2v_v33.stop import inspect, owned_roots, descendants, same_process


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Run the stop command on the Linux training server')
    processes = [item for entry in Path('/proc').iterdir() if entry.name.isdigit()
                 for item in [inspect(int(entry.name))] if item is not None]
    targets = descendants(processes, owned_roots(processes, ROOT.resolve(), 'v37'))
    if any(t['pid'] == os.getpid() for t in targets):
        raise RuntimeError('Refusing to signal the stop command itself')
    if not targets:
        print('No matching active V3.7 job in this checkout.'); return
    for t in targets:
        print(f"PID={t['pid']} "+' '.join(t['args']), flush=True)
    if not args.apply:
        print('Preview only; add --apply to stop these processes.'); return
    ordered = sorted(targets, key=lambda t: 0 if 'w2v_v37.workflow' in t['args'] else 1)
    for t in ordered:
        if same_process(t):
            try:
                os.kill(t['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+15
    while time.monotonic() < deadline and any(same_process(t) for t in targets):
        time.sleep(.25)
    pending = [t['pid'] for t in targets if same_process(t)]
    if pending:
        raise RuntimeError(f'Processes still exiting: {pending}; no forced kill issued')
    print('TRAINING_STOPPED=True; saved checkpoints retained. Committed feature rows are reused on resume.')


if __name__ == '__main__':
    main()
