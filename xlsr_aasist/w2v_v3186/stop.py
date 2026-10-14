"""Stop one verified detached V3.18.6 training session, without deleting files."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LINUX = sys.platform.startswith('linux')


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    group: int
    session: int
    start: int
    uid: int
    state: str
    command: tuple
    cwd: str


def processes(proc=Path('/proc')):
    result = {}
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
            command = tuple(x.decode(errors='replace') for x in (directory/'cmdline').read_bytes().split(b'\0') if x)
            result[int(directory.name)] = Process(int(directory.name), int(fields[1]), int(fields[2]),
                int(fields[3]), int(fields[19]), directory.stat().st_uid, fields[0], command,
                str((directory/'cwd').resolve(strict=True)))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return result


def verify_session(records, pid, root, log, uid, own_group):
    """Require the exact launcher identity and an isolated, same-user process tree."""
    root, log = Path(root).resolve(), Path(log).resolve()
    leader = records.get(pid)
    if leader is None:
        raise ProcessLookupError('Recorded supervisor is no longer running')
    expected = str(root/'v3186_supervise.sh')
    if (leader.uid != uid or leader.session != pid or leader.group != pid or
        leader.group == own_group or Path(leader.cwd).resolve() != root or
        len(leader.command) < 3 or Path(leader.command[0]).name != 'bash' or
        leader.command[1] != expected or Path(leader.command[2]).resolve() != log or
        log.parent != root/'exp' or not log.name.startswith('v3186_') or log.suffix != '.log'):
        raise ValueError('Refusing to signal: recorded PID is not the expected isolated V3.18.6 supervisor')
    members = [item for item in records.values() if item.session == pid and item.state != 'Z']
    for item in members:
        if item.uid != uid or item.group == own_group:
            raise ValueError('Refusing to signal a different user or the current terminal group')
        ancestor, visited = item.pid, set()
        while ancestor != pid:
            if ancestor in visited or ancestor not in records:
                raise ValueError('Session contains an unverified process; no signal sent')
            visited.add(ancestor)
            ancestor = records[ancestor].ppid
        # killpg must not include any process outside the validated session.
        if any(p.group == item.group and (p.session != pid or p.uid != uid) for p in records.values()):
            raise ValueError('Process group contains an unrelated process; no signal sent')
    return leader, members


def verify_run(root, log, run):
    run = Path(run).expanduser().resolve()
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    if config.get('version') != '3.18.6':
        raise ValueError('--run must name a V3.18.6 run')
    # Link the explicit run to THIS supervisor log, not merely a latest-run marker.
    with Path(log).open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if line.startswith('V3186_RUN='):
                if Path(line.split('=', 1)[1].strip()).resolve() != run:
                    raise ValueError('--run differs from the run recorded in this supervisor log')
                return run
    raise ValueError('The supervisor log has no V3186_RUN identity yet; omit --run only after checking --dry-run')


def stop(root=ROOT, run=None, dry_run=False, timeout=30, snapshot=processes,
         signal_group=None, sleeper=time.sleep, monotonic=time.monotonic):
    if not LINUX:
        raise RuntimeError('This command requires Linux /proc')
    root = Path(root).resolve()
    pid = int((root/'exp'/'.latest_v3186_pid').read_text().strip())
    log = Path((root/'exp'/'.latest_v3186_log').read_text().strip()).resolve()
    records = snapshot()
    if pid not in records:
        remaining = [p.pid for p in records.values() if p.session == pid and p.state != 'Z']
        if remaining:
            raise RuntimeError('Supervisor exited but its session still has processes: '+str(remaining)+'. No signal sent.')
        print('ALREADY_STOPPED: recorded V3.18.6 supervisor is not running', flush=True)
        return True
    leader, members = verify_session(records, pid, root, log, os.getuid(), os.getpgrp())
    if run is not None:
        print('RUN='+str(verify_run(root, log, run)), flush=True)
    print('LOG='+str(log), flush=True)
    print('SESSION='+str(pid)+' PROCESSES='+','.join(str(p.pid) for p in members), flush=True)
    print('Last committed epoch checkpoint and logs are retained. Unsaved current-epoch work will be lost.', flush=True)
    if dry_run:
        print('DRY_RUN: no signal sent', flush=True)
        return True
    fresh = snapshot()
    current, members = verify_session(fresh, pid, root, log, os.getuid(), os.getpgrp())
    if current.start != leader.start:
        raise RuntimeError('Supervisor PID was reused; no signal sent')
    groups = {p.group for p in members}
    # Descendant groups first; supervisor/main group last. No blanket pkill/killall.
    for group in sorted(groups, key=lambda value: value == pid):
        try:
            (signal_group or os.killpg)(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = monotonic()+timeout
    while True:
        remaining = [p.pid for p in snapshot().values() if p.session == pid and p.state != 'Z']
        if not remaining:
            print('STOPPED: verified V3.18.6 training session exited', flush=True)
            return True
        if monotonic() >= deadline:
            print('STOP_PENDING: remaining PIDs='+','.join(map(str, remaining))+'. No SIGKILL sent; do not start another run yet.', flush=True)
            return False
        sleeper(.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT), help='xlsr_aasist checkout directory, useful for running this helper before git pull')
    parser.add_argument('--run', help='Assert the run recorded in the active supervisor log')
    parser.add_argument('--dry-run', action='store_true', help='Verify and display processes without signalling')
    parser.add_argument('--timeout', type=int, default=30, help='Wait at most this many seconds after SIGTERM')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 60:
        parser.error('--timeout must be between 1 and 60 seconds')
    try:
        success = stop(root=args.root, run=args.run, dry_run=args.dry_run, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, 'STOP_REFUSED: '+str(error)+'\n')
    if not success:
        parser.exit(2)


if __name__ == '__main__':
    main()
