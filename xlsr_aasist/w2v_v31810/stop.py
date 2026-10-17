"""Gracefully pause one verified V3.18.10 session at an optimizer boundary."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
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
    expected = str(root/'v31810_supervise.sh')
    if (leader.uid != uid or leader.session != pid or leader.group != pid or
        leader.group == own_group or Path(leader.cwd).resolve() != root or
        len(leader.command) < 3 or Path(leader.command[0]).name != 'bash' or
        leader.command[1] != expected or Path(leader.command[2]).resolve() != log or
        log.parent != root/'exp' or not log.name.startswith('v31810_') or log.suffix != '.log'):
        raise ValueError('Refusing to signal: recorded PID is not the expected isolated V3.18.10 supervisor')
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
    if config.get('version') != '3.18.10':
        raise ValueError('--run must name a V3.18.10 run')
    # Link the explicit run to THIS supervisor log, not merely a latest-run marker.
    with Path(log).open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if line.startswith('V31810_RUN='):
                if Path(line.split('=', 1)[1].strip()).resolve() != run:
                    raise ValueError('--run differs from the run recorded in this supervisor log')
                return run
    raise ValueError('The supervisor log has no V31810_RUN identity yet; omit --run only after checking --dry-run')


def run_from_log(log):
    found = None
    with Path(log).open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if line.startswith('V31810_RUN='):
                found = line.split('=', 1)[1].strip()
    if not found:
        raise ValueError('Run initialization is not complete yet; wait until V31810_RUN is recorded')
    return Path(found).expanduser().resolve()


def stop(root=ROOT, run=None, dry_run=False, timeout=30, snapshot=processes,
         sleeper=time.sleep, monotonic=time.monotonic):
    """Request a checkpoint at the next optimizer boundary; never kill workers."""
    if not LINUX:
        raise RuntimeError('This command requires Linux /proc')
    root = Path(root).resolve()
    pid = int((root/'exp'/'.latest_v31810_pid').read_text().strip())
    log = Path((root/'exp'/'.latest_v31810_log').read_text().strip()).resolve()
    records = snapshot()
    if pid not in records:
        remaining = [p.pid for p in records.values() if p.session == pid and p.state != 'Z']
        if remaining:
            raise RuntimeError('Supervisor exited but its session still has processes: '+str(remaining)+'. No request written.')
        print('ALREADY_STOPPED: recorded V3.18.10 supervisor is not running', flush=True)
        return True
    leader, members = verify_session(records, pid, root, log, os.getuid(), os.getpgrp())
    run = verify_run(root, log, run or run_from_log(log))
    print('RUN='+str(run), flush=True)
    print('SESSION='+str(pid)+' PROCESSES='+','.join(str(p.pid) for p in members), flush=True)
    if dry_run:
        print('DRY_RUN: no stop request written', flush=True)
        return True
    current, _ = verify_session(snapshot(), pid, root, log, os.getuid(), os.getpgrp())
    if current.start != leader.start:
        raise RuntimeError('Supervisor PID was reused; no stop request written')
    if (run/'completed.json').is_file():
        value = json.loads((run/'completed.json').read_text(encoding='utf-8'))
        if value.get('status') == 'complete':
            print('TRAINING_COMPLETE: artifact export may still be running; no stop request written', flush=True)
            return True
    request = run/'stop_requested.json'
    temporary = request.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(version='3.18.10', action='pause', supervisor_pid=pid,
        supervisor_start=leader.start, requested_at=time.time()), indent=2)+'\n', encoding='utf-8')
    temporary.replace(request)
    print('STOP_REQUESTED: save model, optimizer, RNG and next update at a safe boundary. Do not power off yet.', flush=True)
    deadline = monotonic()+timeout
    while True:
        remaining = [p.pid for p in snapshot().values() if p.session == pid and p.state != 'Z']
        if not remaining:
            paused = run/'paused.json'
            if paused.is_file() and json.loads(paused.read_text(encoding='utf-8')).get('status') == 'paused':
                print('PAUSED: resumable checkpoint saved and training session exited', flush=True)
                return True
            completed = run/'completed.json'
            if completed.is_file() and json.loads(completed.read_text(encoding='utf-8')).get('status') == 'complete':
                print('COMPLETE: training finished before the pause request', flush=True)
                return True
            raise RuntimeError('Session exited without a pause acknowledgement. Check the log before resuming.')
        if monotonic() >= deadline:
            print('STOP_PENDING: checkpoint request remains active. Wait for PAUSED in the viewer; no force kill sent.', flush=True)
            return False
        sleeper(.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--run', help='Assert the run recorded in the active supervisor log')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--timeout', type=int, default=30)
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
