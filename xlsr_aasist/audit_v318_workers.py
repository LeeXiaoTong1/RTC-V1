"""Read-only worker failure evidence; no ML imports, GPU use or worker launch.

Safe alongside a running job. Prints an ASCII summary and writes one JSON report
outside training run directories. Historical OOM counters are NOT proof that a
particular failed worker was killed. Exact kernel PID evidence is reported apart.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def job_status(root):
    log_name = read(root/'exp'/'.latest_v318_log').strip()
    pid_value = read(root/'exp'/'.latest_v318_pid').strip()
    result = dict(log=log_name, supervisor_pid=pid_value, supervisor_alive=None)
    if pid_value.isdigit() and os.name == 'posix':
        pid = int(pid_value)
        try:
            os.kill(pid, 0)
            result['supervisor_alive'] = True
        except ProcessLookupError:
            result['supervisor_alive'] = False
        except (PermissionError, OSError):
            pass
        if Path('/proc').is_dir():
            status = pairs(read(Path('/proc')/str(pid)/'status'))
            result['supervisor_state'] = status.get('State')
            if status.get('State', '').startswith('Z'):
                result['supervisor_alive'] = False
            cmd = read(Path('/proc')/str(pid)/'cmdline')
            result['supervisor_identity_matches'] = bool(cmd and 'v318_supervise.sh' in cmd and log_name in cmd)
    if log_name:
        p = Path(log_name)
        result['exit_code'] = read(Path(log_name+'.exit')).strip() or None
        if p.is_file():
            result['log_age_seconds'] = round(time.time()-p.stat().st_mtime, 1)
            with p.open('rb') as f:
                f.seek(max(0, p.stat().st_size-4096))
                result['log_tail'] = f.read().decode('utf-8', errors='replace').splitlines()[-18:]
    return result


def read(path, limit=1024*1024):
    try:
        with Path(path).open('rb') as f:
            return f.read(limit).decode('utf-8', errors='replace')
    except OSError:
        return ''


def pairs(text):
    result = {}
    for line in text.splitlines():
        parts = line.replace(':', ' ', 1).split()
        if len(parts) >= 2:
            result[parts[0]] = ' '.join(parts[1:])
    return result


def recent_steps(path, count=50, limit=2*1024*1024):
    try:
        with Path(path).open('rb') as f:
            size = f.seek(0, 2)
            f.seek(max(0, size-limit))
            if size > limit:
                f.readline()
            lines = f.read().splitlines()
    except OSError:
        return {}
    rows = []
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if not all(k in row for k in ('epoch', 'step', 'compute_seconds', 'wait_seconds')):
                continue
            if rows and row['epoch'] != rows[0]['epoch']:
                break
            rows.append(row)
            if len(rows) == count:
                break
        except (ValueError, TypeError):
            continue
    if not rows:
        return {}
    compute = sum(r['compute_seconds'] for r in rows)/len(rows)
    wait = sum(r['wait_seconds'] for r in rows)/len(rows)
    return dict(epoch=rows[0]['epoch'], step=rows[0]['step'], count=len(rows),
                compute_seconds=compute, wait_seconds=wait,
                wait_fraction=wait/max(compute+wait, 1e-9))


def failed_logs(root):
    results = []
    files = sorted((root/'exp').glob('v318_*.log'), key=lambda p:p.stat().st_mtime, reverse=True)[:12]
    for path in files:
        # Failures before the first update have tiny logs. Retain bounded tails
        # as well so a later long-running worker failure is not omitted.
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size-256*1024))
            content = f.read().decode('utf-8', errors='replace')
        if 'DataLoader worker' not in content or 'exited unexpectedly' not in content:
            continue
        pids = []
        for group in re.findall(r'pid\(s\)\s+([\d, ]+)\) exited unexpectedly', content):
            pids.extend(int(x) for x in re.findall(r'\d+', group))
        lines = content.splitlines()
        evidence = [line for line in lines if re.search(
            r'Traceback|Error|Killed|killed|SIG|signal|core dumped|segfault|Segmentation|Bus error|V318_RUN=', line)]
        results.append(dict(log=str(path), worker_pids=sorted(set(pids)),
                            modified_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                            evidence=evidence[-35:]))
        if len(results) == 3:
            break
    return results


def processes(proc=Path('/proc')):
    records = {}
    if not proc.is_dir():
        return []
    uid = os.getuid()
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != uid:
                continue
            status = pairs(read(p/'status'))
            cmd = read(p/'cmdline').replace('\0', ' ')
            records[int(p.name)] = dict(pid=int(p.name), parent=int(status.get('PPid', 0)),
                name=status.get('Name'), rss=status.get('VmRSS'), threads=status.get('Threads'),
                sig_ignored=status.get('SigIgn'), sig_caught=status.get('SigCgt'),
                oom_score=read(p/'oom_score').strip(),
                is_v318='w2v_v318.workflow' in cmd or 'v318_supervise.sh' in cmd)
        except (OSError, ValueError):
            continue
    selected = {pid for pid, r in records.items() if r['is_v318']}
    while True:
        expanded = selected | {pid for pid, r in records.items() if r['parent'] in selected}
        if expanded == selected:
            break
        selected = expanded
    result = []
    for pid in sorted(selected):
        r = records[pid]
        r['cgroup'] = read(proc/str(pid)/'cgroup')
        r['limits'] = read(proc/str(pid)/'limits')
        rollup = pairs(read(proc/str(pid)/'smaps_rollup'))
        r['pss'] = rollup.get('Pss')
        result.append(r)
    return result


def cgroup_dirs(text, base=Path('/sys/fs/cgroup')):
    paths = set()
    for line in text.splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, controllers, rel = parts
        roots = [base] if not controllers else ([base/'memory'] if 'memory' in controllers.split(',') else [])
        for mount in roots:
            mount = mount.resolve()
            candidate = (mount/rel.lstrip('/')).resolve()
            if candidate != mount and mount not in candidate.parents:
                continue
            paths.add(candidate)
            paths.update(p for p in candidate.parents if p == mount or mount in p.parents)
    return paths


def cgroups(jobs):
    paths = cgroup_dirs(read('/proc/self/cgroup'))
    for r in jobs:
        paths.update(cgroup_dirs(r['cgroup']))
    names = ('memory.current', 'memory.max', 'memory.peak', 'memory.events', 'memory.events.local',
             'memory.swap.current', 'memory.swap.max', 'pids.current', 'pids.max', 'pids.events',
             'memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt', 'memory.oom_control')
    return {str(p): {n: read(p/n).strip() for n in names if (p/n).is_file()}
            for p in sorted(paths) if p.is_dir()}


def kernel_evidence(failures, runner=subprocess.run):
    pids = {pid for item in failures for pid in item['worker_pids']}
    try:
        r = runner(['dmesg', '--color=never'], capture_output=True, text=True, timeout=8)
        if r.returncode:
            return dict(available=False, reason=r.stderr.strip()[:500], matched_worker_pids=[])
        lines = [line for line in r.stdout.splitlines() if re.search(
            r'out of memory|oom-kill|oom_reaper|Killed process|segfault|segmentation|bus error', line, re.I)]
        matched = []
        evidence = []
        for line in lines:
            hits = [pid for pid in pids if re.search(r'(?<!\d)'+str(pid)+r'(?!\d)', line)]
            if hits:
                matched.extend(hits); evidence.append(line)
        return dict(available=True, matched_worker_pids=sorted(set(matched)),
                    matching_lines=evidence[-20:], recent_system_events=lines[-12:],
                    caveat='PID matches require timestamp confirmation; PIDs can be reused.')
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(available=False, reason=str(exc), matched_worker_pids=[])


def run_info(root):
    value = read(root/'exp'/'.latest_v318_run').strip()
    if not value:
        return {}
    p = Path(value)
    info = dict(path=str(p), recent=recent_steps(p/'steps.jsonl'))
    if (p/'steps.jsonl').is_file():
        info['steps_age_seconds'] = round(time.time()-(p/'steps.jsonl').stat().st_mtime, 1)
    config = p/'config.json'
    if config.is_file() and config.stat().st_size < 16*1024**2:
        cfg = json.loads(config.read_text())
        keys = ('workers','stream_sources','microbatch','raw_audio_cache_mib','noise_cache_mib','omni_arch')
        info['config'] = {k:cfg.get(k) for k in keys}
    info['metadata_bytes'] = {n:(p/n).stat().st_size for n in ('config.json','train_rows.json','dev_rows.json') if (p/n).is_file()}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures = failed_logs(root)
    jobs = processes()
    info = run_info(root)
    report = dict(time_utc=datetime.now(timezone.utc).isoformat(), active_run=info, job_status=job_status(root), failures=failures,
                  memory={k:v for k,v in pairs(read('/proc/meminfo')).items() if k in ('MemTotal','MemAvailable','SwapTotal','SwapFree')},
                  processes=jobs, cgroups=cgroups(jobs), kernel=kernel_evidence(failures),
                  notes=['No model/audio loaded; no worker launched; no training process modified.',
                         'Current memory/event counters are not a reconstruction of the failure.'])
    if Path('/dev/shm').exists():
        usage = shutil.disk_usage('/dev/shm')
        report['shm'] = dict(total=usage.total, free=usage.free)
    path = root/'exp'/('worker_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True)+'\n', encoding='ascii')
    print('WORKER_AUDIT='+str(path))
    print('JOB_STATUS='+json.dumps(report['job_status'], ensure_ascii=True))
    print('STEPS_AGE_SECONDS='+str(info.get('steps_age_seconds')))
    print('ACTIVE_CONFIG='+json.dumps(info.get('config', {}), ensure_ascii=True))
    print('RECENT_STEPS='+json.dumps(info.get('recent', {})))
    print('HOST_MEMORY='+json.dumps(report['memory']))
    print('RUN_METADATA_BYTES='+json.dumps(info.get('metadata_bytes', {})))
    print('FAILED_WORKER_PIDS='+json.dumps([r['worker_pids'] for r in failures]))
    for r in jobs:
        print('ACTIVE_PROCESS='+json.dumps({k:r[k] for k in ('pid','parent','name','rss','pss','threads')}, ensure_ascii=True))
    for group, values in report['cgroups'].items():
        print('CGROUP='+json.dumps(dict(path=group, values=values), ensure_ascii=True))
    print('KERNEL_EVIDENCE='+json.dumps(report['kernel'], ensure_ascii=True))
    print('TRAINING_UNCHANGED=True')


if __name__ == '__main__':
    main()
