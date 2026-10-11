"""One-terminal detached launcher, read-only viewer, worker and upload retry."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT/'exp'/'.latest_v318_model_audit'


def process_identity(pid):
    """Read Linux process start ticks; distinguish exited/reused PIDs."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def finish_report(out, upload):
    from .report import render, package
    render(out); archive = package(out)
    print('REPORT_ZIP='+str(archive), flush=True)
    if upload:
        try:
            from w2v_aasist.launch import upload_archive
            url = upload_archive(archive)
            (out/'temp_download_url.txt').write_text(url+'\n', encoding='utf-8')
            write(out/'upload_status.json', dict(status='uploaded', url=url))
        except Exception:
            write(out/'upload_status.json', dict(status='failed', error=traceback.format_exc()))
            print('UPLOAD_FAILED=True; local report retained. Retry with:', flush=True)
            print('python -m v318_model_audit --upload-only '+str(out), flush=True)


def worker(out):
    out = Path(out).resolve()
    request = json.loads((out/'request.json').read_text(encoding='utf-8'))
    code = 1
    # Native subprocesses inherit this detached process group, never the viewer's.
    def terminate(signum, _): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        from .runner import execute
        result = execute(request, out)
        code = 0 if result['status'] == 'complete' and result['original_unchanged'] else 1
    except BaseException:
        error = traceback.format_exc(); traceback.print_exc()
        path = out/'result.json'
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        result.update(status='failed', errors=result.get('errors', [])+[error])
        write(path, result)
    finally:
        print('AUDIT_COMPUTE_EXIT='+str(code), flush=True)
        try: finish_report(out, request['upload_temp'])
        except Exception: traceback.print_exc(); code = 1
        write(out/'exit.json', dict(exit_code=code))
        print('AUDIT_EXIT='+str(code), flush=True)
    return code


def watch(out):
    out = Path(out).resolve()
    print('AUDIT_DIR='+str(out)+'\nRead-only viewer; Ctrl+C closes the viewer only.', flush=True)
    offset = 0
    try:
        while True:
            log = out/'audit.log'
            if log.is_file():
                with log.open('rb') as f:
                    f.seek(offset); chunk = f.read(); offset = f.tell()
                if chunk:
                    print(chunk.decode('utf-8', errors='replace'), end='', flush=True)
            if (out/'exit.json').exists():
                # exit.json can precede the final stdout flush; no information depends on that last line.
                status = json.loads((out/'exit.json').read_text(encoding='utf-8'))
                print('AUDIT_FINISHED='+str(status['exit_code']), flush=True)
                return status['exit_code']
            if os.name == 'posix' and (out/'process.json').is_file():
                proc = json.loads((out/'process.json').read_text(encoding='utf-8'))
                identity = process_identity(proc['pid'])
                if identity is None or (proc.get('start_ticks') and identity != proc['start_ticks']):
                    print('AUDIT_PROCESS_LOST=True; no completed result. Inspect audit.log; do not treat as success.', flush=True)
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nViewer closed; audit continues. Reopen: bash watch_v318_model_audit.sh', flush=True)
        return 0


def launch(args):
    if os.name != 'posix': raise RuntimeError('Launch on the Linux training server')
    run = Path(args.run).expanduser().resolve()
    if not (run/'last.pt').is_file() or not (run/'completed.json').is_file():
        raise FileNotFoundError('Completed source run with last.pt is required: '+str(run))
    if args.workers < 0 or args.bn_batches < 1 or args.update_steps < 1:
        raise ValueError('workers must be nonnegative; diagnostic counts must be positive')
    out = ROOT/'exp'/('v318_model_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
    out.mkdir(parents=True, exist_ok=False)
    request = dict(run=str(run), workers=args.workers, bn_batches=args.bn_batches,
                   update_steps=args.update_steps, device=args.device, upload_temp=args.upload_temp)
    write(out/'request.json', request)
    files = sorted(Path(__file__).parent.glob('*.py'))+[ROOT/'v318_diagnostics.py']
    write(out/'audit_code_sha256.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with (out/'audit.log').open('ab', buffering=0) as log:
        process = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-u', '-m',
            'v318_model_audit', '--worker', str(out)], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    write(out/'process.json', dict(pid=process.pid, start_ticks=process_identity(process.pid)))
    LATEST.write_text(str(out)+'\n', encoding='utf-8')
    print('AUDIT_PID='+str(process.pid)+'\nAUDIT_DIR='+str(out), flush=True)
    # Drop source-run arguments before exec: legacy idle scanners must not see
    # the read-only viewer as another training process.
    if sys.stdout.isatty() and not args.no_watch:
        os.execv(sys.executable, [sys.executable, '-u', '-m', 'v318_model_audit', '--watch', str(out)])
    print('VIEW=bash watch_v318_model_audit.sh', flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--run'); modes.add_argument('--worker', help=argparse.SUPPRESS)
    modes.add_argument('--watch', nargs='?', const='latest'); modes.add_argument('--upload-only')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--bn-batches', type=int, default=128)
    p.add_argument('--update-steps', type=int, default=100)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--upload-temp', action='store_true')
    p.add_argument('--no-watch', action='store_true')
    args = p.parse_args()
    if args.worker: return worker(args.worker)
    if args.watch: return watch(LATEST.read_text().strip() if args.watch == 'latest' else args.watch)
    if args.upload_only:
        out = Path(args.upload_only).resolve()
        if not (out/'exit.json').is_file(): raise ValueError('Upload retry requires a finished audit')
        finish_report(out, True); return 0
    if not args.run: p.error('--run is required to start an audit')
    return launch(args)


if __name__ == '__main__': sys.exit(main())
