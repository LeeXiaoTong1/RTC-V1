"""Run the three matched arms sequentially, with durable stop/resume state."""
import argparse
import codecs
from datetime import datetime, timezone
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
from urllib.parse import urlparse
import zipfile
from .ce import CE_WEIGHTS, classification_weights


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('baseline', 'expanded', 'hard')
SCHEMA = 'rtc_v3191_sequential_suite_v1'
RESERVED = frozenset(('--resume', '--arm', '--baseline-only', '--submit-best',
                      '--preflight-only',
                      '--upload-submission', '--submission-protocol', '--submission-audio-root',
                      '--submission-out', '--submission-output', '--submission-destination'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(part)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare_manifest(path, parent, common_args, epochs, seed, *, upload_reports=False):
    """Persist immutable parent bytes and exact shared arguments before work."""
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path.exists():
        raise FileExistsError('Suite already exists; use --resume-suite ' + str(path))
    files = ('last.pt', 'config.json', 'train_rows.json', 'dev_rows.json', 'dev_partition.json')
    origin = {str(parent/name): checksum(parent/name) for name in files}
    immutable = dict(parent=str(parent), parent_files=origin, common_args=list(common_args),
                     epochs=int(epochs), sampling_seed=int(seed), arms=list(ARMS),
                     classification_ce_weights=dict(CE_WEIGHTS),
                     upload_reports=bool(upload_reports))
    path.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, created_at=now(), updated_at=now(),
                    immutable=immutable, immutable_sha256=signature(immutable), status='pending',
                    arms={name: dict(status='pending', run=None, attempts=0) for name in ARMS})
    write(path/'suite.json', manifest)
    return manifest


def validate_manifest(manifest, *, verify_parent=True):
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Not a V3.19.1 matched-arm suite')
    immutable = manifest['immutable']
    if signature(immutable) != manifest['immutable_sha256']:
        raise ValueError('Immutable suite configuration changed')
    if tuple(immutable['arms']) != ARMS or set(manifest['arms']) != set(ARMS):
        raise ValueError('Suite must contain exactly baseline, expanded and hard')
    if verify_parent:
        for path, expected in immutable['parent_files'].items():
            if checksum(path) != expected:
                raise ValueError('Pinned parent file changed: ' + path)


def validate_child(run, arm, manifest):
    run = Path(run).resolve()
    cfg = read(run/'config.json')
    expected = manifest['immutable']
    if (cfg.get('version') != '3.19.1' or cfg.get('robust_arm') != arm
            or cfg.get('epochs') != expected['epochs']
            or cfg.get('sampling_seed', cfg.get('seed')) != expected['sampling_seed']):
        raise ValueError('Child arm/version/epoch/seed differs from the immutable suite')
    if cfg.get('classification_ce_weights') != expected.get('classification_ce_weights'):
        raise ValueError('Child CE allocation differs from the immutable suite')
    origin = cfg.get('parent_init', {})
    parent = expected['parent']
    if (str(Path(origin.get('run', '')).resolve()) != parent
            or origin.get('checkpoint_sha256') != expected['parent_files'][str(Path(parent)/'last.pt')]):
        raise ValueError('Child did not start from the pinned parent checkpoint')
    return cfg


def recover_record(path, arm, manifest):
    state = manifest['arms'][arm]
    record = Path(path)/(arm+'.run.json')
    if record.exists():
        found = str(Path(read(record)['run']).resolve())
        if state.get('run') and str(Path(state['run']).resolve()) != found:
            raise ValueError('Suite run record disagrees with persisted arm state')
        validate_child(found, arm, manifest)
        state['run'] = found
    elif state.get('run'):
        validate_child(state['run'], arm, manifest)
    return state.get('run')


def completed(run, arm, manifest):
    if not run or not (Path(run)/'completed.json').is_file():
        return False
    validate_child(run, arm, manifest)
    value = read(Path(run)/'completed.json')
    return (value.get('status') == 'complete' and
            value.get('completed_epochs') == manifest['immutable']['epochs'])


def request_stop(path):
    path = Path(path).expanduser().resolve()
    manifest = read(path/'suite.json')
    validate_manifest(manifest, verify_parent=False)
    write(path/'stop_requested.json', dict(requested_at=now(), reason='user_requested_suite_pause'))
    print('SUITE_STOP_REQUESTED='+str(path), flush=True)
    print('The current arm will save at a training boundary; the next arm will not start.', flush=True)


class StopState:
    def __init__(self, path):
        self.path = Path(path)
        self.signaled = False

    def signal(self, *_):
        self.signaled = True

    def requested(self):
        if self.signaled and not (self.path/'stop_requested.json').exists():
            write(self.path/'stop_requested.json', dict(requested_at=now(), reason='suite_signal'))
        return self.signaled or (self.path/'stop_requested.json').exists()


def execute(command, env, log, stopped):
    """Stop through the child's checkpoint request, never kill training."""
    with Path(log).open('ab', buffering=0) as output:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, close_fds=True)
        # Keep the suite viewer useful while preserving each arm's own log.
        # A regular file avoids pipe backpressure or readline blocking when
        # training emits no newline during a long startup operation.
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        with Path(log).open('rb') as reader:
            while child.poll() is None:
                chunk = reader.read(65536)
                if chunk:
                    print(decoder.decode(chunk), end='', flush=True)
                stopped()
                time.sleep(.5)
            for chunk in iter(lambda: reader.read(65536), b''):
                print(decoder.decode(chunk), end='', flush=True)
            print(decoder.decode(b'', final=True), end='', flush=True)
        stopped()
        return child.returncode


def render_comparison(value):
    for item in value.get('comparisons', []):
        score = item['results']['audit']['metrics']['score']
        print(f'{item["arm"]}: Audit delta={score["delta"]*100:+.3f} pp; '
              f'95% source interval=[{score["low"]*100:+.3f}, {score["high"]*100:+.3f}] pp', flush=True)
    print('Historical Dev comparison; this is not a probability of benchmark improvement.', flush=True)


def package_report(path, manifest, *, uploader=None):
    """Archive metrics and selected Dev scores; never audio or checkpoint bytes."""
    path = Path(path).resolve()
    archive = path/'report.zip'
    status_file = path/'report_status.json'
    old = read(status_file) if status_file.exists() else {}
    # Transient delivery timestamps/errors do not change the report content
    # and do not force another upload when retrying an already uploaded ZIP.
    snapshot = {k: v for k, v in manifest.items() if k not in ('updated_at', 'last_error')}
    suite_bytes = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, indent=2).encode('utf-8')
    entries = [('suite.json', suite_bytes), ('comparison.json', path/'comparison.json')]
    names = ('config.json', 'training_history.json', 'robust_final.json', 'robust_panel.json',
             'robust_selection_locked.json', 'completed.json', 'initialization_audit.json',
             'input_audit.json', 'source_registry.json', 'parent_baseline.json')
    for arm in ARMS:
        run = Path(manifest['arms'][arm]['run']).resolve()
        candidates = [run/name for name in names]
        candidates.extend(sorted(run.glob('robust_scores_*.npz')))
        for file in candidates:
            if not file.is_file():
                continue
            if file.is_symlink() or file.resolve().parent != run:
                raise ValueError('Suite report input must be a regular file inside its arm')
            entries.append((arm+'/'+file.name, file))
    inventory = [dict(file=name, sha256=(hashlib.sha256(value).hexdigest()
                                        if isinstance(value, bytes) else checksum(value)),
                      bytes=len(value) if isinstance(value, bytes) else value.stat().st_size)
                 for name, value in entries]
    identity = signature(inventory)
    reusable = (archive.exists() and old.get('content_identity') == identity
                and old.get('zip_sha256') == checksum(archive))
    if not reusable:
        temporary = archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as output:
            for (name, value), row in zip(entries, inventory):
                if isinstance(value, bytes):
                    output.writestr(name, value)
                else:
                    output.write(value, name)
                    if value.stat().st_size != row['bytes'] or checksum(value) != row['sha256']:
                        raise ValueError('Suite report input changed while packaging: '+name)
            output.writestr('REPORT_MANIFEST.json', json.dumps(
                dict(schema=SCHEMA, content_identity=identity, files=inventory), sort_keys=True, indent=2))
        os.replace(temporary, archive)
    archive_hash = checksum(archive)
    status = dict(status='exported', archive=str(archive), zip_sha256=archive_hash,
                  content_identity=identity, file_count=len(entries), uploaded=False, url=None)
    write(status_file, status)
    print('SUITE_REPORT_ZIP='+str(archive), flush=True)
    if manifest['immutable'].get('upload_reports', False):
        previous_url = urlparse(old.get('url') or '')
        if (old.get('zip_sha256') == archive_hash and previous_url.scheme == 'https'
                and previous_url.netloc == 'temp.sh' and previous_url.path not in ('', '/', '/upload')):
            url = old['url']
        else:
            if uploader is None:
                from w2v_aasist.launch import upload_archive
                uploader = upload_archive
            try:
                url = uploader(archive)
                parsed = urlparse(url)
                if parsed.scheme != 'https' or parsed.netloc != 'temp.sh' or parsed.path in ('', '/', '/upload'):
                    raise ValueError('Uploader returned an unexpected temporary download URL')
                if checksum(archive) != archive_hash:
                    raise ValueError('Suite archive changed during upload')
            except Exception as exc:
                write(status_file, dict(status, status='failed', error=str(exc)))
                raise RuntimeError('Suite report upload failed; all completed arms are retained. '
                                   'Retry --resume-suite '+str(path)) from exc
        status.update(status='complete', uploaded=True, url=url)
        write(status_file, status)
        (path/'report_temp_download_url.txt').write_text(url+'\n', encoding='ascii')
        print('SUITE_REPORT_URL='+url, flush=True)
        print('REPORT_URL='+url, flush=True)
    return status


def run_suite(path, *, executor=execute, comparer=None, stop=None):
    """A failed/paused child never advances to the following experimental arm."""
    path = Path(path).resolve()
    manifest = read(path/'suite.json')
    validate_manifest(manifest)
    classification_weights(manifest['immutable'])
    stop = stop or StopState(path)
    manifest['status'] = 'running'

    def persist():
        manifest['updated_at'] = now()
        write(path/'suite.json', manifest)

    persist()
    try:
        for arm in ARMS:
            state = manifest['arms'][arm]
            run = recover_record(path, arm, manifest)
            if stop.requested():
                manifest['status'] = 'paused'
                persist()
                print('SUITE_PAUSED='+str(path), flush=True)
                return manifest
            if completed(run, arm, manifest):
                state['status'] = 'complete'
                persist()
                continue
            validate_manifest(manifest)
            # Never construct a fresh run after a recorded run already exists.
            # --resume also recovers an initialized run that has not yet saved
            # its first optimizer checkpoint.
            shared = manifest['immutable']['common_args']
            arguments = ['--resume', run] if run else [*shared, '--arm', arm]
            command = [sys.executable, '-X', 'faulthandler', '-u', '-m',
                       'w2v_v3191.workflow', *arguments]
            env = dict(os.environ, V3191_SUITE_PARENT_PID=str(os.getpid()),
                       V3191_SUITE_RUN_RECORD=str(path/(arm+'.run.json')))
            state.update(status='running', attempts=state['attempts']+1, started_at=now())
            log = path/(arm+f'.attempt{state["attempts"]}.log')
            state['log'] = str(log)
            persist()
            print(f'SUITE_ARM={arm} attempt={state["attempts"]} LOG={log}', flush=True)
            stop_forwarded = False

            def forward_stop():
                nonlocal stop_forwarded
                if not stop.requested():
                    return False
                child_run = recover_record(path, arm, manifest)
                if child_run and not stop_forwarded:
                    write(Path(child_run)/'stop_requested.json',
                          dict(requested_at=now(), reason='matched_suite_pause'))
                    stop_forwarded = True
                    persist()
                    print('SUITE_CHILD_PAUSE_REQUESTED='+str(child_run), flush=True)
                return True

            code = executor(command, env, log, forward_stop)
            state['exit_code'] = int(code)
            state['finished_at'] = now()
            run = recover_record(path, arm, manifest)
            done = completed(run, arm, manifest)
            paused = bool(run and (Path(run)/'paused.json').is_file())
            # An explicit suite pause wins even if the child concurrently
            # finishes its final epoch. Resume will then skip that completed arm.
            if stop.requested() or paused:
                state['status'] = 'complete' if done else 'paused'
                manifest['status'] = 'paused'
                persist()
                print('SUITE_PAUSED='+str(path), flush=True)
                return manifest
            if code != 0 or not done:
                state['status'] = 'failed'
                manifest['status'] = 'failed'
                state['failure_reason'] = ('child_exit_'+str(code) if code else
                                           'child_did_not_write_completed_epoch_budget')
                persist()
                print('SUITE_FAILED='+str(path)+' ARM='+arm, flush=True)
                return manifest
            state['status'] = 'complete'
            persist()
        if stop.requested():
            manifest['status'] = 'paused'
            persist()
            return manifest
        if comparer is None:
            from .compare import compare_runs
            comparer = compare_runs
        value = comparer([manifest['arms'][arm]['run'] for arm in ARMS],
                         draws=2000, seed=3191001)
        write(path/'comparison.json', value)
        render_comparison(value)
        manifest.update(status='complete', comparison=str(path/'comparison.json'), report=str(path/'report.zip'))
        manifest.setdefault('completed_at', now())
        manifest.pop('last_error', None)
        persist()
        package_report(path, manifest)
        print('SUITE_COMPLETE='+str(path), flush=True)
        print('COMPARISON='+str(path/'comparison.json'), flush=True)
        return manifest
    except BaseException:
        manifest['status'] = 'failed'
        manifest['last_error'] = traceback.format_exc()
        persist()
        raise


def parser():
    value = argparse.ArgumentParser(description=__doc__,
        epilog='New suites also accept ordinary training options, such as --parent-run, --epochs (default 2), '
               '--seed and --workers. Resume accepts no training overrides. No submission is scheduled.')
    group = value.add_mutually_exclusive_group()
    group.add_argument('--resume-suite', metavar='DIRECTORY')
    group.add_argument('--stop-suite', metavar='DIRECTORY')
    value.add_argument('--suite-root', metavar='NEW_DIRECTORY')
    return value


def main(argv=None):
    arguments, shared = parser().parse_known_args(argv)
    if arguments.stop_suite:
        if shared or arguments.suite_root:
            raise ValueError('--stop-suite accepts only an existing suite directory')
        request_stop(arguments.stop_suite)
        return
    if arguments.resume_suite:
        if shared or arguments.suite_root:
            raise ValueError('Resume uses immutable saved arguments; no training overrides are permitted')
        path = Path(arguments.resume_suite).expanduser().resolve()
        manifest = read(path/'suite.json')
        validate_manifest(manifest)
        classification_weights(manifest['immutable'])
        # Explicit resume authorizes clearing only this suite's old stop request.
        (path/'stop_requested.json').unlink(missing_ok=True)
    else:
        for token in shared:
            if token.split('=', 1)[0] in RESERVED:
                raise ValueError('This option is controlled/disabled by the matched suite: '+token)
        from .config import parser as training_parser, validate
        from .parent import resolve
        args = training_parser().parse_args(shared)
        validate(args)
        if args.resume or args.baseline_only or args.submit_best or args.upload_submission:
            raise ValueError('Suite controls child resume and epoch budget; preflight/submission flags are disabled')
        parent = resolve(args.parent_run)
        seed = args.seed if args.seed is not None else int(read(parent/'config.json')['seed'])
        # Append canonical values so all children have an explicit, shared
        # parent, seed and epoch budget even when launch options omit them.
        common = [*shared, '--parent-run', str(parent), '--epochs', str(args.epochs), '--seed', str(seed)]
        path = Path(arguments.suite_root).expanduser().resolve() if arguments.suite_root else (
            ROOT/'exp'/('v3191_suite_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2)))
        prepare_manifest(path, parent, common, args.epochs, seed, upload_reports=args.upload_temp)
    from w2v_aasist.launch import run_lock
    from .processes import ensure_idle
    ensure_idle()
    stop = StopState(path)
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, stop.signal)
    (ROOT/'exp').mkdir(parents=True, exist_ok=True)
    (ROOT/'exp'/'.latest_v3191_suite').write_text(str(path)+'\n', encoding='utf-8')
    print('V3191_SUITE='+str(path), flush=True)
    with run_lock(path/'.suite.lock'):
        outcome = run_suite(path, stop=stop)
    if outcome['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
