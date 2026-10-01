"""Prepare one shared cache, then run matched control/candidate experiments."""
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import zipfile
from live_progress import phase, publish
from w2v_aasist.launch import ROOT, run_lock, check_environment, upload_archive
from w2v_aasist.full_workflow import ensure_idle, find_ffmpeg
from w2v_aasist.runtime import atomic_json, sha256
from .config import parser, configuration


def export_report(run, destination, upload=False):
    run, destination = Path(run), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/(run.name+'_report.zip')
    temporary = archive.with_suffix('.zip.tmp')
    # Only experiment diagnostics. Audio, weights and optimizer state are excluded.
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob('*')):
            if path.is_symlink():
                continue
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.md', '.log'):
                path.resolve().relative_to(run.resolve())
                z.write(path, path.relative_to(run).as_posix())
    os.replace(temporary, archive)
    print('REPORT_ZIP='+str(archive), flush=True)
    if upload:
        url = upload_archive(archive)
        (run/'temp_download_url.txt').write_text(url+'\n', encoding='utf-8')
    return archive


def compare(run, cfg):
    """Compare protected winners, retaining fallback and explicit success flags."""
    from .control import success_criteria
    arms = {}
    for arm in ('control', 'candidate'):
        path = Path(run)/arm/'completed.json'
        if not path.is_file():
            continue
        done = json.loads(path.read_text(encoding='utf-8'))
        selection = done['selection']
        arms[arm] = {'selected': selection['best_safe'], 'anchor': selection['anchor'],
                     'checkpoint': str(path.parent/'best_model.pt')}
    lines = ['# V3.3 matched source-paired experiments', '',
             'Fixed Dev proxies at threshold 0.5; not official platform scores.',
             'Identical source exposure and update budget; auxiliary losses add computation.', '',
             '| Arm | Tag | Clean | Noisy | Weighted | Noisy EN real | Noisy EN fake |',
             '|---|---|---:|---:|---:|---:|---:|']
    for arm, result in arms.items():
        q = result['selected']
        lines.append('| '+ ' | '.join([arm, q['tag'], *[f'{100*q[k]:.3f}' for k in
            ('clean','noisy','weighted','noisy_en_real','noisy_en_fake')]])+' |')
    result = {'arms': arms, 'status': 'complete' if len(arms)==len(cfg['arms']) else 'partial'}
    if arms:
        winner = max(arms, key=lambda name: (arms[name]['selected']['weighted'], arms[name]['selected']['noisy']))
        result.update(selected_arm=winner, checkpoint=arms[winner]['checkpoint'])
        # A baseline fallback cannot become evidence for the new method.
        q = arms[winner]['selected']; base = arms[winner]['anchor']
        result['practical_success'] = success_criteria(q, base, cfg)
        if 'candidate' in arms and 'control' in arms:
            c, a = arms['candidate']['selected'], arms['control']['selected']
            result['candidate_minus_control_pp'] = {k:100*(c[k]-a[k]) for k in
                ('weighted','noisy','clean','noisy_en_real','noisy_en_fake')}
            lines += ['', 'Candidate minus control (percentage points): '+json.dumps(result['candidate_minus_control_pp'])]
        lines += ['', 'Selected single-model checkpoint: '+result['checkpoint'],
                  'Practical success: '+str(result['practical_success']),
                  'A small eligible gain is retained but is not automatically a successful robustness result.']
    atomic_json(Path(run)/'comparison.json', result)
    temporary = Path(run)/'report.md.tmp'
    temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    os.replace(temporary, Path(run)/'report.md')
    return result


def _train_arm(run, arm, cfg, smoke_steps=0):
    folder = Path(run)/arm
    folder.mkdir(exist_ok=True)
    arm_cfg = dict(cfg, arm=arm)
    if arm == 'control':
        arm_cfg['pair_weight'] = 0.
    target = folder/'config.json'
    if target.is_file():
        if json.loads(target.read_text(encoding='utf-8')) != arm_cfg:
            raise ValueError('Saved arm configuration differs: '+str(target))
    else:
        atomic_json(target, arm_cfg)
    if (folder/'completed.json').is_file():
        print('V33_ARM_ALREADY_COMPLETE='+arm, flush=True)
        return 0
    print('V33_ARM='+arm, flush=True)
    # Existing viewer uses this stable event protocol and keeps both arms' history.
    print('V32_RUN='+str(folder), flush=True)
    command = [sys.executable, '-u', '-m', 'w2v_v33.train', '--config', str(target), '--out', str(folder)]
    if (folder/'last.pt').is_file():
        command += ['--resume', str(folder/'last.pt')]
    if smoke_steps:
        command += ['--smoke-steps', str(smoke_steps)]
    logfile = folder/('execution_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
    with logfile.open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            env={**os.environ,'AASIST_PROGRESS_OWNER':str(os.getpid())})
        for line in child.stdout:
            stream.write(line); stream.flush(); print(line, end='', flush=True)
        return child.wait()


def main():
    args = parser().parse_args()
    check_environment(gpu=args.device.startswith('cuda'))
    ensure_idle()
    with run_lock(ROOT/'exp'/'.v33-workflow.lock'), run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            if args.prepare_only or args.smoke_steps:
                raise ValueError('Resume cannot be combined with prepare-only or smoke mode')
            run = Path(args.resume).expanduser().resolve()
            cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
            if cfg.get('version') != '3.3' or cfg.get('arm') not in ('both','control','candidate'):
                raise ValueError('Resume requires the V3.3 experiment root, not a prior version/arm directory')
            print('RESUME: saved recipe; completed arms skipped, partial arm resumes last validation boundary.', flush=True)
        else:
            phase('V3.3 checking fixed V3 best and preparing communication diversity')
            cfg = configuration(args)
            cfg['arms'] = ['control','candidate'] if args.arm=='both' else [args.arm]
            cfg['ffmpeg'] = find_ffmpeg(cfg, args.ffmpeg)
            from .cache import prepare_cache
            from .data import build_data
            run = ROOT/'exp'/('w2v_v33_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True, exist_ok=False)
            print('V33_RUN='+str(run), flush=True)
            prepare_cache(cfg, cfg['train_noisy_cache_v33'], args.cache_workers)
            phase('V3.3 validating complete source pairs, cache ownership and fixed Dev isolation')
            plan, validation, weights, counts, fingerprints = build_data(cfg)
            coverage = plan.coverage()
            atomic_json(run/'coverage.json', coverage)
            cfg['preparation_fingerprints'] = fingerprints
            atomic_json(run/'config.json', cfg)
            atomic_json(run/'source.json', {'source_run':cfg['source_run'],
                'warm_checkpoint':cfg['warm_checkpoint'], 'warm_checkpoint_sha256':cfg['warm_checkpoint_sha256'],
                'expected_warm_tag':cfg['expected_warm_tag'], 'paired_manifest':cfg['train_pair_manifest']})
            del plan, validation, weights, counts
            if not args.keep_old_caches and not args.smoke_steps:
                from .maintenance import retire_replaced_cache
                phase('V3.3 retiring only verified replaced full2 Train cache files')
                retire_replaced_cache(cfg, ROOT, ensure_idle, run)
            if args.prepare_only:
                print('V33_PREPARATION_COMPLETE=True RUN='+str(run), flush=True)
                return
        for path, expected in cfg.get('preparation_fingerprints', {}).items():
            if sha256(path) != expected:
                raise RuntimeError('Prepared input changed: '+path)
        for key, digest_key in (('warm_checkpoint','warm_checkpoint_sha256'),
                                ('baseline','baseline_sha256')):
            if digest_key not in cfg or sha256(cfg[key]) != cfg[digest_key]:
                raise RuntimeError('Protected checkpoint changed: '+cfg[key])
        if not args.smoke_steps:
            (ROOT/'exp'/'.latest_v33_run').write_text(str(run)+'\n', encoding='utf-8')
        code = 0
        try:
            for arm in cfg['arms']:
                phase('V3.3 '+arm+' arm: independent warm start from '+cfg['expected_warm_tag'])
                code = _train_arm(run, arm, cfg, args.smoke_steps)
                if code:
                    atomic_json(run/'failed.json', {'arm':arm,'exit_code':code})
                    break
                if not args.smoke_steps:
                    compare(run, cfg)
        finally:
            phase('V3.3 saving local comparison and diagnostic report')
            try:
                export_report(run, args.download_dir, args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED='+str(exc)+'; local reports: '+str(run), flush=True)
        publish('V3.3 complete' if not code else 'V3.3 failed; see saved diagnostics',
                status='complete' if not code else 'failed', force=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
