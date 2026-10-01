"""Reuse existing complete caches and V3 best in a separate bounded adaptation run."""
from datetime import datetime
import json
from pathlib import Path
import secrets
import subprocess
import sys
from live_progress import phase, publish
from w2v_aasist.launch import ROOT, run_lock, check_environment, export_report
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.runtime import atomic_json
from .config import parser, configuration


def main():
    args = parser().parse_args()
    check_environment(gpu=args.device.startswith('cuda'))
    ensure_idle()
    with run_lock(ROOT/'exp'/'.v32-workflow.lock'), run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            if args.smoke_steps:
                raise ValueError('Smoke diagnostics cannot resume or modify an existing run')
            run = Path(args.resume).expanduser().resolve()
            cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
            if cfg.get('version') != '3.2' or not (run/'last.pt').is_file():
                raise ValueError('Resume requires a V3.2 directory with last.pt')
            print('RESUME: exact saved recipe and validation boundary; CLI recipe overrides ignored.', flush=True)
        else:
            phase('V3.2 checking original and selected V3 weights; no cache generation')
            cfg = configuration(args)
            run = ROOT/'exp'/('w2v_v32_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True, exist_ok=False)
            atomic_json(run/'config.json', cfg)
            atomic_json(run/'source.json', {'source_run':cfg['source_run'],
                'source_config_sha256':cfg['source_config_sha256'],
                'warm_checkpoint':cfg['warm_checkpoint'], 'warm_checkpoint_sha256':cfg['warm_checkpoint_sha256'],
                'cache_policy':'reuse complete two-view cache; no regeneration or deletion'})
        # A diagnostic run must never become the default submission source.
        if not args.smoke_steps:
            (ROOT/'exp'/'.latest_v32_run').write_text(str(run)+'\n', encoding='utf-8')
        print('V32_RUN='+str(run), flush=True)
        command = [sys.executable, '-u', '-m', 'w2v_v32.train', '--config', str(run/'config.json'), '--out', str(run)]
        if args.resume:
            command += ['--resume', str(run/'last.pt')]
        if args.smoke_steps:
            command += ['--smoke-steps', str(args.smoke_steps)]
        log = run/('execution_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
        with log.open('x', encoding='utf-8') as output:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace', bufsize=1)
            for line in child.stdout:
                output.write(line); output.flush(); print(line, end='', flush=True)
            code = child.wait()
        if code:
            atomic_json(run/'failed.json', {'exit_code':code, 'log':str(log), 'resume_boundary':'last.pt if present'})
        phase('V3.2 exporting diagnostic report')
        try:
            export_report(run, args.download_dir, args.upload_temp)
        except Exception as exc:
            print('REPORT_EXPORT_FAILED='+str(exc)+'; reports remain at '+str(run), flush=True)
        publish('V3.2 finished; report and validation metrics saved',
                status='failed' if code else 'complete', force=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
