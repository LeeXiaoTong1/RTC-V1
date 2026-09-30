"""One guarded production entrypoint; no writes to original audio or checkpoints."""
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from live_progress import phase
from w2v_aasist.launch import ROOT, find_source, run_lock, check_environment, export_report
from w2v_aasist.full_workflow import ensure_idle, find_ffmpeg
from w2v_aasist.full_cache import prepare
from w2v_aasist.cache_retirement import retire
from w2v_aasist.runtime import atomic_json, sha256
from .config import parser, configuration


def main():
    args = parser().parse_args()
    check_environment(gpu=args.device.startswith('cuda'))
    ensure_idle()
    run = None
    with run_lock(ROOT/'exp'/'.v3-workflow.lock'), run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            if args.prepare_only or args.warm_checkpoint or args.smoke_steps:
                raise ValueError('Resume cannot be combined with preparation, warm start or smoke mode')
            run = Path(args.resume).expanduser().resolve()
            cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
            if cfg.get('algorithm','').find('MultiConv') < 0 or not (run/'last.pt').is_file():
                raise ValueError('Resume requires an existing V3 run and last.pt')
            print('RESUME: saved configuration and exact validation-boundary state; CLI recipe overrides are ignored.',flush=True)
        else:
            source_path,source = find_source(ROOT/'exp',args.source_config)
            print('SOURCE_CONFIG='+str(source_path),flush=True)
            phase('V3 checking protected original encoder and recipe')
            cfg = configuration(source,args)
            phase('V3 preparing two full-duration noisy views')
            ffmpeg = find_ffmpeg(source,args.ffmpeg)
            manifest = args.noise_manifest or source.get('train_noise_manifest') or '/home/ubuntu/LXT/RTC/external_noise/noise_split/train.jsonl'
            # Keep the previously published generation recipe byte-for-byte compatible.
            # A validated existing full cache is reused, including resumable partial generation.
            prepare(source,cfg['train_caches'][0],manifest,ffmpeg,args.cache_workers)
            phase('V3 validating Train/Dev isolation and complete epoch coverage')
            from .data import build_data
            plan,validation,weights,counts,fingerprints = build_data(cfg)
            coverage = plan.coverage()
            print('V3_COVERAGE='+json.dumps(coverage),flush=True)
            if not args.keep_old_caches and not args.smoke_steps:
                phase('V3 retiring replaced Train caches; protected weights and Dev retained')
                retire(source,cfg,ROOT,ensure_idle)
            if sha256(cfg['baseline']) != cfg['baseline_sha256'] or any(sha256(p)!=h for p,h in fingerprints.items()):
                raise RuntimeError('Protected original or active cache metadata changed')
            atomic_json(ROOT/'exp'/'v3_prepared_config.json',cfg)
            atomic_json(ROOT/'exp'/'v3_coverage.json',coverage)
            if args.prepare_only:
                print('V3_PREPARATION_COMPLETE=True ORIGINAL_BEST_PRESERVED=True',flush=True)
                return
            # Keep large index dictionaries out of the training process's host memory budget.
            del plan,validation,weights,counts,fingerprints
            run = ROOT/'exp'/('w2v_v3_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False)
            atomic_json(run/'config.json',cfg)
            atomic_json(run/'coverage.json',coverage)
            atomic_json(run/'source.json',{'source_config':str(source_path),'ffmpeg':ffmpeg})
        (ROOT/'exp'/'.latest_v3_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V3_RUN='+str(run),flush=True)
        command = [sys.executable,'-u','-m','w2v_v3.train','--config',str(run/'config.json'),'--out',str(run)]
        if args.resume: command += ['--resume',str(run/'last.pt')]
        if args.smoke_steps: command += ['--smoke-steps',str(args.smoke_steps)]
        log = run/('execution_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
        with log.open('x',encoding='utf-8') as output:
            child = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding='utf-8',errors='replace',bufsize=1)
            for line in child.stdout:
                output.write(line);output.flush();print(line,end='',flush=True)
            code = child.wait()
        if code:
            atomic_json(run/'failed.json',{'exit_code':code,'log':str(log),'resume_boundary':'last.pt if present'})
        phase('V3 exporting diagnostic report')
        try:
            export_report(run,args.download_dir,args.upload_temp)
        except Exception as exc:
            print('REPORT_EXPORT_FAILED='+str(exc)+'; reports remain at '+str(run),flush=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
