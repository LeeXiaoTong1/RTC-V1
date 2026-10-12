"""Launch/resume only the new detector; reports contain no model/audio uploads."""
from datetime import datetime
from pathlib import Path
import secrets
import traceback
import zipfile
import sys
from w2v_aasist.launch import run_lock,upload_archive
from w2v_aasist.full_workflow import ensure_idle
from .common import ROOT,RECIPE,atomic_json,read_json,digest
from .config import parser,validate,validate_locked_cli,configuration,verify_inputs
from .train import run_experiment
from .sampling import describe
from .delivery import prepare_request,finish_request


def report(run,destination,upload=False):
    run=Path(run);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    path=destination/(run.name+'_report.zip')
    temporary=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.log','.md','.csv','.html','.npz'):
                z.write(p,p.relative_to(run))
    temporary.replace(path)
    print('REPORT_ZIP='+str(path),flush=True)
    if upload:
        try:upload_archive(path)
        except Exception as exc:print('REPORT_UPLOAD_FAILED='+str(exc)+'; local report retained',flush=True)
    return path


def main():
    args=parser().parse_args();validate(args);validate_locked_cli(args,sys.argv[1:]);ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve();cfg=read_json(run/'config.json')
            if cfg.get('version')!='3.18.4' or cfg.get('recipe')!=RECIPE:
                raise ValueError('--resume accepts only this AUG5 recipe; use --parent-run for R2')
            verify_inputs(cfg)
            train=read_json(run/'train_rows.json');dev=read_json(run/'dev_rows.json')
            verify={str(run/n):h for n,h in cfg['saved_manifests'].items()};from .common import verify_files;verify_files(verify)
            print('[Resume] recorded V3.18.4 configuration; replay from last committed epoch',flush=True)
        else:
            cfg,train,dev=configuration(args)
            run=ROOT/'exp'/('w2v_v3184_aug5_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False)
            atomic_json(run/'train_rows.json',train);atomic_json(run/'dev_rows.json',dev)
            cfg['saved_manifests']={n:digest(run/n) for n in ('train_rows.json','dev_rows.json')}
            atomic_json(run/'config.json',cfg);atomic_json(run/'input_audit.json',describe(train,cfg['seed'],cfg['stream_sources'],cfg['group_quotas'],cfg['view_factor']))
        (ROOT/'exp'/'.latest_v3184_aug5_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V3184_RUN='+str(run),flush=True)
        try:
            request=prepare_request(args,run,cfg)
            if not (run/'completed.json').exists() or read_json(run/'completed.json')['status']!='complete':
                run_experiment(cfg,run,train,dev,baseline_only=args.baseline_only)
            if request and not args.baseline_only:
                print('[Submission] training complete; export this run best and deliver requested scores',flush=True)
                finish_request(run,cfg,request)
            (run/'failure.log').unlink(missing_ok=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
        finally:
            try:report(run,args.download_dir,args.upload_temp)
            except Exception as exc:print('REPORT_EXPORT_FAILED='+str(exc),flush=True)


if __name__=='__main__':main()
