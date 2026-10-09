"""Launch/resume only the new detector; reports contain no model/audio uploads."""
from datetime import datetime
from pathlib import Path
import secrets
import traceback
import zipfile
from w2v_aasist.launch import run_lock,upload_archive
from w2v_aasist.full_workflow import ensure_idle
from .common import ROOT,atomic_json,read_json,digest
from .config import parser,validate,configuration,verify_inputs
from .train import run_experiment
from .audit import describe


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
        try:print('TEMP_DOWNLOAD_URL='+upload_archive(path),flush=True)
        except Exception as exc:print('REPORT_UPLOAD_FAILED='+str(exc)+'; local report retained',flush=True)
    return path


def main():
    args=parser().parse_args();validate(args);ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve();cfg=read_json(run/'config.json');verify_inputs(cfg)
            train=read_json(run/'train_rows.json');dev=read_json(run/'dev_rows.json')
            verify={str(run/n):h for n,h in cfg['saved_manifests'].items()};from .common import verify_files;verify_files(verify)
            print('[Resume] recorded V3.18 configuration; replay from last committed epoch',flush=True)
        else:
            cfg,train,dev=configuration(args)
            run=ROOT/'exp'/('w2v_v318_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False)
            atomic_json(run/'train_rows.json',train);atomic_json(run/'dev_rows.json',dev)
            cfg['saved_manifests']={n:digest(run/n) for n in ('train_rows.json','dev_rows.json')}
            atomic_json(run/'config.json',cfg);atomic_json(run/'input_audit.json',describe(train,cfg['seed']))
        (ROOT/'exp'/'.latest_v318_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V318_RUN='+str(run),flush=True)
        try:
            if not (run/'completed.json').exists():run_experiment(cfg,run,train,dev)
            (run/'failure.log').unlink(missing_ok=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
        finally:
            try:report(run,args.download_dir,args.upload_temp)
            except Exception as exc:print('REPORT_EXPORT_FAILED='+str(exc),flush=True)


if __name__=='__main__':main()
