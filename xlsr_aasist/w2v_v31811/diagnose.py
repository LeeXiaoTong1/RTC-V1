"""V3.18.11 read-only EN-real diagnostic for existing V3.18.10 models.

Scores mode: no GPU, uses stored per-epoch Select and locked Audit scores.
Features mode: one frozen Omni model; Train-only linear probes and bias fitting.
No Progress/Eval inference, checkpoint training, or Dev-fitted calibration.
"""
import argparse
import json
import traceback
from pathlib import Path
from .scores import analyze


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('scores','features','all'),default='scores')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run',type=Path,help='Original V3.18.10 server run directory')
    source.add_argument('--report-zip',type=Path,help='Existing V3.18.10 report.zip (scores mode only)')
    p.add_argument('--out',type=Path,default=None,help='Separate, new diagnostic output directory')
    p.add_argument('--checkpoint',choices=('best','last'),default='best')
    p.add_argument('--max-train-per-class',type=int,default=64)
    p.add_argument('--max-dev-per-class',type=int,default=64)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--upload-temp',action='store_true',help='Package outputs and upload to temp.sh')
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp',help='Local ZIP destination before temp.sh upload')
    p.add_argument('--include-audit-features',action='store_true',
        help='Descriptive use only; historical Audit has already been exposed during development')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode in ('features','all') and args.run is None:
        raise ValueError('GPU feature probes require --run and its saved detector checkpoint')
    if args.workers<0:raise ValueError('workers must be nonnegative')
    source=args.report_zip or args.run
    out=args.out or (args.run/'v31811_diagnostics' if args.run else Path.cwd()/'v31811_diagnostics')
    out=out.expanduser().resolve()
    if args.run and (out==args.run.resolve() or out==args.run.resolve()/'last.pt'):
        raise ValueError('Do not overwrite the original training run')
    out.mkdir(parents=True,exist_ok=True)
    overview={}
    try:
        if args.mode in ('scores','all'):
            overview['scores']=analyze(source,out/'scores')
        if args.mode in ('features','all'):
            from .probes import run
            overview['features']=run(args.run,out/'features',args.max_train_per_class,
                 args.max_dev_per_class,args.workers,args.checkpoint,args.include_audit_features)
        (out/'completed.json').write_text(json.dumps(overview,indent=2),encoding='utf8')
        (out/'failure.json').unlink(missing_ok=True)
        print('V31811_COMPLETE='+str(out),flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc(),partial_outputs=overview),indent=2),encoding='utf8')
        # The completed.json of an older successful run must not masquerade as
        # completion of a failed new diagnosis in the same output directory.
        (out/'completed.json').unlink(missing_ok=True)
        if args.upload_temp:
            try:
                from .delivery import deliver
                deliver(out,args.download_dir)
            except Exception as upload_exc:
                print('V31811_PARTIAL_UPLOAD_FAILED='+str(upload_exc),flush=True)
        raise
    if args.upload_temp:
        from .delivery import deliver
        deliver(out,args.download_dir)



if __name__=='__main__':
    main()
