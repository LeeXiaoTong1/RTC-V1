"""Launch/resume only the new detector; reports contain no model/audio uploads."""
from datetime import datetime
from pathlib import Path
import secrets
import traceback
import sys
from w2v_aasist.launch import run_lock
from .processes import ensure_idle
from .common import ROOT,RECIPE,atomic_json,read_json,digest
from .config import parser,validate,validate_locked_cli,configuration,verify_inputs
from .train import run_experiment
from .sampling import describe
from .delivery import prepare_request,finish_request
from .reports import report,prepare_request as prepare_report_request


def main():
    args=parser().parse_args();validate(args);validate_locked_cli(args,sys.argv[1:]);ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve();cfg=read_json(run/'config.json')
            if cfg.get('version')!='3.18.8' or cfg.get('recipe')!=RECIPE:
                raise ValueError('--resume accepts only V3.18.8 runs; use --parent-run to initialize from V3.18.7 best')
            verify_inputs(cfg)
            train=read_json(run/'train_rows.json');dev=read_json(run/'dev_rows.json')
            verify={str(run/n):h for n,h in cfg['saved_manifests'].items()};from .common import verify_files;verify_files(verify)
            print('[Resume] recorded V3.18.8 configuration; replay from saved optimizer boundary',flush=True)
        else:
            cfg,train,dev=configuration(args)
            run=ROOT/'exp'/('w2v_v3188_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False)
            atomic_json(run/'train_rows.json',train);atomic_json(run/'dev_rows.json',dev)
            cfg['saved_manifests']={n:digest(run/n) for n in ('train_rows.json','dev_rows.json')}
            atomic_json(run/'config.json',cfg)
            atomic_json(run/'input_audit.json',describe(train,cfg['seed'],cfg['stream_sources'],cfg['group_quotas'],cfg['view_factors'],cfg['refresh_base_variants'],cfg=cfg))
        (ROOT/'exp'/'.latest_v3188_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V3188_RUN='+str(run),flush=True)
        # Existing user workflow expects BOTH temporary links when submission upload is requested.
        if args.upload_submission:args.upload_temp=True
        report_request=prepare_report_request(args,run,sys.argv[1:])
        outcome=None
        try:
            request=prepare_request(args,run,cfg)
            if not (run/'completed.json').exists() or read_json(run/'completed.json')['status']!='complete':
                outcome=run_experiment(cfg,run,train,dev,baseline_only=args.baseline_only)
                if outcome['status'] in ('paused','initialization_verified'):
                    (run/'failure.log').unlink(missing_ok=True)
                    return
            if request and not args.baseline_only:
                print('[Submission] training complete; export this run best and deliver requested scores',flush=True)
                finish_request(run,cfg,request)
            (run/'failure.log').unlink(missing_ok=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
        finally:
            already_failed=sys.exc_info()[0] is not None
            # Pause/preflight are nonterminal: exit promptly after the saved
            # checkpoint, without a potentially long network upload. Manual
            # best submission remains available after an acknowledged pause.
            if already_failed or not outcome or outcome.get('status') not in ('paused','initialization_verified'):
                try:report(run,report_request['destination'],report_request['upload'])
                except Exception as exc:
                    print('REPORT_EXPORT_FAILED='+str(exc),flush=True)
                    if not already_failed:raise


if __name__=='__main__':main()
