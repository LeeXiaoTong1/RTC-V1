"""Production continuation; previous versions and completed runs remain immutable."""
from datetime import datetime
from pathlib import Path
import secrets
import traceback
from live_progress import publish
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.launch import run_lock
from w2v_v33.workflow import export_report
from w2v_v39.common import ROOT,atomic_json,read_json
from .arguments import parser,validate
from .config import configuration,verify_inputs
from .train import run_experiment,final_diagnostics
from .state import load_selected


def report(run):
    run=Path(run);cfg=read_json(run/'config.json')
    history=read_json(run/'training_history.json') if (run/'training_history.json').exists() else []
    lines=['# V3.16.1 continuation','',
        'Training starts from V3.16 LAST: '+cfg['source_run']+' / '+cfg['source_last_tag'],
        'Detector Adam restored; new four-epoch schedule; new train-only SSL attention/projection.',
        'Full SSL features, bidirectional soft attention and both-side gradients; no confidence gate or saliency.',
        'Complete valid sequence CKA per source; shared temporal projection. Only structure arm pools to 201 bins.',
        'Paper objectives adapted to full variable-duration audio and source-balanced microbatches; not a bit-identical paper reproduction.',
        'Offline/Online/Noisy CE=10/50/40; missing Online=20/80. Original Train augmentation distribution preserved.',
        'Best: complete fixed Dev Weighted only, among genuinely trained V3.16/continuation states. Threshold P(fake)>=0.5.',
        'AP/AUC/EER/recall and class counts remain diagnostics; no Progress/Eval scores enter selection.',
        'Per-step losses and per-language/label/edge coverage: training_steps.jsonl; whole epoch totals: training_history.json.',
        'Additional local audit is diagnostic only; official pair treatment is unavailable without verified Dev mapping.',
        'No generated audio/feature disk cache; frozen SSL prefix referenced; one atomic current/best/optimizer checkpoint.','',
        '| Checkpoint | Clean | Noisy | Weighted | Best updated |',
        '|---|---:|---:|---:|---|']
    for e in history:
        m=e['metrics']
        lines.append('| '+e['tag']+' | '+' | '.join(f'{100*m[k]:.3f}' for k in ('clean_f1','noisy_f1','weighted_f1'))+' | '+str(e['promoted'])+' |')
    if (run/'completed.json').exists():
        done=read_json(run/'completed.json')
        lines+=['','Best: '+done['best_tag'],'Last: '+done['last_tag'],
            'Completed additional epochs: '+str(done['additional_epochs'])]
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    args=parser().parse_args();validate(args);ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve();cfg=read_json(run/'config.json');verify_inputs(cfg)
            if args.source_run and Path(args.source_run).expanduser().resolve()!=Path(cfg['source_run']):
                raise ValueError('Resume cannot change the source run')
            print('[Resume] recorded configuration; replay from last fully committed epoch',flush=True)
        else:
            cfg=configuration(args)
            run=ROOT/'exp'/('w2v_v3161_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False);atomic_json(run/'config.json',cfg)
        (ROOT/'exp'/'.latest_v3161_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V3161_RUN='+str(run),flush=True)
        if cfg.get('source_code_migrations'):
            print('[Compatibility] approved V3.16 argument-only update; source checkpoint and data unchanged',flush=True)
            atomic_json(run/'source_code_migrations.json',cfg['source_code_migrations'])
        try:
            if (run/'completed.json').exists(): load_selected(run)
            else: run_experiment(cfg,run)
            # Training/selection is already committed. A diagnostic failure must
            # not invalidate a usable checkpoint or rerun finished epochs.
            try:
                final_diagnostics(cfg,run)
            except Exception:
                (run/'diagnostic_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
                print('[Audit] diagnostic failed; trained checkpoint retained; see diagnostic_failure.log',flush=True)
            (run/'failure.log').unlink(missing_ok=True)
            publish('V3.16.1 complete',status='complete',force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            publish('V3.16.1 failed; last committed epoch retained',status='failed',force=True)
            raise
        finally:
            try: report(run);export_report(run,args.download_dir,args.upload_temp)
            except Exception as exc: print('REPORT_EXPORT_FAILED='+str(exc)+'; local files retained',flush=True)


if __name__=='__main__':main()
