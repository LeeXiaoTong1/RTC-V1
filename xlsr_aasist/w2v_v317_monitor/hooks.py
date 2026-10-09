"""Attach observations after the original atomic epoch save, without changing training."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import sys
import traceback
from w2v_v39.common import digest
from w2v_v317.data import bundles
from .panel import evaluate_panel
from .history import collect,console_summary
from .render import write_artifacts
from .metrics import table_lines


@contextmanager
def attach():
    from w2v_v317 import train
    original=train.save
    original_execution=train.select_execution
    display=sys.stdout
    def observe(run,cfg,model,tag):
        try:
            with bundles(cfg) as (dataset,_):
                result=evaluate_panel(model,dataset['rows'],cfg,run,tag,digest(Path(run)/'last.pt'))
            report=collect(run);path=write_artifacts(report,Path(run)/'diagnostics')
            print('\n'.join(table_lines('[Train FULL] '+tag+'; one fixed Noisy view per original source',result['metrics'])),flush=True)
            print('[Fit] '+report['fit']['text'],flush=True)
            if 'recall_gap_pp' in report['fit']:
                print(f'  Full Online Train–Dev recall gap={report["fit"]["recall_gap_pp"]:.2f} pp; '
                      f'Dev–Train CE gap={report["fit"]["ce_gap"]:.4f}',flush=True)
            print(f'[Train evaluation] full sweep={result["seconds"]/60:.1f} min; curves={path}',flush=True)
        except Exception as exc:
            folder=Path(run)/'diagnostics';folder.mkdir(exist_ok=True)
            (folder/('failure_'+tag+'.log')).write_text(traceback.format_exc(),encoding='utf-8')
            print('[Diagnostics failed] '+str(exc)+'; trained epoch is already saved; no invented Train metrics',flush=True)
    def observed_save(run,cfg,state,model,auxiliary,optimizer):
        original(run,cfg,state,model,auxiliary,optimizer)
        if state['last_tag'].startswith('epoch_'):observe(run,cfg,model,state['last_tag'])
    def observed_execution(model,auxiliary,optimizer,plan,cfg,run):
        result=original_execution(model,auxiliary,optimizer,plan,cfg,run)
        # A resumed original run may not have observed its last saved epoch.
        # Parameters and RNG were already restored by the original trainer.
        if (Path(run)/'last.pt').exists():
            state=train.load_resume(run,cfg)
            if state['last_tag'].startswith('epoch_'):
                with redirect_stdout(display),redirect_stderr(display):
                    observe(run,cfg,model,state['last_tag'])
        return result
    train.save=observed_save
    train.select_execution=observed_execution
    try:yield
    finally:train.save=original;train.select_execution=original_execution
