"""Additive V3.15.1 workflow; V3.15 and V3.16 execution remain untouched."""
from datetime import datetime
from pathlib import Path
import secrets
import traceback

from live_progress import publish
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.launch import run_lock
from w2v_v33.workflow import export_report
from w2v_v39.common import ROOT,atomic_json,read_json
from .config import parser,configuration,verify_inputs
from .state import load_selected
from .train import run_experiment,final_audit


def report(run):
    run=Path(run)
    if not (run/'baseline_metrics.json').is_file():
        return
    cfg=read_json(run/'config.json')
    history=read_json(run/'training_history.json') if (run/'training_history.json').is_file() else []
    entries=[dict(tag='starting_parent',metrics=read_json(run/'baseline_metrics.json'),promoted=[],
                  panel=read_json(run/'panel_baseline_tune.json'))]+history
    lines=['# V3.15.1 official Online anchored TFCL','',
        'All metrics here are Train/Dev diagnostics; no official Progress/Eval data enters training or selection.',
        'Pinned parent: '+cfg['parent_run']+' / '+cfg['parent_selector']+' / '+cfg['parent_selected_tag'],
        'Parent SHA256: '+cfg['parent_checkpoint_sha256'],
        'Existing w2v-BERT + MultiConv + adapter architecture; last eight encoder layers and all existing head parameters update.',
        '16 balanced sources / 48 full-wave views. Main CE mass: official Online 50%, simulated reference 10%, noisy RTC 40%.',
        'Two TFCL edges share those three forwards: official Online/reference and reference/noisy, with equal auxiliary budgets.',
        'Temporal comparisons use all valid frames; independent per-source channel CKA alone pools to 201 bins.',
        'Persistent worker processes and bounded RAM caches; no generated waveform/feature disk cache.',
        'Historical Weighted = .3 Online Clean + .7 mean(Seen,Heldout). Additional panel scores are not added to this metric.',
        'best_guarded requires fixed-Dev safety, broader panel improvement and ranking evidence, then uses a separate ranking score.',
        'Final-only audit processed views cannot alter checkpoint selection, learning rate or stopping.',
        'Audit source recordings also appear in historical Dev; this is not an independent unseen-source guarantee.','',
        '| State | Clean | Noisy | Historical Weighted | Selection panel F1 | Panel AUC | Saved as |',
        '|---|---:|---:|---:|---:|---:|---|']
    for entry in entries:
        m,p=entry['metrics'],entry.get('panel')
        values=[100*m[k] for k in ('clean_f1','noisy_f1','weighted_f1')]
        values += [100*p['macro_f1'],100*p['macro_auc']] if p else [float('nan')]*2
        lines.append('| '+entry['tag']+' | '+' | '.join(f'{x:.3f}' for x in values)+' | '+','.join(entry.get('promoted',[]))+' |')
    if (run/'completed.json').is_file():
        done=read_json(run/'completed.json')
        lines += ['','Selections: '+str(done['selections']),'Last: '+done['last_tag'],
            'Committed/planned updates: '+str(done['committed_updates'])+'/'+str(done['planned_updates'])]
    if (run/'post_selection_audit.json').is_file():
        a=read_json(run/'post_selection_audit.json')
        lines += ['','Final-only audit of '+a['selected']+': F1 '+f'{100*a["metrics"]["macro_f1"]:.3f}'+
                  ', parent F1 '+f'{100*a["baseline"]["macro_f1"]:.3f}'+'. Selection was not changed.']
    lines += ['','Detailed AP/AUC/EER/F1, class confusion/recall and matched-fake-recall are in training_history.json.',
        'Panel per-mechanism/language/noise/severity breakdowns identify failures hidden by an aggregate.',
        'AUC and matched recall are diagnostics; deployment threshold stays P(fake)>=0.5.',
        'sampling_epoch_*.json records Online exclusions, unique source coverage and repeats.',
        'Execution profiling rolls back model, auxiliary branch, Adam state and RNG, including nonempty Adam snapshots.',
        'Timing separates compute and input wait. Actual A100 speed and official accuracy require server execution.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    args=parser().parse_args();ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'),run_lock(ROOT/'exp'/'.v3151-workflow.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve();cfg=read_json(run/'config.json');verify_inputs(cfg)
            if args.parent_run and Path(args.parent_run).expanduser().resolve()!=Path(cfg['parent_run']):
                raise ValueError('Resume cannot change the pinned parent')
            if args.parent_checkpoint and args.parent_checkpoint!=cfg['parent_selector']:
                raise ValueError('Resume cannot change the selected parent checkpoint')
            if args.device!='auto' and args.device!=cfg['device']:
                raise ValueError('Resume preserves numerical execution and device')
            print('V3151_RESUME_POLICY=recorded config; replay only after last committed validation',flush=True)
        else:
            cfg=configuration(args)
            run=ROOT/'exp'/('w2v_v3151_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False);atomic_json(run/'config.json',cfg)
        (ROOT/'exp'/'.latest_v3151_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V3151_RUN='+str(run),flush=True)
        try:
            if (run/'completed.json').is_file():
                load_selected(run);print('V3151_ALREADY_COMPLETE=True',flush=True)
            else:
                run_experiment(cfg,run)
            final_audit(cfg,run)
            (run/'failure.log').unlink(missing_ok=True)
            publish('V3.15.1 complete',status='complete',force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            publish('V3.15.1 failed; parent and committed states retained',status='failed',force=True)
            raise
        finally:
            try:
                report(run);export_report(run,args.download_dir,args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED='+str(exc)+'; local diagnostics retained',flush=True)


if __name__=='__main__':main()
