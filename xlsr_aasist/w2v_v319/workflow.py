"""Additive V3.19 workflow; V3.15 and V3.16 execution remain untouched."""
from datetime import datetime
from pathlib import Path
import secrets
import sys
import traceback

from live_progress import publish
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.launch import run_lock
from w2v_v39.common import ROOT,atomic_json,read_json
from .config import parser,configuration,verify_inputs,validate_locked_cli
from . import delivery,reports
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
    lines=['# V3.19 received-audio consistency and targeted EN6 augmentation','',
        'All metrics here are Train/Dev diagnostics; no official Progress/Eval data enters training or selection.',
        'Pinned parent: '+cfg['parent_run']+' / '+cfg['parent_selector']+' / '+cfg['parent_selected_tag'],
        'Parent SHA256: '+cfg['parent_checkpoint_sha256'],
        'Existing w2v-BERT + MultiConv + adapter architecture; last eight encoder layers and all existing head parameters update.',
        '16 sources / 48 augmented views; quotas EN-fake/EN-real/ZH-fake/ZH-real = '+str(cfg['group_quotas'])+'. Each group has 25% loss mass.',
        'Main CE mass: official Online (or independent clean RTC fallback) 30%, simulated clean reference 10%, noisy RTC 60%. Raw Offline never enters the detector.',
        'EN total fixed versions = 6/6; ZH = 1/3. The planned '+str(cfg['epochs'])+' epochs cover every EN version on both available streams; original-source count does not increase.',
        'Consistency mode: '+cfg['consistency']+'. Matched/bridge auxiliary mass: '+str(cfg['tfcl_matched_mass'])+'/'+str(cfg['tfcl_bridge_mass'])+'. Missing/cropped bridges are masked without reallocating their mass.',
        'This parameter-free TFCL-inspired redesign is not a reproduction of the published learned cross-attention module. Matched time loss uses native frames; bridge alignment is bounded to 192 bins; channel CKA uses at most 201 bins.',
        'Initial LR encoder/head/adapter/output: '+str([cfg[k] for k in ('encoder_lr','head_lr','adapter_lr','output_lr')])+'. Layer decay '+str(cfg['layer_decay'])+'. No warmup; cosine floor '+str(cfg['lr_floor'])+'.',
        'Persistent worker processes and bounded RAM caches; no generated waveform/feature disk cache.',
        'Historical Weighted = .3 Online Clean + .7 mean(Seen,Heldout). Additional panel scores are not added to this metric.',
        'Automatic submission defaults to best_weighted. best_guarded additionally requires fixed-Dev safety and scene-panel evidence; both selectors retain an explicit parent fallback.',
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
        'sampling_epoch_*.json records missing Online fallbacks, unique source/version coverage, repeats and minimum coverage budget.',
        'Train and Dev noise recordings are disjoint. RTC mechanisms are shared, not held-out-platform validation.',
        'Gradient comparisons describe the first physical microbatch of each epoch; they are not whole-epoch gradient estimates.',
        'Execution profiling rolls back model, auxiliary branch, Adam state and RNG, including nonempty Adam snapshots.',
        'Timing separates compute and input wait. Actual A100 speed and official accuracy require server execution.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    argv=sys.argv[1:];args=parser().parse_args(argv);validate_locked_cli(args,argv);ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'),run_lock(ROOT/'exp'/'.v319-workflow.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve();cfg=read_json(run/'config.json');verify_inputs(cfg)
            print('V319_RESUME_POLICY=recorded config; replay only after last committed validation',flush=True)
        else:
            cfg=configuration(args)
            run=ROOT/'exp'/('w2v_v319_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False);atomic_json(run/'config.json',cfg)
        (ROOT/'exp'/'.latest_v319_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V319_RUN='+str(run),flush=True)
        submission=delivery.prepare_request(args,run,cfg)
        report_request=reports.prepare_request(args,run,argv)
        errors=[]
        try:
            if (run/'completed.json').is_file():
                load_selected(run);print('V319_ALREADY_COMPLETE=True',flush=True)
            else:
                run_experiment(cfg,run)
            try:
                final_audit(cfg,run,(submission or {}).get('checkpoint_kind','best_weighted'))
            except Exception as exc:
                errors.append('Final audit: '+str(exc))
                (run/'audit_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            # A failed diagnostic cannot prevent delivery of a committed best.
            try:
                delivery.finish_request(run,cfg,submission)
            except Exception as exc:
                errors.append('Submission: '+str(exc))
                (run/'submission_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            publish('V3.19 failed; parent and committed states retained',status='failed',force=True)
            raise
        finally:
            try:
                report(run)
                reports.report(run,report_request['destination'],report_request['upload'])
            except Exception as exc:
                errors.append('Report: '+str(exc))
                print('REPORT_EXPORT_FAILED='+str(exc)+'; local diagnostics retained',flush=True)
        if errors:
            raise RuntimeError('Training checkpoints retained; '+'; '.join(errors))
        (run/'failure.log').unlink(missing_ok=True)
        publish('V3.19 complete',status='complete',force=True)


if __name__=='__main__':main()
