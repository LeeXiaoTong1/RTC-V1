"""Isolated Omni training: at most four epochs and complete provenance."""
from datetime import datetime
from pathlib import Path
import secrets
import traceback

from live_progress import publish
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.launch import run_lock
from w2v_v36.workflow import export_report
from w2v_v39.common import ROOT,atomic_json,read_json,digest
from .config import parser,configuration,verify_inputs
from .train import run_experiment


def report(run):
    if not (run/'reference_metrics.json').exists(): return
    cfg=read_json(run/'config.json')
    values=[dict(tag='V3.15 reference: '+cfg['reference_tag'],metrics=read_json(run/'reference_metrics.json'))]
    if (run/'training_history.json').exists(): values+=read_json(run/'training_history.json')
    text=['# V3.16 OmniASR W2V 7B + LoRA','',
        'Local fixed Dev results, not official Progress/Eval scores. No old detector weights imported.',
        'Omni SSL features + newly initialized layer fusion/MultiConv; head warmup then upper-layer LoRA.',
        'Official Train supplies all gradients. Same V3.15 source groups and 30/10/60 final view CE mass.',
        'Full utterances; no feature/audio disk cache. Persistent training workers and length-based physical batches.',
        'AP/AUC/EER/Recall/F1 from one scoring pass, fixed P(fake)>=0.5.',
        'No extra Offline/panel forward pass; guards apply to unchanged Online/Seen/Heldout Dev.',
        'No teacher, ASR decoding, language labels or TFCL module required at inference.',
        'best_weighted means best trained Omni; best_guarded can be absent. V3.15 remains separately preserved.','',
        '| State | Clean | Noisy | Weighted | Guarded |','|---|---:|---:|---:|---|']
    for item in values:
        m=item['metrics']; text.append('| '+item['tag']+' | '+' | '.join(f'{100*m[k]:.3f}' for k in ('clean_f1','noisy_f1','weighted_f1'))+' | '+str(item.get('eligible','reference'))+' |')
    (run/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    args=parser().parse_args(); ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'),run_lock(ROOT/'exp'/'.v316-workflow.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve(); cfg=read_json(run/'config.json'); verify_inputs(cfg)
            if args.source_run and Path(args.source_run).resolve()!=Path(cfg['source_run']):
                raise ValueError('Resume cannot change source metadata/reference')
            print('V316_RESUME=recorded configuration; uncommitted segment is replayed',flush=True)
        else:
            cfg=configuration(args)
            run=ROOT/'exp'/('w2v_v316_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True); atomic_json(run/'config.json',cfg)
        (ROOT/'exp'/'.latest_v316_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V316_RUN='+str(run),flush=True)
        try:
            if (run/'completed.json').is_file():
                from .state import load_resume
                done=read_json(run/'completed.json'); state=load_resume(run,cfg)
                if done.get('checkpoint_sha256')!=digest(run/'last.pt') or done.get('committed_updates')!=state['cursor']:
                    raise ValueError('Completed checkpoint digest/cursor mismatch')
                print('V316_ALREADY_COMPLETE=True',flush=True)
            else: run_experiment(cfg,run)
            (run/'failure.log').unlink(missing_ok=True); publish('V3.16 complete',status='complete',force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            publish('V3.16 failed; committed weights and old best retained',status='failed',force=True)
            raise
        finally:
            try: report(run); export_report(run,args.download_dir,args.upload_temp)
            except Exception as exc: print('REPORT_EXPORT_FAILED='+str(exc),flush=True)


if __name__=='__main__': main()
