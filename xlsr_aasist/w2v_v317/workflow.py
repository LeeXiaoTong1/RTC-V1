"""Independent V3.17 training with immutable previous runs and concise output."""
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
from .train import run_experiment
from .state import load_selected


def report(run):
    run=Path(run);cfg=read_json(run/'config.json')
    history=read_json(run/'training_history.json') if (run/'training_history.json').exists() else []
    lines=['# V3.17: LoRA and shared forensic TFCL','',
        'Original public w2v-BERT weights remain frozen. Fresh top-layer Q/V LoRA and fresh detector; no old LAST/Adam import.',
        'Normalized weighted multi-layer fusion; four MultiConv blocks; shared 128-dim forensic frames; whole-utterance statistics and dropout/linear classifier.',
        'Bidirectional temporal attention and channel CKA act on the exact post-MultiConv features used by classification.',
        'Square-root source quotas reduce minority repeats. CE and TFCL budgets remain 25% per EN/ZH x fake/real group.',
        'Official Offline/Online/generated-noisy views retain 10/50/40 CE mass (20/80 if Online absent). No new waveform cache.',
        'Fixed Train Online replay is seen-data diagnostics, never a held-out generalization claim. Fixed Dev selects best at threshold 0.5.',
        'AP/AUC/EER/Recall/F1 printed once per epoch; losses, learning rates, gradient reach, coverage and Train/Dev CE saved.',
        'Only current-run trained checkpoints can be best. One atomic file retains last, best, optimizer, auxiliary and RNG; public encoder referenced.',
        'These are project-specific adaptations and do not guarantee a particular leaderboard score.','',
        '| Epoch | Clean | Noisy | Weighted | Best updated |','|---|---:|---:|---:|---|']
    for e in history:
        lines.append('| '+e['tag']+' | '+' | '.join(f'{100*e["metrics"][k]:.3f}' for k in ('clean_f1','noisy_f1','weighted_f1'))+' | '+str(e['promoted'])+' |')
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    args=parser().parse_args();validate(args);ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).expanduser().resolve();cfg=read_json(run/'config.json');verify_inputs(cfg)
            if args.data_run and Path(args.data_run).expanduser().resolve()!=Path(cfg['data_run']):
                raise ValueError('Resume cannot change the data manifest')
            print('[Resume] recorded V3.17 configuration; replay from last complete committed epoch',flush=True)
        else:
            cfg=configuration(args)
            run=ROOT/'exp'/('w2v_v317_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False);atomic_json(run/'config.json',cfg)
        (ROOT/'exp'/'.latest_v317_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V317_RUN='+str(run),flush=True)
        try:
            if (run/'completed.json').exists():load_selected(run)
            else:run_experiment(cfg,run)
            (run/'failure.log').unlink(missing_ok=True)
            publish('V3.17 complete',status='complete',force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            publish('V3.17 failed; last committed epoch retained',status='failed',force=True)
            raise
        finally:
            try:report(run);export_report(run,args.download_dir,args.upload_temp)
            except Exception as exc:print('REPORT_EXPORT_FAILED='+str(exc)+'; local files retained',flush=True)


if __name__=='__main__':main()
