"""Launch/resume V3.13; keep diagnostics even if training or report upload fails."""
from datetime import datetime
from pathlib import Path
import secrets
import traceback

from live_progress import publish
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.launch import run_lock
from w2v_v36.workflow import export_report
from w2v_v39.common import ROOT, atomic_json, read_json
from .config import parser, configuration, verify_inputs
from .state import load_selected
from .train import run_experiment


def report(run):
    run=Path(run)
    if not (run/'baseline_metrics.json').is_file():return
    cfg=read_json(run/'config.json')
    entries=[('starting_last',read_json(run/'baseline_metrics.json'),False)]
    history=read_json(run/'training_history.json') if (run/'training_history.json').is_file() else []
    entries += [(e['tag'],e['metrics'],e['promoted']) for e in history]
    lines=['# V3.13 paired discrimination and difficult-example learning','',
        'All Dev values are LOCAL proxies, not official Progress/Eval scores.',
        'Starting model: trained V3.12 LAST '+cfg['starting_tag']+'; SHA256 '+cfg['starting_checkpoint_sha256'],
        'Original base plus the starting LAST reconstruct the anchor; a fallback keeps this trained LAST, not old baseline.',
        'Two full-wave views per source; official Offline/Online plus existing simulated conditions; no newly generated audio.',
        'Each loss microgroup contains same-language/same-condition real and fake sources; positives only share the same original source.',
        'Correct/confident label-gated paired margin and normalized-feature preservation; bounded source difficulty weighting and matched-negative separation.',
        'Language adversary is weak auxiliary; independent probes are diagnostic, never a language-success requirement for best selection.',
        'Fresh Train reference sweep stores only per-recording scores and scalar norm; zero extra teacher encoder passes per optimizer update.',
        'P(fake), fixed threshold 0.5. Local Weighted=.3 Clean+.7 mean(Seen,Heldout).','',
        '| Checkpoint | Clean | Noisy | Weighted | EN online real | EN seen real | EN heldout real | Promoted |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for tag,value,promoted in entries:
        scores=[100*value[k] for k in ('clean_f1','noisy_f1','weighted_f1')]
        scores += [100*value['groups'][c+'/en']['recall'][1] for c in ('online','seen','heldout')]
        lines.append('| '+tag+' | '+' | '.join(f'{x:.3f}' for x in scores)+f' | {promoted} |')
    lines += ['', '| Checkpoint | Pair loss/update | Ranking loss/update | Seconds/update | Rejected reasons |',
              '|---|---:|---:|---:|---|']
    for entry in history:
        stats=entry['segment_training'];n=stats['updates']
        lines.append(f'| {entry["tag"]} | {stats["pair_loss"]/n:.5f} | {stats["ranking_loss"]/n:.5f} | '
                     f'{stats["seconds_per_update"]:.2f} | '+', '.join(entry['reasons'])+' |')
    if (run/'completed.json').is_file():
        done=read_json(run/'completed.json')
        lines += ['', 'Selected: '+done['selected'], 'Committed updates: '+str(done['committed_updates']),
                  'Export best or last explicitly; submission metadata binds every starting/checkpoint hash.']
    lines += ['', 'pair_diagnostic_*.json: treatment-induced mistakes on Train sources held out from this adaptation; earlier models saw Train.',
              'These diagnostics and independent language probes are not unseen-corpus or causal evidence.',
              'No training, threshold selection, or model fitting uses Progress/Eval predictions.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    args = parser().parse_args()
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'), run_lock(ROOT/'exp'/'.v313-workflow.lock'):
        ensure_idle()
        if args.resume:
            run = Path(args.resume).expanduser().resolve()
            cfg = read_json(run/'config.json')
            verify_inputs(cfg)
            if args.source_run and Path(args.source_run).expanduser().resolve() != Path(cfg['source_run']):
                raise ValueError('Resume cannot change the starting model')
            if args.device != 'auto' and args.device != cfg['device']:
                raise ValueError('Resume preserves device and numerical execution')
            print('V313_RESUME_POLICY=recorded configuration; replay steps after last committed validation if interrupted',flush=True)
        else:
            cfg = configuration(args)
            run = ROOT/'exp'/('w2v_v313_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False)
            atomic_json(run/'config.json',cfg)
        (ROOT/'exp'/'.latest_v313_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V313_RUN='+str(run),flush=True)
        try:
            if (run/'completed.json').is_file():
                load_selected(run)
                print('V313_ALREADY_COMPLETE=True',flush=True)
            else:
                run_experiment(cfg,run)
            (run/'failure.log').unlink(missing_ok=True)
            publish('V3.13 complete',status='complete',force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            publish('V3.13 failed; committed last.pt and submitted V3.12 LAST retained',status='failed',force=True)
            raise
        finally:
            try:
                report(run)
                export_report(run,args.download_dir,args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED='+str(exc)+'; local diagnostics retained',flush=True)


if __name__ == '__main__':
    main()
