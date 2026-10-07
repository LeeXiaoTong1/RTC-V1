"""Launch and resume two-stage supervised adaptation with visible results."""
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
    run = Path(run)
    if not (run/'baseline_metrics.json').is_file():
        return
    cfg = read_json(run/'config.json')
    history = read_json(run/'training_history.json') if (run/'training_history.json').is_file() else []
    entries = [dict(tag='starting_last', metrics=read_json(run/'baseline_metrics.json'), promoted=[])] + history
    lines = ['# V3.14 supervised output adaptation and balanced fine-tuning', '',
        'All metrics are local fixed Dev proxies, not official Progress/Eval scores.',
        'Starting checkpoint: actual V3.12 LAST '+cfg['starting_tag']+'; SHA256 '+cfg['starting_checkpoint_sha256'],
        'Stage A fits only the final binary Linear on fresh starting-LAST Train vectors, using CE plus anchored parameter L2.',
        'Stage B updates the last eight encoder layers, MultiConv, existing bounded adapter and BOTH classifier layers.',
        'Only official Train supplies gradients. Dev validates/selects; no Progress/Eval fitting or tuning.',
        'Ordinary/Noisy classification mass is 50/50, with each EN/ZH x fake/real group receiving 25%.',
        'No pair, ranking, language adversary, teacher, retention or feature-scale loss.',
        'P(fake), fixed threshold 0.5; Weighted=.3 Online Clean+.7 mean(Seen,Heldout). Offline is diagnostic only.', '',
        '| State | Clean | Noisy | Weighted | EN online real | EN seen real | EN heldout real | Saved as |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for entry in entries:
        value = entry['metrics']
        numbers = [100*value[k] for k in ('clean_f1', 'noisy_f1', 'weighted_f1')]
        numbers += [100*value['groups'][c+'/en']['recall'][1] for c in ('online', 'seen', 'heldout')]
        lines.append('| '+entry['tag']+' | '+' | '.join(f'{v:.3f}' for v in numbers)+' | '+','.join(entry.get('promoted', []))+' |')
    lines += ['', 'best_weighted and best_guarded are independent aliases. LAST remains the last committed joint state.',
              'The same selected tensor storage is shared; the frozen encoder prefix is never copied.',
              'training_history.json records old errors rescued/new errors, AUC, matched-fake-recall and timing.']
    if (run/'completed.json').is_file():
        done = read_json(run/'completed.json')
        lines += ['', 'Selections: '+str(done['selections']), 'Last: '+done['last_tag'],
                  'Committed/planned joint updates: '+str(done['committed_updates'])+'/'+str(done['planned_updates'])]
    (run/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    args = parser().parse_args()
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'), run_lock(ROOT/'exp'/'.v314-workflow.lock'):
        ensure_idle()
        if args.resume:
            run = Path(args.resume).expanduser().resolve()
            cfg = read_json(run/'config.json')
            verify_inputs(cfg)
            if args.source_run and Path(args.source_run).expanduser().resolve() != Path(cfg['source_run']):
                raise ValueError('Resume cannot change the starting model')
            if args.device != 'auto' and args.device != cfg['device']:
                raise ValueError('Resume preserves device and numerical execution')
            print('V314_RESUME_POLICY=recorded configuration; replay only after the last committed validation', flush=True)
        else:
            cfg = configuration(args)
            run = ROOT/'exp'/('w2v_v314_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True, exist_ok=False)
            atomic_json(run/'config.json', cfg)
        (ROOT/'exp'/'.latest_v314_run').write_text(str(run)+'\n', encoding='utf-8')
        print('V314_RUN='+str(run), flush=True)
        try:
            if (run/'completed.json').is_file():
                load_selected(run)
                print('V314_ALREADY_COMPLETE=True', flush=True)
            else:
                run_experiment(cfg, run)
            (run/'failure.log').unlink(missing_ok=True)
            publish('V3.14 complete', status='complete', force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
            publish('V3.14 failed; source and committed checkpoint retained', status='failed', force=True)
            raise
        finally:
            try:
                report(run)
                export_report(run, args.download_dir, args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED='+str(exc)+'; local diagnostics retained', flush=True)


if __name__ == '__main__':
    main()
