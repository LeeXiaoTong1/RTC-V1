"""Launch/resume V3.10; keep diagnostics even if training or report upload fails."""
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
    entries = [('baseline', read_json(run/'baseline_metrics.json'), False, False)]
    if (run/'training_history.json').is_file():
        entries += [(e['tag'],e['metrics'],e['promoted'],e['language_evidence']['evidence'])
                    for e in read_json(run/'training_history.json')]
    lines = ['# V3.10 feature language debias', '',
        'Local full-wave Online Dev proxy, not the official Progress/Eval score.',
        'Weighted = 0.3 Clean + 0.7 mean(Seen, Heldout); P(fake) threshold = 0.5.',
        'Frozen prefix referenced from original best. Last 8 encoder layers, MultiConv and feature adapter trained.',
        'Language probes use source-disjoint real Train partitions held out from this adaptation.', '',
        '| Checkpoint | Clean | Noisy | Weighted | EN online real | EN seen real | EN heldout real | Promoted | Probe evidence |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for tag, value, promoted, evidence in entries:
        scores = [100*value[k] for k in ('clean_f1','noisy_f1','weighted_f1')]
        scores += [100*value['groups'][c+'/en']['recall'][1] for c in ('online','seen','heldout')]
        lines.append('| '+tag+' | '+' | '.join(f'{v:.3f}' for v in scores)+f' | {promoted} | {evidence} |')
    if (run/'completed.json').is_file():
        done = read_json(run/'completed.json')
        lines += ['', 'Selected: '+done['selected'], 'Language-probe evidence: '+str(done['language_probe_evidence'])]
    lines += ['', 'Lower language-probe accuracy alone does not establish useful debiasing; classification must also improve.',
              'A performance improvement with no probe evidence is reported as fine-tuning improvement, not proven language removal.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    args = parser().parse_args()
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'), run_lock(ROOT/'exp'/'.v310-workflow.lock'):
        ensure_idle()
        if args.resume:
            run = Path(args.resume).expanduser().resolve()
            cfg = read_json(run/'config.json')
            verify_inputs(cfg)
            if args.source_run and Path(args.source_run).expanduser().resolve() != Path(cfg['source_run']):
                raise ValueError('Resume cannot change the starting model')
            if args.device != 'auto' and args.device != cfg['device']:
                raise ValueError('Resume preserves device and numerical execution')
            print('V310_RESUME_POLICY=recorded configuration; replay steps after last committed validation if interrupted',flush=True)
        else:
            cfg = configuration(args)
            run = ROOT/'exp'/('w2v_v310_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True,exist_ok=False)
            atomic_json(run/'config.json',cfg)
        (ROOT/'exp'/'.latest_v310_run').write_text(str(run)+'\n',encoding='utf-8')
        print('V310_RUN='+str(run),flush=True)
        try:
            if (run/'completed.json').is_file():
                load_selected(run)
                print('V310_ALREADY_COMPLETE=True',flush=True)
            else:
                run_experiment(cfg,run)
            (run/'failure.log').unlink(missing_ok=True)
            publish('V3.10 complete',status='complete',force=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
            publish('V3.10 failed; committed last.pt and original best retained',status='failed',force=True)
            raise
        finally:
            try:
                report(run)
                export_report(run,args.download_dir,args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED='+str(exc)+'; local diagnostics retained',flush=True)


if __name__ == '__main__':
    main()
