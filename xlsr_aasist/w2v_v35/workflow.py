"""One fresh V3.5 run, fixed full Dev, lazy epoch noise, and a protected reference."""
from datetime import datetime
import json
from pathlib import Path
import secrets

from live_progress import publish
from w2v_aasist.launch import ROOT, run_lock, check_environment
from w2v_aasist.full_workflow import ensure_idle, find_ffmpeg
from w2v_aasist.runtime import atomic_json, sha256
from w2v_v33.workflow import export_report
from .config import parser, configuration
from . import pretrained


def verify_protected(cfg):
    for path, digest in ((cfg['baseline'], cfg['baseline_sha256']),
                         (cfg['reference_checkpoint'], cfg['reference_checkpoint_sha256'])):
        if sha256(path) != digest:
            raise RuntimeError('Protected checkpoint changed: '+path)
    for path, digest in cfg.get('source_provenance', {}).get('file_fingerprints', {}).items():
        if sha256(path) != digest:
            raise RuntimeError('Submitted model provenance changed: '+path)
    pretrained.verify(cfg)


def main():
    args = parser().parse_args()
    check_environment(gpu=args.device.startswith('cuda'))
    ensure_idle()
    with run_lock(ROOT/'exp'/'.v35-workflow.lock'), run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            if args.prepare_only or args.smoke_steps:
                raise ValueError('Resume cannot change into preparation/smoke mode')
            run = Path(args.resume).expanduser().resolve()
            cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
            if cfg.get('version') != '3.5':
                raise ValueError('Resume requires an existing V3.5 experiment')
        else:
            cfg = configuration(args)
            cfg['ffmpeg'] = find_ffmpeg(cfg, args.ffmpeg)
            pretrained.prepare(cfg)
            run = ROOT/'exp'/('w2v_v35_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True, exist_ok=False)
            cfg['run_dir'] = str(run)
            atomic_json(run/'config.json', cfg)
            atomic_json(run/'source.json', {'reference_checkpoint': cfg['reference_checkpoint'],
                'reference_checkpoint_sha256': cfg['reference_checkpoint_sha256'],
                'reference_tag': cfg['reference_tag'], 'pretrained_origin': cfg['pretrained_origin'],
                'source_provenance': cfg['source_provenance'],
                'note': 'Submitted weights are an evaluation-only reference; training starts from generic speech pretraining.'})
        verify_protected(cfg)
        print('V35_RUN='+str(run), flush=True)
        print('V32_RUN='+str(run), flush=True)
        print('V35_INITIALIZATION=generic w2v-BERT 2.0 + new MultiConv', flush=True)
        print('V35_REFERENCE_SHA256='+cfg['reference_checkpoint_sha256'], flush=True)
        print('V35_RECIPE=full views only; equal four-group budgets; Offline/Online/Noisy=10/30/60; hard/easy noisy=75/25', flush=True)
        if not args.smoke_steps:
            (ROOT/'exp'/'.latest_v35_run').write_text(str(run)+'\n', encoding='utf-8')
        failed = True
        try:
            if args.prepare_only:
                from .data import build_data
                plan, _, _, _, fingerprints = build_data(cfg)
                atomic_json(run/'prepared.json', {'data_fingerprints': fingerprints, 'coverage': plan.coverage()})
                print('V35_PREPARATION_COMPLETE=True; full Dev ready; Train noise generated lazily during reading.', flush=True)
            else:
                from .train import train
                last = run/'last.pt'
                train(cfg, run, last if last.is_file() else None, args.smoke_steps)
            verify_protected(cfg)
            failed = False
        except BaseException as exc:
            atomic_json(run/'failed.json', {'error': type(exc).__name__, 'detail': str(exc)})
            raise
        finally:
            try:
                export_report(run, args.download_dir, args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED='+str(exc)+'; local report: '+str(run), flush=True)
            publish('V3.5 failed; inspect saved diagnostics' if failed else 'V3.5 preparation complete' if args.prepare_only else 'V3.5 complete',
                    status='failed' if failed else 'complete', force=True)


if __name__ == '__main__':
    main()
