"""Reuse the completed V3.3 data; train one submission-anchored V3.4 model."""
from datetime import datetime
import json
from pathlib import Path
import secrets
from live_progress import publish
from w2v_aasist.launch import ROOT, run_lock, check_environment
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.runtime import atomic_json, sha256
from w2v_v33.workflow import export_report
from .config import parser, configuration


def verify_protected(cfg):
    for key, digest in (('baseline', 'baseline_sha256'),
                        ('warm_checkpoint', 'warm_checkpoint_sha256')):
        if not cfg.get(digest) or sha256(cfg[key]) != cfg[digest]:
            raise RuntimeError('Protected checkpoint changed: ' + str(cfg.get(key)))
    for path, digest in cfg.get('source_provenance', {}).get('file_fingerprints', {}).items():
        if sha256(path) != digest:
            raise RuntimeError('Submitted model provenance changed: ' + path)


def main():
    args = parser().parse_args()
    check_environment(gpu=args.device.startswith('cuda'))
    ensure_idle()
    with run_lock(ROOT/'exp'/'.v34-workflow.lock'), run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            if args.prepare_only or args.smoke_steps:
                raise ValueError('Resume uses the saved recipe; cannot combine with preparation/smoke mode')
            run = Path(args.resume).expanduser().resolve()
            cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
            if cfg.get('version') != '3.4':
                raise ValueError('Resume requires a V3.4 run, not V3.3')
            print('RESUME: V3.4 saved recipe and last committed validation boundary.', flush=True)
        else:
            cfg = configuration(args)
            run = ROOT/'exp'/('w2v_v34_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
            run.mkdir(parents=True, exist_ok=False)
            atomic_json(run/'config.json', cfg)
            atomic_json(run/'source.json', {
                'warm_checkpoint': cfg['warm_checkpoint'],
                'warm_checkpoint_sha256': cfg['warm_checkpoint_sha256'],
                'expected_warm_tag': cfg['expected_warm_tag'],
                'source_provenance': cfg.get('source_provenance', {}),
                'note': 'The screenshot is user-reported platform evidence, never a Dev training target.'})
        verify_protected(cfg)
        print('V34_RUN='+str(run), flush=True)
        # Keep the established viewer event protocol; no duplicate progress loop.
        print('V32_RUN='+str(run), flush=True)
        print('V34_WARM_CHECKPOINT='+cfg['warm_checkpoint'], flush=True)
        print('V34_WARM_SHA256='+cfg['warm_checkpoint_sha256'], flush=True)
        print('V34_WARM_TAG='+cfg['expected_warm_tag'], flush=True)
        print('V34_SOURCE_ARM='+cfg['arm'], flush=True)
        print('V34_SOURCE_BASELINE_FALLBACK='+str(cfg['expected_warm_tag']=='baseline'), flush=True)
        print('V34_SINGLE_RUN=True; existing V3.3 caches are read-only; no control arm is launched.', flush=True)
        if not args.smoke_steps:
            (ROOT/'exp'/'.latest_v34_run').write_text(str(run)+'\n', encoding='utf-8')
        failed = True
        try:
            if args.prepare_only:
                from w2v_v33.data import build_data
                plan, _, _, _, fingerprints = build_data(cfg)
                if fingerprints != cfg.get('source_data_fingerprints'):
                    raise RuntimeError('Prepared data differs from the submitted V3.3 inputs')
                atomic_json(run/'planned_coverage.json', plan.coverage())
                atomic_json(run/'preparation_inputs.json', fingerprints)
                print('V34_PREPARATION_COMPLETE=True; no optimizer update performed.', flush=True)
            else:
                from .train import train
                resume = run/'last.pt'
                train(cfg, run, resume if resume.is_file() else None, args.smoke_steps)
            verify_protected(cfg)
            failed = False
        except BaseException as exc:
            atomic_json(run/'failed.json', {'error': type(exc).__name__, 'detail': str(exc)})
            raise
        finally:
            try:
                export_report(run, args.download_dir, args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED='+str(exc)+'; local reports: '+str(run), flush=True)
            publish('V3.4 failed; inspect saved diagnostics' if failed else 'V3.4 preparation complete' if args.prepare_only else 'V3.4 complete',
                    status='failed' if failed else 'complete', force=True)


if __name__ == '__main__':
    main()
