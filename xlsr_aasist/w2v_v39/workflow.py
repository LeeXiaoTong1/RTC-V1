"""Train adaptive corrections using immutable completed V3.7/V3.8 provenance."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import traceback

import torch

from .common import ROOT, announce, atomic_json, digest, read_json, save_small, verify_files
from .config import parser, configuration, runtime_versions
from .data import borrowed_bundles, base_weights, replay_base
from .deployment import evaluate_candidates, report
from .fit import fit_all
from .patch import save_patch, load_selected


class Stages:
    def __init__(self, run, cfg):
        self.path = Path(run) / 'stages'
        self.identity = hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def __call__(self, name, compute):
        path, marker = self.path / (name + '.pt'), self.path / (name + '.json')
        if marker.is_file():
            meta = read_json(marker)
            if (meta.get('identity') != self.identity or not path.is_file()
                    or digest(path) != meta.get('sha256')):
                raise ValueError('Compact fitting stage changed: ' + name)
            print('V39_STAGE_REUSE=' + name, flush=True)
            return torch.load(path, map_location='cpu', weights_only=True)
        result = compute()
        save_small(path, result)
        atomic_json(marker, dict(identity=self.identity, sha256=digest(path)))
        return result


def verify_inputs(cfg):
    verify_files(cfg['code_fingerprints'])
    verify_files(cfg['source_fingerprints'])
    verify_files(cfg['data_fingerprints'])
    verify_files({cfg['base_checkpoint']: cfg['base_checkpoint_sha256']})
    if cfg['runtime_versions'] != runtime_versions():
        raise ValueError('Fitting runtime changed; start a new V3.9 run instead of reusing compact stages')


def run_experiment(cfg, run):
    verify_inputs(cfg)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg['device'].startswith('cuda'):
        torch.cuda.set_device(torch.device(cfg['device']))
    run = Path(run)
    if shutil.disk_usage(run).free < 128 * 1024**2:
        raise OSError('Need 128 MiB free for compact stages and reports; no old files were deleted')
    announce('V3.9 verifying the submitted base and completed source artifacts')
    print('V39_SOURCE=' + cfg['source_run'] + '; source_version=' + cfg['source_version'], flush=True)
    weight, bias = base_weights(cfg)
    with borrowed_bundles(cfg) as (train, dev, paths):
        atomic_json(run / 'cache_reuse.json', dict(paths=paths, borrowed_read_only=True,
                    train_rows=len(train['rows']), dev_rows=len(dev['rows']), new_audio_bytes=0, new_feature_bytes=0))
        print(f'V39_CACHE_REUSE Train={len(train["rows"])} Dev={len(dev["rows"])}; new audio/features=0 bytes', flush=True)
        print('V39_FIT_DEVICE=' + cfg['device'] + '; math_threads=1; frozen w2v-BERT and MultiConv', flush=True)
        # Binding checks are performed before any optimization. They do not select hyperparameters.
        replay_base(train, weight, bias, cfg['device'])
        baseline_logits = replay_base(dev, weight, bias, cfg['device'])
        announce('V3.9 Train-only language fitting and adaptive hard-example repair')
        result = fit_all(train, weight, bias, cfg, Stages(run, cfg))
        announce('V3.9 final fixed Dev replay, calibration comparison, and EN/ZH guardrails')
        result, spec = evaluate_candidates(result, dev, baseline_logits, weight, bias, cfg, run)
    verify_inputs(cfg)
    report(run, result)
    patch = save_patch(run / 'best_patch.pt', cfg, result['selected'], spec)
    done = dict(version='3.9', status='complete', selected=result['selected'],
        baseline_fallback=patch['baseline_fallback'], language_debias_applied=patch['language_debias_applied'],
        source_run=cfg['source_run'], source_version=cfg['source_version'],
        base_checkpoint=cfg['base_checkpoint'], base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        patch_sha256=digest(run / 'best_patch.pt'), external_teacher_at_inference=False)
    atomic_json(run / 'completed.json', done)
    print('V39_SELECTED=' + done['selected'], flush=True)
    print('V39_BASELINE_FALLBACK=' + str(done['baseline_fallback']), flush=True)
    return done


def main():
    from live_progress import publish
    from w2v_aasist.full_workflow import ensure_idle
    from w2v_aasist.launch import run_lock
    from w2v_v36.workflow import export_report
    args = parser().parse_args()
    ensure_idle()
    with run_lock(ROOT / 'exp' / '.aasist-launch.lock'), run_lock(ROOT / 'exp' / '.v39-workflow.lock'):
        ensure_idle()
        if args.resume:
            run = Path(args.resume).expanduser().resolve()
            cfg = read_json(run / 'config.json')
            if cfg.get('version') != '3.9':
                raise ValueError('Resume requires a V3.9 run')
            verify_inputs(cfg)
            if args.source_run and Path(args.source_run).expanduser().resolve() != Path(cfg['source_run']):
                raise ValueError('Resume preserves the recorded source run; start a new run to change it')
            if args.device != 'auto' and args.device != cfg['device']:
                raise ValueError('Resume preserves the recorded device; restart a new run to change it')
        else:
            cfg = configuration(args)
            run = ROOT / 'exp' / ('w2v_v39_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(2))
            run.mkdir(parents=True, exist_ok=False)
            atomic_json(run / 'config.json', cfg)
        (ROOT / 'exp' / '.latest_v39_run').write_text(str(run) + '\n', encoding='utf-8')
        print('V39_RUN=' + str(run), flush=True)
        try:
            if (run / 'completed.json').is_file():
                load_selected(run)
                print('V39_ALREADY_COMPLETE=True', flush=True)
            else:
                run_experiment(cfg, run)
            (run / 'failure.log').unlink(missing_ok=True)
            publish('V3.9 complete', status='complete', force=True)
        except BaseException:
            (run / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
            publish('V3.9 failed; completed fitting stages retained', status='failed', force=True)
            raise
        finally:
            try:
                export_report(run, args.download_dir, args.upload_temp)
            except Exception as exc:
                print('REPORT_EXPORT_FAILED=' + str(exc) + '; local diagnostics retained', flush=True)


if __name__ == '__main__':
    main()
