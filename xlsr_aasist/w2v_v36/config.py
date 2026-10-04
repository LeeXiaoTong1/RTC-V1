"""Resolve the submitted best, existing full caches and a bounded classifier recipe."""
import argparse
import json
from pathlib import Path

from w2v_aasist.launch import ROOT
from w2v_aasist.runtime import sha256
from w2v_v34.source import resolve_source


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', default=str(ROOT/'exp'/'w2v_v33_20261002_004018_198a'))
    p.add_argument('--submission-meta')
    p.add_argument('--dev-run', default=str(ROOT/'exp'/'w2v_v35_20261003_161443_4fb5'))
    p.add_argument('--resume', help='Resume one V3.6 run; committed feature rows are reused')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--frame-budget', type=int, default=2400)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def code_fingerprints():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'w2v_v3'/name for name in ('model.py', 'data.py', 'step.py')]
    paths += [ROOT/'w2v_aasist'/name for name in ('data.py', 'model.py', 'runtime.py')]
    return {str(p.resolve()): sha256(p) for p in sorted(paths) if not p.name.startswith('test_')}


def verify_files(fingerprints):
    for name, digest in fingerprints.items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise ValueError('A pinned input changed: '+name)


def configuration(args):
    if args.workers < 0 or args.microbatch < 1 or args.frame_budget < 1:
        raise ValueError('Nonnegative workers and positive inference batch limits required')
    source = resolve_source(args.source_run, args.submission_meta)
    dev_path = Path(args.dev_run).expanduser().resolve()/'config.json'
    dev = json.loads(dev_path.read_text(encoding='utf-8'))
    if dev.get('version') != '3.5' or not dev.get('full_dev_cache_root'):
        raise ValueError('Existing V3.5 fixed full Dev configuration required; no audio is regenerated')
    for key in ('train_protocol', 'dev_protocol', 'train_data_path', 'dev_data_path'):
        if Path(dev[key]).resolve() != Path(source['config'][key]).resolve():
            raise ValueError('Submitted best and fixed full Dev use different data: '+key)
    cfg = dict(source['config'])
    cfg.update(version='3.6', source_config=source['config'], dev_config=dev,
        source_run=str(Path(args.source_run).expanduser().resolve()),
        base_checkpoint=source['checkpoint'], base_checkpoint_sha256=source['checkpoint_sha256'],
        base_tag=source['checkpoint_tag'], source_provenance=source['provenance'],
        dev_config_path=str(dev_path), dev_config_sha256=sha256(dev_path),
        workers=args.workers, microbatch=args.microbatch, frame_budget=args.frame_budget,
        eval_batch=8, device=args.device, seed=3601, amp='none', eval_amp='none',
        trainable_layers=0, checkpointing=False,
        lambda_grid=[0.1, 1.0], max_iterations=200, fit_chunk_rows=4096,
        holdout_fraction=.2, min_gain=.002, max_clean_drop=.001,
        max_noisy_fake_recall_drop=.005,
        metric_definition='full Dev online Clean; mean Seen/Heldout Noisy; 0.3 Clean + 0.7 Noisy',
        threshold=.5, code_fingerprints=code_fingerprints())
    return cfg
