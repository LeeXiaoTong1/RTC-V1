"""Bind the submitted V3.12 LAST, not its baseline-selected best.pt."""
import argparse
from pathlib import Path

import torch

from w2v_v39.common import ROOT, digest, read_json, verify_files
from w2v_v39.config import runtime_versions
from w2v_v312.config import code_fingerprints as prior_code
from w2v_v312.state import load_resume as load_v312_resume


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', help='Completed V3.12 run; its committed last.pt is required')
    p.add_argument('--resume')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--frame-budget', type=int, default=2400)
    p.add_argument('--pair-micro-sources', type=int, choices=(4,8), default=8)
    p.add_argument('--adv-weight', type=float, default=.005)
    p.add_argument('--pair-weight', type=float, default=.1)
    p.add_argument('--ranking-weight', type=float, default=.1)
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def source_configuration(source):
    source = Path(source).expanduser().resolve()
    old = read_json(source/'config.json')
    done = read_json(source/'completed.json')
    if old.get('version') != '3.12' or done.get('version') != '3.12' or done.get('status') != 'complete':
        raise ValueError('A completed V3.12 LAST run is required')
    if not (source/'last.pt').is_file():
        raise FileNotFoundError('V3.12 last.pt missing; never substitute its baseline best.pt')
    state = load_v312_resume(source/'last.pt', old)
    history = state.get('history', [])
    if (not state.get('model') or not history or not history[-1].get('committed')
            or state['cursor'] != history[-1]['cursor'] or state['cursor'] != done['committed_updates']
            or old['base_checkpoint_sha256'] != done['base_checkpoint_sha256']):
        raise ValueError('V3.12 LAST is not the matching committed trained checkpoint')
    verify_files(old['code_fingerprints'])
    verify_files({old['base_checkpoint']: old['base_checkpoint_sha256']})
    cfg = dict(old)
    cfg.update(source_run=str(source), source_version='3.12',
        starting_checkpoint=str(source/'last.pt'), starting_checkpoint_sha256=digest(source/'last.pt'),
        starting_tag=history[-1]['tag'], starting_cursor=state['cursor'], starting_kind='trained_v312_last',
        starting_config_identity=state['identity'],
        user_reported_progress_weighted=(93.558 if source.name=='w2v_v312_20261006_004830_7d2f' else None))
    cfg['source_fingerprints'] = dict(old['source_fingerprints'])
    cfg['source_fingerprints'].update({str(source/name):digest(source/name)
                                     for name in ('last.pt', 'config.json', 'completed.json')})
    del state
    return cfg


def code_fingerprints():
    result = prior_code()
    result.update({str(p.resolve()):digest(p) for p in (ROOT/'w2v_v313').glob('*.py')
                   if not p.name.startswith('test_')})
    return result


def configuration(args):
    if not args.source_run:
        raise ValueError('Specify --source-run explicitly: the submitted V3.12 LAST run')
    if (not 1 <= args.epochs <= 4 or args.workers < 0 or args.microbatch < 1 or args.frame_budget < 1
            or any(not 0 <= x <= 1 for x in (args.adv_weight, args.pair_weight, args.ranking_weight))):
        raise ValueError('Epoch budget must be 1..4; workers/batches/loss weights must be valid')
    cfg = source_configuration(args.source_run)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg.update(version='3.13', device=device, seed=31301, trainable_layers=8,
        epochs=args.epochs, minimum_epochs=args.epochs, workers=args.workers, feature_workers=args.workers,
        source_batch=16, pair_micro_sources=args.pair_micro_sources, microbatch=args.microbatch, frame_budget=args.frame_budget,
        eval_batch=16, feature_batch=16, checkpointing=True,
        encoder_lr=2e-7, head_lr=2e-6, adapter_lr=1e-5, adversary_lr=1e-4,
        adv_weight=args.adv_weight, pair_weight=args.pair_weight, ranking_weight=args.ranking_weight,
        auxiliary_warmup_fraction=.25, hard_weight_strength=1., hard_weight_max=2.,
        pair_margin_weight=1., pair_feature_weight=.1, pair_min_confidence=.75,
        pair_margin_cap=4., pair_margin_slack=.25,
        ranking_logit_margin=.2, ranking_temperature=.2, ranking_score_temperature=2.,
        ranking_feature_margin=.15, ranking_feature_weight=1., ranking_logit_weight=.25,
        retention_weight=.1, stability_weight=.1, pair_probe_sources_per_group=64,
        probe_sources_per_language=64,
        reference_batch=8192, amp='bf16' if device.startswith('cuda') else 'none',
        min_gain=.0002, max_clean_drop=.003, max_fake_drop=.005, max_real_drop=.015,
        max_auc_drop=.002, max_matched_real_drop=.02, min_en_real_gain=0.,
        patience=8, checks_per_epoch=2, catastrophic_weighted_drop=.02,
        threshold=.5, code_fingerprints=code_fingerprints(), runtime_versions=runtime_versions(),
        algorithm='two-view official-source decision consistency; condition/language matched hard-negative geometry; weak optional language auxiliary',
        selection_policy='fixed local weighted gain with explicit recall/clean safety limits; language probes diagnostic only')
    return cfg


def verify_inputs(cfg):
    if cfg.get('version') != '3.13' or cfg.get('starting_kind') != 'trained_v312_last':
        raise ValueError('Expected V3.13 bound to the trained V3.12 LAST')
    for key in ('code_fingerprints', 'source_fingerprints', 'data_fingerprints'):
        verify_files(cfg[key])
    verify_files({cfg['base_checkpoint']:cfg['base_checkpoint_sha256'],
                  cfg['starting_checkpoint']:cfg['starting_checkpoint_sha256']})
    if cfg['runtime_versions'] != runtime_versions():
        raise ValueError('Runtime changed; resume requires the recorded environment')
