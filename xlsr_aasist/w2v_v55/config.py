"""Exact V5.1 final continuation, with configurable paper loss coefficients."""
import argparse
import ast
import copy
from pathlib import Path
from .common import ROOT, RECIPE, digest, schema_for
from .parent import FINAL_KIND, DEFAULT_PARENT, continuation_rates
from .objective import configuration
from w2v_v318.config import runtime_versions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', help='Completed V5.1 run; default exp/' + DEFAULT_PARENT)
    p.add_argument('--resume', help='Resume a V5.5 run at its last committed update')
    p.add_argument('--phase', choices=('all', 'final'), default='final')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int)
    p.add_argument('--eval-workers', type=int)
    p.add_argument('--prefetch-factor', type=int, default=1)
    p.add_argument('--microbatch', type=int, default=16,
                   help='Complete metric/PCL group can require 13 views; no cropping fallback')
    p.add_argument('--frame-budget', type=int, default=9000)
    p.add_argument('--eval-batch', type=int, default=12)
    p.add_argument('--window-batch', type=int, default=16)
    p.add_argument('--checkpoint-interval', type=int, default=100)
    p.add_argument('--seed', type=int, default=50001)
    p.add_argument('--hp-mode', choices=('hard_positive', 'control'), default='hard_positive',
                   help='Control uses identical extra views and CE, but zero metric weights')
    p.add_argument('--guidance-weight', type=float, default=.01,
                   help='Project starting coefficient, not reported by the paper')
    p.add_argument('--triplet-weight', type=float, default=.01)
    p.add_argument('--guidance-margin', type=float, default=.2)
    p.add_argument('--triplet-margin', type=float, default=.3)
    p.add_argument('--hp-ramp-updates', type=int, default=200)
    p.add_argument('--submit-best', action='store_true')
    p.add_argument('--upload-submission', action='store_true')
    p.add_argument('--upload-temp', action='store_true')
    p.add_argument('--submission-protocol')
    p.add_argument('--submission-audio-root')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    return p


def validate(args, argv=None):
    if args.resume and args.parent_run:
        raise ValueError('Use --resume OR --parent-run')
    if args.microbatch < 13:
        raise ValueError('V5.5 requires microbatch >=13 for the complete metric/PCL group')
    if min(args.epochs, args.frame_budget, args.eval_batch, args.window_batch,
           args.prefetch_factor, args.checkpoint_interval) < 1:
        raise ValueError('Positive training budgets required')
    if any(n is not None and n < 0 for n in (args.workers, args.eval_workers)):
        raise ValueError('Worker counts must be nonnegative')
    if not 0 <= args.seed < 2**32:
        raise ValueError('Seed must be uint32')
    configuration(dict(hp_guidance_weight=args.guidance_weight, hp_triplet_weight=args.triplet_weight,
        hp_guidance_margin=args.guidance_margin, hp_triplet_margin=args.triplet_margin,
        hp_ramp_updates=args.hp_ramp_updates, hp_mode=args.hp_mode))
    if args.upload_submission and not args.submit_best:
        raise ValueError('--upload-submission requires --submit-best')
    if args.resume and argv is not None:
        allowed = {'--resume', '--phase', '--submit-best', '--upload-submission', '--upload-temp',
                   '--submission-protocol', '--submission-audio-root', '--download-dir'}
        overrides = {s.split('=')[0] for s in argv if s.startswith('--')} - allowed
        if overrides:
            raise ValueError('Resume uses saved settings; remove ' + ', '.join(sorted(overrides)))


def code_inventory(root=ROOT):
    root = Path(root)
    from .baseline import verify_baseline
    baseline = verify_baseline(root)
    pending = list((root / 'w2v_v55').glob('*.py'))
    seen = {}
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file() or path.name.startswith('test_'):
            continue
        seen[path] = digest(path)
        package = list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split('.') for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                target = prefix + (node.module.split('.') if node.module else [])
                names = [target] + [target + a.name.split('.') for a in node.names if a.name != '*']
            for parts in names:
                for depth in range(1, len(parts)+1):
                    item = root.joinpath(*parts[:depth])
                    pending.extend((item.with_suffix('.py'), item / '__init__.py'))
    forbidden = {'w2v_v52', 'w2v_v53', 'w2v_v54'}
    if any(forbidden.intersection(p.relative_to(root).parts) for p in seen):
        raise ValueError('A newer model/recipe entered the V5.1 dependency tree')
    return {str(p): h for p, h in seen.items()} | {str(baseline): digest(baseline)}


def training_configuration(old, origin, args):
    if old.get('version') != '5' or origin.get('kind') != FINAL_KIND:
        raise ValueError('Rollback requires the completed V5.1 final.pt, not V5.2/V5.3/V5.4')
    if old.get('classification_ce_weights') != dict(online=.25, clean_rtc=.25, noisy=.5):
        raise ValueError('Unexpected V5.1 CE recipe')
    if any(k.startswith(('v52_', 'v53_', 'v54_', 'adaptive_fusion_', 'repair_', 'contrastive_')) for k in old):
        raise ValueError('Parent contains newer experimental settings')
    cfg = copy.deepcopy(old)
    for key in ('saved_manifests', 'warm_start', 'augmentation_budget_audit',
                'window_resource_audit', 'v5_recipe_lock_sha256'):
        cfg.pop(key, None)
    cfg.update(version='5.5', recipe=RECIPE, training_recipe_base='5.1',
        v5_stage='final', epochs=args.epochs, warm_epochs=0, lora_layers=16,
        checkpoint_layers=16, checkpointing=True,
        retained_lora_layers=old.get('retained_lora_layers', old['lora_layers']),
        device=args.device, microbatch=args.microbatch, frame_budget=args.frame_budget,
        eval_batch=args.eval_batch, window_batch=args.window_batch,
        checkpoint_interval=args.checkpoint_interval, sampling_seed=args.seed,
        parent_init=origin, data_files=origin['manifest_files'],
        source_epoch_offset=int(old.get('source_epoch_offset', 0)) + origin['epoch'],
        initialization=dict(mode=FINAL_KIND, run=origin['run'], selected=origin['tag'],
            old_optimizer_loaded=False, all_parent_adapters_retained=True),
        selection='None: continue V5.1 Train+Dev with inherited locked input/weight rule',
        hp_mode=args.hp_mode, hp_guidance_weight=args.guidance_weight,
        hp_triplet_weight=args.triplet_weight, hp_guidance_margin=args.guidance_margin,
        hp_triplet_margin=args.triplet_margin, hp_ramp_updates=args.hp_ramp_updates,
        hp_distance='squared_euclidean_raw_384_pre_final_dropout',
        hp_group_policy='one_rotating_anchor_per_language_side_2_positive_3_distinct_negative',
        runtime_versions=runtime_versions())
    cfg.update(continuation_rates(old, origin))
    from w2v_v3189.workers import settings
    cfg.update(settings(args, old))
    schema_for(cfg)
    configuration(cfg)
    return cfg


def verify_inputs(cfg, *, training=True):
    from w2v_v3185.config import verify_unique
    from w2v_v3184.assets import configured_spec
    from w2v_v5.manifests import bind_parent_phonemes
    from .common import read_json
    schema_for(cfg)
    configuration(cfg)
    configured_spec(cfg)
    verify_unique(cfg['code_fingerprints'], {cfg['omni_checkpoint']: cfg['omni_sha256']},
        cfg['data_files'] if training else {}, cfg['augmentation_files'] if training else {},
        {cfg['ffmpeg']: cfg['ffmpeg_sha256']} if training else {},
        cfg['scene_noise_files'] if training else {}, cfg.get('phoneme_files', {}) if training else {},
        cfg.get('identity_input_files', {}) if training else {},
        {cfg['parent_init']['checkpoint']: cfg['parent_init']['checkpoint_sha256']} if training else {})
    if cfg['runtime_versions'] != runtime_versions():
        raise ValueError('Pinned runtime changed')
    if training and cfg.get('consistency') == 'dual':
        bind_parent_phonemes(cfg, read_json(cfg['parent_init']['training_rows']))


def compact_configuration(cfg):
    keys = ('version', 'recipe', 'training_recipe_base', 'epochs', 'v5_stage', 'v5_input',
        'classification_ce_weights', 'pcl_noise_weight', 'pcl_platform_weight', 'pcl_variance_weight',
        'lora_layers', 'retained_lora_layers', 'source_draws_per_stream', 'microbatch', 'frame_budget',
        'source_epoch_offset', 'sampling_seed', 'head_lr', 'fusion_lr', 'layer_mix_lr', 'lora_lr',
        'selection', 'v5_pcl_scope', 'v5_context_seconds')
    return dict(settings={k: v for k, v in cfg.items() if k in keys or k.startswith('hp_')},
                parent={k: cfg['parent_init'][k] for k in ('run', 'checkpoint', 'checkpoint_sha256', 'tag')},
                note='V5.1 architecture and recipe; additional RTC hard-positive metric training')


if __name__ == '__main__':
    import sys
    validate(parser().parse_args(), sys.argv[1:])
