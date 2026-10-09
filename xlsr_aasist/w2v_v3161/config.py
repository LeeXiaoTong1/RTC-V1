"""New immutable run; do not invalidate completed V3.16 code fingerprints."""
from pathlib import Path
import re
import torch
from w2v_v39.common import ROOT, digest, read_json, verify_files
from w2v_v316_tfcl.config import verify_inputs as verify_source
from w2v_v316_tfcl.state import load_selected as source_selected
from .arguments import validate


def configuration(args):
    validate(args)
    source = args.source_run
    if not source:
        source = (ROOT/'exp'/'.latest_v316_tfcl_run').read_text(encoding='utf-8').strip()
    source = Path(source).expanduser().resolve()
    checkpoint, meta = source_selected(source, 'last')
    original = checkpoint['config']
    verify_source(original)
    if meta['baseline_fallback'] or not meta['committed_updates']:
        raise ValueError('A genuinely trained V3.16 LAST is required')
    match = re.fullmatch(r'epoch_(\d+)_step_(\d+)', meta['selected'])
    if match is None: raise ValueError('Cannot identify the last completed V3.16 sampling epoch')
    del checkpoint
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg = dict(original)
    cfg.update(version='3.16.1', variant='ssl_bidirectional_tfcl_continuation_v1',
        source_run=str(source), continuation_source_config=original,
        source_checkpoint=meta['checkpoint_path'], source_checkpoint_sha256=meta['checkpoint_sha256'],
        source_last_tag=meta['selected'], source_committed_updates=meta['committed_updates'],
        sampling_epoch_offset=int(match.group(1)),
        continuation_files={str(source/name):digest(source/name) for name in
                            ('config.json', 'completed.json', 'execution_plan.json')},
        device=device, epochs=args.epochs, workers=args.workers, feature_workers=args.workers,
        microbatch=args.microbatch, frame_budget=args.frame_budget, source_batch=16,
        trainable_layers=original['trainable_layers'], head_warmup_updates=0,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, adapter_lr=args.head_lr,
        output_lr=args.head_lr, tfcl_lr=args.tfcl_lr,
        tfcl_time_weight=args.time_weight, tfcl_structure_weight=args.structure_weight,
        tfcl_feature_site='last_ssl_hidden_state', tfcl_mode='ssl_bidirectional', tfcl_heads=8, tfcl_bins=201,
        objective_ramp_epochs=.25, lr_warmup_fraction=.025, checks_per_epoch=1,
        fixed_budget=True, autotune=not args.no_autotune, checkpointing=True,
        amp='bf16' if device.startswith('cuda') else 'none', eval_amp='none',
        augmentation_policy='preserve_v316_train_distribution_for_continuation',
        best_policy='highest_complete_fixed_dev_weighted_among_trained_v316_candidates',
        effective_objective='full SSL bidirectional soft attention + per-source global channel CKA + three-view CE',
        code_fingerprints=dict(original['code_fingerprints']))
    cfg['code_fingerprints'].update({str(p.resolve()):digest(p)
        for p in (ROOT/'w2v_v3161').glob('*.py') if not p.name.startswith('test_')})
    return cfg


def verify_inputs(cfg):
    if cfg.get('variant') != 'ssl_bidirectional_tfcl_continuation_v1':
        raise ValueError('Not a V3.16 continuation')
    verify_source(cfg['continuation_source_config'])
    verify_files(cfg['continuation_files'])
    verify_files({cfg['source_checkpoint']:cfg['source_checkpoint_sha256']})
    verify_files(cfg['code_fingerprints'])
