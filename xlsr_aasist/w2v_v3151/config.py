"""Pin an explicitly selected V3.15 parent; preserve its immutable base chain."""
import argparse
from pathlib import Path

import torch

from w2v_v315.config import code_fingerprints as previous_code
from w2v_v315.state import load_selected as load_parent
from w2v_v315.augment import bind_augmentation, runtime
from w2v_v39.common import ROOT, read_json, digest, verify_files
from w2v_v39.config import runtime_versions
from w2v_v313.state import identity


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', help='Completed V3.15 run; defaults to exp/.latest_v315_run')
    p.add_argument('--parent-checkpoint', choices=('best_guarded','best_weighted','last'),
                   help='New runs default to best_guarded; resume preserves its recorded parent')
    p.add_argument('--resume')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--microbatch', type=int, default=18, help='Physical waveform limit; logical batch stays 16 sources/48 full views')
    p.add_argument('--frame-budget', type=int, default=10800)
    p.add_argument('--no-autotune', action='store_true')
    p.add_argument('--ce-only', action='store_true', help='Explicit same-data control, never launched automatically')
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    return p


def code_fingerprints():
    result = previous_code()
    result.update({str(p.resolve()):digest(p) for p in (ROOT/'w2v_v3151').glob('*.py')
                   if not p.name.startswith('test_')})
    return result


def source_configuration(run, kind='best_guarded'):
    run = Path(run).expanduser().resolve()
    parent, meta = load_parent(run,kind)
    old = parent['config']
    verify_files(old['code_fingerprints'])
    verify_files(old['source_fingerprints'])
    cfg = dict(old)
    cfg['source_fingerprints'] = dict(old['source_fingerprints'])
    cfg['source_fingerprints'].update({str(run/name):digest(run/name)
        for name in ('config.json','completed.json','execution_plan.json')})
    cfg['source_fingerprints'][meta['checkpoint_path']] = meta['checkpoint_sha256']
    cfg.update(parent_run=str(run),parent_version='3.15',parent_selector=kind,
        parent_selected_tag=meta['selected'],parent_config_identity=identity(old),
        parent_checkpoint=meta['checkpoint_path'],parent_checkpoint_sha256=meta['checkpoint_sha256'],
        parent_baseline_fallback=meta['baseline_fallback'])
    del parent
    return cfg


def configuration(args):
    if not 1<=args.epochs<=4 or args.workers<0 or args.microbatch<3 or args.frame_budget<1:
        raise ValueError('epochs=1..4, workers>=0, physical microbatch>=3, positive frame budget required')
    parent=args.parent_run
    if parent is None:
        pointer=ROOT/'exp'/'.latest_v315_run'
        if not pointer.is_file():
            raise ValueError('Specify --parent-run: the completed V3.15 best_guarded run')
        parent=pointer.read_text(encoding='utf-8').strip()
    cfg=source_configuration(parent,args.parent_checkpoint or 'best_guarded')
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device!='cpu' and not device.startswith('cuda'):
        raise ValueError('Use cpu or cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg.update(version='3.15.1',seed=315101,device=device,epochs=args.epochs,
        workers=args.workers,feature_workers=args.workers,feature_batch=16,eval_batch=16,
        microbatch=args.microbatch,frame_budget=args.frame_budget,source_batch=16,
        trainable_layers=8,checkpointing=True,encoder_lr=2e-7,head_lr=2e-6,
        adapter_lr=5e-6,output_lr=5e-6,tfcl_lr=1e-4,layer_decay=.8,
        lr_warmup_fraction=.05,weight_decay=.01,max_grad_norm=1.,
        amp='bf16' if device.startswith('cuda') else 'none',eval_amp='none',
        tfcl_time_weight=0. if args.ce_only else .15,
        tfcl_structure_weight=0. if args.ce_only else .045,
        tfcl_bridge_mass=.5,tfcl_matched_mass=.5,tfcl_heads=8,tfcl_bins=201,
        objective_ramp_epochs=.25,online_ce_mass=.5,reference_ce_mass=.1,
        initial_noisy_ce_mass=.3,main_noisy_ce_mass=.4,
        raw_audio_cache_mib=128,noise_cache_mib=64,prefetch_factor=2,
        rolling_cache_bytes=0,minimum_epochs=2,patience=4,checks_per_epoch=2,
        progress_min_delta=.0002,min_gain=.0002,max_clean_drop=.003,
        max_noisy_drop=.0005,max_fake_drop=.005,max_real_drop=.015,
        max_auc_drop=.002,max_matched_real_drop=.02,matched_fake_recall=.99,
        catastrophic_weighted_drop=.02,threshold=.5,
        disk_margin_bytes=128*1024**2,free_reserve_bytes=10*1024**3,
        maximum_new_peak_bytes=20*1024**3,gpu_reserve_bytes=6*1024**3,
        autotune=not args.no_autotune,panel_seed=3151901,
        panel_sources_per_group=128,
        code_fingerprints=code_fingerprints(),runtime_versions=runtime_versions(),
        effective_objective='balanced full-wave CE + official-online/reference and reference/noisy TFCL',
        effective_views='verified official Online / simulated RTC reference / same-setting noisy RTC',
        selection_policy='independent fixed-dev best_weighted and broader-robustness best_guarded; parent fallback retained',
        external_teacher_at_inference=False)
    from .selection import DEFAULTS
    cfg.update(DEFAULTS)
    cfg.update(bind_augmentation(cfg))
    return cfg


def verify_parent(cfg):
    verify_files({cfg['parent_checkpoint']:cfg['parent_checkpoint_sha256']})
    old=read_json(Path(cfg['parent_run'])/'config.json')
    done=read_json(Path(cfg['parent_run'])/'completed.json')
    if identity(old)!=cfg['parent_config_identity'] or done.get('checkpoint_sha256')!=cfg['parent_checkpoint_sha256']:
        raise ValueError('Pinned V3.15 parent changed; never silently change the starting detector')
    return old


def verify_inputs(cfg):
    if cfg.get('version')!='3.15.1' or cfg.get('parent_version')!='3.15':
        raise ValueError('Expected V3.15.1 with a pinned V3.15 parent')
    for key in ('code_fingerprints','source_fingerprints','data_fingerprints','augmentation_files'):
        verify_files(cfg[key])
    verify_files({cfg['base_checkpoint']:cfg['base_checkpoint_sha256'],
                  cfg['starting_checkpoint']:cfg['starting_checkpoint_sha256']})
    verify_parent(cfg)
    if cfg['runtime_versions']!=runtime_versions():
        raise ValueError('Runtime changed; resume requires the recorded environment')
    if cfg['augmentation_runtime']!=runtime(cfg['augmentation_runtime'].get('ffmpeg_path')):
        raise ValueError('FFmpeg/WebRTC runtime changed')
