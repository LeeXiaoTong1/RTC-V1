"""Reuse verified data identities, never historical detector/optimizer weights."""
from pathlib import Path
import torch
from w2v_v39.common import ROOT, digest, read_json, verify_files
from w2v_v39.config import runtime_versions
from w2v_v316_tfcl.config import pretrained_assets, code_fingerprints as dependencies
from w2v_v315.augment import runtime
from .arguments import validate


def code_fingerprints():
    result = dependencies()
    for folder in ('w2v_v3161','w2v_v317'):
        result.update({str(p.resolve()):digest(p) for p in (ROOT/folder).glob('*.py')
                       if not p.name.startswith('test_')})
    return result


def configuration(args):
    validate(args)
    source = args.data_run
    if not source:
        pointer = ROOT/'exp'/'.latest_v316_tfcl_run'
        if not pointer.is_file(): raise ValueError('Specify --data-run pointing to the existing V3.16 run')
        source = pointer.read_text(encoding='utf-8').strip()
    source = Path(source).expanduser().resolve()
    original = read_json(source/'config.json')
    if original.get('version') not in ('3.16','3.16.1'):
        raise ValueError('--data-run must be V3.16 or V3.16.1; no detector weights are imported')
    cfg = dict(original.get('continuation_source_config', original))
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device != 'cpu' and (not device.startswith('cuda') or not torch.cuda.is_available()):
        raise ValueError('Requested CUDA device is unavailable')
    ssl = Path(args.ssl_path or cfg.get('pretrained_path') or cfg['ssl_path']).expanduser().resolve()
    # Existing full-wave metadata is bound to the same feature-extractor geometry.
    if digest(ssl/'preprocessor_config.json') != digest(Path(cfg['ssl_path'])/'preprocessor_config.json'):
        raise ValueError('The new encoder must use the existing feature-extractor configuration')
    for key in ('parent_checkpoint','parent_checkpoint_sha256','parent_run','source_checkpoint',
                'continuation_source_config','source_code_migrations','continuation_files',
                'encoder_lr','adapter_lr','output_lr','head_warmup_updates','selection_policy'):
        cfg.pop(key, None)
    cfg.update(version='3.17',variant='lora_post_multiconv_tfcl_v1',init_mode='pretrained',
        data_run=str(source),data_run_config_sha256=digest(source/'config.json'),
        pretrained_path=str(ssl),ssl_path=str(ssl),pretrained_fingerprints=pretrained_assets(ssl),
        seed=args.seed,device=device,epochs=args.epochs,workers=args.workers,feature_workers=args.workers,
        source_batch=16,microbatch=args.microbatch,frame_budget=args.frame_budget,
        feature_batch=16,eval_batch=16,checkpointing=True,autotune=not args.no_autotune,
        lora_layers=args.lora_layers,lora_rank=args.lora_rank,lora_alpha=2.*args.lora_rank,lora_dropout=.05,
        lora_lr=args.lora_lr,head_lr=args.head_lr,tfcl_lr=args.tfcl_lr,
        feature_dim=128,head_expansion=1024,head_blocks=4,block_dropout=.1,classifier_dropout=.2,
        fusion_chunk_layers=5,tfcl_heads=8,tfcl_bins=201,
        tfcl_time_weight=args.time_weight,tfcl_structure_weight=args.structure_weight,
        tfcl_feature_site='post_multiconv_forensic',tfcl_mode='post_multiconv_bidirectional',
        trainable_layers=0,output_projection_frozen=False,objective_ramp_epochs=.5,
        effective_objective='balanced three-view CE + bidirectional post-MultiConv time/CKA; Q/V LoRA only in encoder',
        effective_views={'official_pair':{'offline':.1,'online':.5,'noisy':.4},
                         'missing_online':{'offline':.2,'noisy':.8}},
        lr_warmup_fraction=.05,weight_decay=.01,max_grad_norm=1.,
        source_sampling_power=.5,train_probe_per_group=64,
        fixed_budget=True,threshold=.5,matched_fake_recall=.99,
        amp='bf16' if device.startswith('cuda') else 'none',eval_amp='none',
        rolling_cache_bytes=0,disk_margin_bytes=128*1024**2,free_reserve_bytes=10*1024**3,
        maximum_new_peak_bytes=2*1024**3,gpu_reserve_bytes=6*1024**3,
        raw_audio_cache_mib=128,noise_cache_mib=64,prefetch_factor=2,
        code_fingerprints=code_fingerprints(),runtime_versions=runtime_versions(),
        initialization_provenance=dict(mode='public_pretrained_plus_fresh_lora_and_head',
            detector_checkpoint_loaded=False,optimizer_restored=False,data_only_source=str(source)),
        best_policy='maximum complete fixed Dev weighted within this V3.17 run; no fallback')
    verify_inputs(cfg)
    return cfg


def verify_inputs(cfg):
    if cfg.get('variant') != 'lora_post_multiconv_tfcl_v1': raise ValueError('Expected V3.17 configuration')
    verify_files({str(Path(cfg['data_run'])/'config.json'):cfg['data_run_config_sha256']})
    for key in ('pretrained_fingerprints','data_fingerprints','augmentation_files','code_fingerprints'):
        verify_files(cfg[key])
    if cfg['runtime_versions'] != runtime_versions(): raise ValueError('Recorded Python/runtime versions changed')
    if cfg['augmentation_runtime'] != runtime(cfg['augmentation_runtime'].get('ffmpeg_path')):
        raise ValueError('Recorded noise/RTC runtime changed')
