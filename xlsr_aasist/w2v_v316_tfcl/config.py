"""V3.16 TFCL design, explicit initialization and immutable source provenance."""
import argparse
import json
from pathlib import Path
import torch
from w2v_v3151.config import source_configuration,verify_parent
from w2v_v315.config import code_fingerprints as historical_code
from w2v_v315.augment import bind_augmentation,runtime
from w2v_v39.common import ROOT,digest,read_json,verify_files
from w2v_v39.config import runtime_versions
from .selection import DEFAULTS


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run','--source-run',dest='parent_run')
    p.add_argument('--parent-checkpoint',choices=('best_guarded','best_weighted','last'))
    p.add_argument('--init',dest='init_mode',choices=('pretrained','parent'))
    p.add_argument('--tfcl',choices=('weighted','uniform','original','ce'))
    p.add_argument('--ssl-path',help='Local original official w2v-BERT weights for pretrained initialization')
    p.add_argument('--resume');p.add_argument('--device',default='auto')
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--microbatch',type=int,default=18);p.add_argument('--frame-budget',type=int,default=10800)
    p.add_argument('--no-autotune',action='store_true');p.add_argument('--seed',type=int,default=31601)
    p.add_argument('--fixed-budget',action='store_true',help='Run all requested epochs for equal-budget controls; nonfinite safety still applies')
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp');p.add_argument('--upload-temp',action='store_true')
    return p


def code_fingerprints():
    paths=historical_code()
    for module in ('w2v_v3151','w2v_v316_tfcl'):
        paths.update({str(p.resolve()):digest(p) for p in (ROOT/module).glob('*.py') if not p.name.startswith('test_')})
    return paths


def pretrained_assets(path):
    path=Path(path).expanduser().resolve()
    files=[path/'config.json',path/'preprocessor_config.json']
    single=next((path/n for n in ('model.safetensors','pytorch_model.bin') if (path/n).is_file()),None)
    if single:files.append(single)
    else:
        index=next((path/n for n in ('model.safetensors.index.json','pytorch_model.bin.index.json') if (path/n).is_file()),None)
        if index is None:raise FileNotFoundError('Original SSL weights missing at '+str(path)+'; specify --ssl-path. No automatic model download or parent substitution.')
        files.append(index)
        for name in sorted(set(read_json(index)['weight_map'].values())):
            p=(path/name).resolve()
            if p.parent!=path:raise ValueError('SSL shard escapes model directory')
            files.append(p)
    pre=read_json(path/'preprocessor_config.json')
    model=read_json(path/'config.json')
    if model.get('model_type')!='wav2vec2-bert' or model.get('feature_projection_input_dim',160)!=160:
        raise ValueError('Use original w2v-BERT 2.0 encoder weights compatible with 160-dim filterbank input')
    if pre.get('sampling_rate',16000)!=16000 or pre.get('stride',2)!=2:
        raise ValueError('Expected 16kHz, 10ms fbank hop and stride=2; native mapping otherwise differs')
    return {str(p):digest(p) for p in files}


def configuration(args):
    if not 1<=args.epochs<=4 or args.workers<0 or args.microbatch<3 or args.frame_budget<1:raise ValueError('Invalid epoch/worker/batch budget')
    parent=args.parent_run
    if not parent:
        pointer=ROOT/'exp'/'.latest_v315_run'
        if not pointer.is_file():raise ValueError('Specify completed V3.15 --parent-run as historical reference')
        parent=pointer.read_text(encoding='utf-8').strip()
    cfg=source_configuration(parent,args.parent_checkpoint or 'best_guarded')
    mode=args.init_mode or 'pretrained';device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device!='cpu' and not device.startswith('cuda'):raise ValueError('Use cpu or cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    fresh=mode=='pretrained'
    cfg.update(version='3.16',variant='offline_reference_tfcl_v1',init_mode=mode,tfcl_mode=args.tfcl or 'weighted',
        seed=args.seed,device=device,epochs=args.epochs,fixed_budget=args.fixed_budget,workers=args.workers,feature_workers=args.workers,
        feature_batch=16,eval_batch=16,microbatch=args.microbatch,frame_budget=args.frame_budget,
        source_batch=16,trainable_layers=8,checkpointing=True,
        encoder_lr=2e-5 if fresh else 2e-7,head_lr=1e-4 if fresh else 2e-6,
        adapter_lr=1e-4 if fresh else 5e-6,output_lr=1e-4 if fresh else 5e-6,tfcl_lr=1e-4,
        layer_decay=.8,lr_warmup_fraction=.05,weight_decay=.01,max_grad_norm=1.,
        head_warmup_updates=200 if fresh else 0,minimum_epochs=args.epochs if fresh else min(2,args.epochs),
        patience=4,checks_per_epoch=2,progress_min_delta=.0002,catastrophic_weighted_drop=.02,
        amp='bf16' if device.startswith('cuda') else 'none',eval_amp='none',
        tfcl_time_weight=.15,tfcl_structure_weight=.045,tfcl_heads=8,tfcl_bins=201,
        objective_ramp_epochs=.25,structure_delay_epochs=.125,
        reference_probability=.8,importance_uniform_mass=.25,importance_cap=4.,
        alignment_max_bins=192,alignment_drop_cost=.18,alignment_min_cosine=.6,
        alignment_unique_margin=.02,alignment_local_radius=2,alignment_min_frames=4,alignment_min_coverage=.1,
        time_tolerance=.02,structure_tolerance=.02,structure_window_frames=50,structure_min_frames=8,
        matched_fake_recall=.99,threshold=.5,raw_audio_cache_mib=128,noise_cache_mib=64,prefetch_factor=2,
        rolling_cache_bytes=0,disk_margin_bytes=128*1024**2,free_reserve_bytes=10*1024**3,
        maximum_new_peak_bytes=20*1024**3,gpu_reserve_bytes=6*1024**3,autotune=not args.no_autotune,
        panel_seed=3160901,panel_sources_per_group=128,
        code_fingerprints=code_fingerprints(),runtime_versions=runtime_versions(),
        effective_objective='independent full-wave CE + reliable one-way local temporal and channel consistency',
        effective_views='Offline 10% / official Online 50% / noise-before-RTC 40%; missing Online source 20/80',
        source_exclusion_policy='none for missing Online; missing auxiliary edge is disabled without budget redistribution',
        external_teacher_at_inference=False)
    cfg.update(DEFAULTS)
    cfg['pretrained_path']=str(Path(args.ssl_path or cfg['ssl_path']).expanduser().resolve())
    cfg['pretrained_fingerprints']=pretrained_assets(cfg['pretrained_path']) if fresh else {}
    cfg['initialization_provenance']=(dict(mode='pretrained',ssl_path=cfg['pretrained_path'],
        ssl_files=cfg['pretrained_fingerprints'],fresh_head_seed=cfg['seed'],parent_weights_used_for_training=False,
        optimizer_restored=False) if fresh else dict(mode='parent',checkpoint=cfg['parent_checkpoint'],
        checkpoint_sha256=cfg['parent_checkpoint_sha256'],selector=cfg['parent_selector'],
        selected_tag=cfg['parent_selected_tag'],parent_weights_used_for_training=True,optimizer_restored=False))
    if fresh:
        historical=read_json(Path(cfg['ssl_path'])/'preprocessor_config.json')
        original=read_json(Path(cfg['pretrained_path'])/'preprocessor_config.json')
        for name in ('sampling_rate','stride','feature_size','num_mel_bins','do_normalize','padding_value'):
            if historical.get(name)!=original.get(name):
                raise ValueError('Original SSL processor differs from fixed historical Dev: '+name)
    # Keep historical Dev's feature extractor fixed. The new encoder must accept it.
    cfg.update(bind_augmentation(cfg))
    return cfg


def verify_inputs(cfg):
    if cfg.get('version')!='3.16' or cfg.get('variant')!='offline_reference_tfcl_v1':raise ValueError('Not a V3.16 TFCL run; Omni uses another entry point')
    for key in ('code_fingerprints','source_fingerprints','data_fingerprints','augmentation_files','pretrained_fingerprints'):verify_files(cfg[key])
    verify_parent(cfg)
    if cfg['runtime_versions']!=runtime_versions() or cfg['augmentation_runtime']!=runtime(cfg['augmentation_runtime'].get('ffmpeg_path')):
        raise ValueError('Recorded runtime differs; cannot silently resume')
