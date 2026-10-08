"""Pin new SSL weights independently of the historical metadata producer."""
import argparse
from importlib.metadata import version
from pathlib import Path

import torch
from w2v_v39.common import ROOT,read_json,digest,verify_files
from w2v_v315.config import code_fingerprints as inherited_code


def runtime_versions():
    return {name:version(name) for name in ('torch','numpy','fairseq2','fairseq2n','omnilingual-asr','soundfile','webrtc-audio-processing')}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',help='V3.15 run with at least one committed Dev validation')
    p.add_argument('--resume')
    p.add_argument('--omni-assets',default='pretrained/omniASR-W2V-7B/assets.json')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--frame-budget',type=int,default=2400)
    p.add_argument('--lora-layers',type=int,default=16)
    p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--no-autotune',action='store_true')
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp',action='store_true')
    return p


def configuration(args):
    source=Path(args.source_run) if args.source_run else Path((ROOT/'exp'/'.latest_v315_run').read_text().strip())
    source=source.expanduser().resolve()
    old=read_json(source/'config.json')
    if old.get('version')!='3.15': raise ValueError('Source metadata/reference must be V3.15')
    history=read_json(source/'training_history.json')
    valid=[v for v in history if v.get('committed') and v.get('metrics',{}).get('complete')]
    if not valid: raise ValueError('V3.15 has no committed Dev result; wait for its next validation save')
    best=max(valid,key=lambda e:e['metrics']['weighted_f1'])
    baseline=read_json(source/'baseline_metrics.json')
    tag='baseline' if baseline['weighted_f1']>=best['metrics']['weighted_f1'] else best['tag']
    score=source/('dev_scores_'+tag+'.npz')
    if not score.is_file() or not (source/'dev_rows.json').is_file():
        raise FileNotFoundError('Committed V3.15 score/row provenance missing')
    assets=read_json(args.omni_assets)
    if assets.get('schema')!='rtc_omni_w2v7b_assets_v1': raise ValueError('Run prepare_omni first')
    if (not 1<=args.epochs<=4 or args.workers<0 or args.microbatch<2 or args.frame_budget<1
            or not 1<=args.lora_layers<=128 or not 1<=args.lora_rank<=128):
        raise ValueError('Invalid epoch/worker/batch/LoRA budget')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('7B training requires a CUDA GPU; CPU is only for unit tests')
    cfg=dict(old)
    # These old identity fields describe row metadata, NOT new detector weights.
    cfg.update(version='3.16',source_run=str(source),source_version='3.15',reference_tag=tag,reference_scores=str(score),
        source_fingerprints={str(p):digest(p) for p in (source/'config.json',source/'dev_rows.json',score)},
        reference_rows=str(source/'dev_rows.json'),reference_kind='V315 scores only; no old weights imported',
        starting_kind='omni_ssl_with_new_detection_head',omni_checkpoint=assets['checkpoint'],
        starting_checkpoint=assets['checkpoint'],starting_checkpoint_sha256=assets['sha256'],starting_tag='official_omni_ssl',
        omni_sha256=assets['sha256'],omni_provenance=assets,
        device=args.device,epochs=args.epochs,seed=31601,workers=args.workers,source_batch=16,
        microbatch=args.microbatch,frame_budget=args.frame_budget,eval_batch=8,
        lora_layers=args.lora_layers,lora_rank=args.lora_rank,lora_alpha=2.*args.lora_rank,
        feature_layers=[15,31,63,95,111,119,123,127],lora_lr=5e-5,head_lr=3e-4,tfcl_lr=1e-4,
        weight_decay=.01,max_grad_norm=1.,head_warmup_updates=200,objective_ramp_epochs=.25,
        tfcl_time_weight=.15,tfcl_structure_weight=.045,tfcl_heads=8,tfcl_bins=201,
        amp='bf16',eval_amp='bf16',checkpointing=True,autotune=not args.no_autotune,
        rolling_cache_bytes=0,noise_cache_mib=64,free_reserve_bytes=10*1024**3,
        disk_margin_bytes=128*1024**2,maximum_new_peak_bytes=4*1024**3,gpu_reserve_bytes=4*1024**3,
        checks_per_epoch=2,patience=4,minimum_epochs=2,progress_min_delta=.0002,
        selection_policy='best_weighted is best trained Omni; best_guarded requires V315 reference guards; no automatic old-model fallback',
        runtime_versions=runtime_versions())
    cfg['code_fingerprints']=inherited_code()
    cfg['code_fingerprints'].update({str(p.resolve()):digest(p) for p in (ROOT/'w2v_v316').glob('*.py') if not p.name.startswith('test_')})
    verify_inputs(cfg)
    return cfg


def verify_inputs(cfg):
    if cfg.get('version')!='3.16' or cfg.get('starting_kind')!='omni_ssl_with_new_detection_head':
        raise ValueError('Not a V3.16 Omni run')
    for key in ('code_fingerprints','source_fingerprints','data_fingerprints','augmentation_files'):
        verify_files(cfg[key])
    verify_files({cfg['omni_checkpoint']:cfg['omni_sha256']})
    if runtime_versions()!=cfg['runtime_versions']: raise ValueError('Recorded runtime changed')
    from w2v_v315.augment import runtime
    if runtime(cfg['augmentation_runtime'].get('ffmpeg_path'))!=cfg['augmentation_runtime']:
        raise ValueError('Augmentation runtime changed')
