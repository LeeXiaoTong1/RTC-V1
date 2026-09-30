"""Explicit V3 recipe; paths come from the existing official Train configuration."""
import argparse
import math
from pathlib import Path
from w2v_aasist.launch import BASELINE, BASELINE_SHA256, RAW, ROOT
from w2v_aasist.runtime import sha256


def parser():
    p = argparse.ArgumentParser(description='V3 full-wave MultiConv: prepare, validate, retire old caches, train')
    p.add_argument('--source-config')
    p.add_argument('--baseline',default=BASELINE)
    p.add_argument('--full-noisy-cache',default=str(ROOT/'data'/'rtc_noisy_full2_v1'/'train'))
    p.add_argument('--noise-manifest')
    p.add_argument('--ffmpeg')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cache-workers',type=int,default=4)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--amp',choices=('bf16','none'),default='bf16')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--ordinary-batch',type=int,default=16)
    p.add_argument('--noisy-batch',type=int,default=16)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--frame-budget',type=int,default=1600)
    p.add_argument('--head-epochs',type=int,default=1)
    p.add_argument('--joint-epochs',type=int,default=5)
    p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--joint-head-lr',type=float,default=1e-5)
    p.add_argument('--encoder-lr',type=float,default=1e-7)
    p.add_argument('--cka-weight',type=float,default=.01)
    p.add_argument('--activation-budget-gib',type=float,default=0.,help='0: bounded automatic available host RAM budget')
    p.add_argument('--keep-activations-on-gpu',action='store_true',help='Explicitly disable CPU offload; needs substantially more VRAM')
    p.add_argument('--warm-checkpoint',help='Optional explicit compatible MultiConv weights; default is original 91.68 encoder + new head')
    p.add_argument('--keep-old-caches',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',help='Exact V3 continuation from this run/last.pt; no new cache generation or recipe changes')
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp',action='store_true')
    p.add_argument('--smoke-steps',type=int,default=0,help='Explicit diagnostic mode; does not save a training checkpoint')
    return p


def configuration(source,args):
    for key in ('workers','smoke_steps'):
        if getattr(args,key) < 0: raise ValueError(key+' must be nonnegative')
    for key in ('cache_workers','microbatch','head_epochs','joint_epochs'):
        if getattr(args,key) < 1: raise ValueError(key+' must be positive')
    if min(args.ordinary_batch,args.noisy_batch)<2 or args.frame_budget<12:
        raise ValueError('Component batches >=2 and frame budget >=12 are required')
    for key in ('head_lr','joint_head_lr','encoder_lr'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0: raise ValueError('Invalid '+key)
    if any(not math.isfinite(v) or v<0 for v in (args.cka_weight,args.activation_budget_gib)):
        raise ValueError('CKA and activation budget must be finite and nonnegative')
    cfg = {key:str(Path(source[key]).expanduser().resolve()) for key in (
        'train_protocol','dev_protocol','train_data_path','dev_data_path','ssl_path','dev_noisy_cache','dev_heldout_cache')}
    baseline = Path(args.baseline).expanduser().resolve()
    digest = sha256(baseline)
    if digest != BASELINE_SHA256:
        raise ValueError('V3 encoder baseline must be the protected original 91.68 checkpoint')
    warm = str(Path(args.warm_checkpoint).expanduser().resolve()) if args.warm_checkpoint else None
    cfg.update(baseline=str(baseline),baseline_sha256=digest,warm_checkpoint=warm,
        warm_checkpoint_sha256=sha256(warm) if warm else None,
        train_caches=[str(Path(args.full_noisy_cache).expanduser().resolve())],full_noisy=True,
        seed=args.seed,device=args.device,amp=args.amp,eval_amp='none',workers=args.workers,
        ordinary_batch=args.ordinary_batch,noisy_batch=args.noisy_batch,eval_batch=8,
        microbatch=args.microbatch,frame_budget=args.frame_budget,max_seconds=0.,
        rawboost=5,rawboost_probability=.5,raw_config={k:source.get(k,v) for k,v in RAW.items()},
        trainable_layers=4,checkpointing=True,offload_activations=not args.keep_activations_on_gpu,
        activation_budget_gib=args.activation_budget_gib,head_epochs=args.head_epochs,joint_epochs=args.joint_epochs,
        head_lr=args.head_lr,joint_head_lr=args.joint_head_lr,encoder_lr=args.encoder_lr,
        head_warmup_steps=200,joint_warmup_steps=100,lr_warmup_steps=100,min_lr_scale=.1,
        weight_decay=1e-4,grad_clip=1.,noisy_weight_start=.3,noisy_weight=.5,noisy_ramp_epochs=1.,
        cka_weight=args.cka_weight,cka_warmup_steps=200,evals_per_epoch=2,
        lr_factor=.5,plateau_evals=2,reduced_lr_evals=2,max_lr_reductions=2,drift_rescues=1,
        en_real_tolerance=.01,noisy_fake_tolerance=.005,selection_min_delta=.0001,drift_score_tolerance=.003,
        input_policy='full utterance',score_column='P(fake)',decision_threshold=.5,
        coverage_policy='every ordinary Train row and both full noisy views exactly once per completed epoch',
        algorithm='w2v-BERT multi-layer fusion + four MultiConv blocks + attentive statistics + exact logical-batch CKA')
    return cfg
