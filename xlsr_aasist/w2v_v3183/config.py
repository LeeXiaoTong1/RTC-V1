"""V3.18.3: six additional C3 epochs from verified epoch 10, short LR warmup."""
import argparse
import copy
import math
from pathlib import Path
import sys
import torch
from w2v_v318.config import runtime_versions, verify_inputs as verify_parent_inputs
from .common import ROOT, read_json, digest, verify_files
from .assets import configured_spec
from .parent import inspect_parent, CONFIRMED_RUN


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',default=str(ROOT/'exp'/CONFIRMED_RUN))
    p.add_argument('--parent-tag',default='epoch_10_step_2417')
    p.add_argument('--resume');p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int,default=6,help='Additional epochs, numbered 1-6 in the new run')
    p.add_argument('--microbatch',type=int);p.add_argument('--frame-budget',type=int);p.add_argument('--eval-batch',type=int)
    p.add_argument('--group-quotas',type=int,nargs=4,default=[4,2,7,3],metavar=('EN_FAKE','EN_REAL','ZH_FAKE','ZH_REAL'))
    p.add_argument('--fusion-layers',type=int,nargs='+',default=[15,30,45],help='One-based intermediate layers; final layer is retained')
    p.add_argument('--no-fusion',action='store_true');p.add_argument('--fusion-max-residual',type=float,default=.25)
    p.add_argument('--rtc-stress-probability',type=float,default=.35,help='Fraction of Noisy B; 0 retains legacy RTC')
    p.add_argument('--rtc-stress-ramp-epochs',type=float,default=2.)
    p.add_argument('--lora-lr',type=float,default=2e-6);p.add_argument('--head-lr',type=float,default=6e-6)
    p.add_argument('--evidence-lr',type=float,default=1e-5);p.add_argument('--fusion-lr',type=float,default=1e-4)
    p.add_argument('--lr-warmup-updates',type=int,default=200);p.add_argument('--lr-floor',type=float,default=.2)
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp');p.add_argument('--upload-temp',action='store_true')
    return p


def validate(args):
    if args.epochs<1 or args.workers<0 or min(args.group_quotas)<1 or sum(args.group_quotas)!=16:
        raise ValueError('Positive epochs, four positive quotas summing to 16, nonnegative workers required')
    for name,minimum in (('microbatch',2),('frame_budget',12),('eval_batch',1)):
        if getattr(args,name) is not None and getattr(args,name)<minimum:raise ValueError('Invalid '+name)
    for key in ('lora_lr','head_lr','evidence_lr','fusion_lr','rtc_stress_ramp_epochs'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0:raise ValueError('Invalid '+key)
    if not 0<=args.rtc_stress_probability<=.5 or not 0<args.fusion_max_residual<=.5 or not 0<args.lr_floor<=1 or args.lr_warmup_updates<1:
        raise ValueError('Invalid fusion, RTC, or LR schedule')
    if len(set(args.fusion_layers))!=len(args.fusion_layers) or min(args.fusion_layers)<1:raise ValueError('Invalid fusion layers')


def validate_locked_cli(args,argv):
    if not args.resume:return
    allowed={'--resume','--baseline-only','--download-dir','--upload-temp','--help'}
    changed={s.split('=')[0] for s in argv if s.startswith('--')}-allowed
    if changed:raise ValueError('Resume uses the saved configuration; remove overrides: '+', '.join(sorted(changed)))


def configuration(args):
    validate(args)
    old,weights,parent=inspect_parent(args.parent_run,args.parent_tag);del weights
    verify_parent_inputs(old)
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production Omni continuation requires BF16 CUDA; CPU fixtures are tests only')
    cfg=copy.deepcopy(old)
    if old.get('omni_arch')!='3b':raise ValueError('Default layer plan is for the confirmed Omni-3B parent')
    if max(args.fusion_layers)>=old['encoder_layers']:raise ValueError('Fusion must read intermediate layers')
    cfg.update(version='3.18.3',epochs=args.epochs,warm_epochs=0,workers=args.workers,device=args.device,
        stream_sources=16,group_quotas=args.group_quotas,fusion_layers=[] if args.no_fusion else args.fusion_layers,
        fusion_max_residual=args.fusion_max_residual,feature_layer='final plus zero-initialized bounded intermediate residual',
        rtc_stress_probability=args.rtc_stress_probability,rtc_stress_ramp_epochs=args.rtc_stress_ramp_epochs,
        lora_lr=args.lora_lr,head_lr=args.head_lr,evidence_lr=args.evidence_lr,fusion_lr=args.fusion_lr,
        lr_warmup_updates=args.lr_warmup_updates,lr_floor=args.lr_floor,patience=0,
        initialization=dict(parent,mode='epoch10_weights_new_optimizer_zero_fusion',old_detector_loaded=True,
                            old_optimizer_loaded=False,parent_epoch=10),
        bn_policy='inherit parent buffers; frozen running statistics, trainable affine for all continuation epochs')
    for key in ('microbatch','frame_budget','eval_batch'):
        if getattr(args,key) is not None:cfg[key]=getattr(args,key)
    cfg['code_fingerprints'].update({str(p):digest(p) for p in (ROOT/'w2v_v3183').glob('*.py') if not p.name.startswith('test_')})
    # The calibration/metrics modules are copied into this namespace; no dependency
    # on mutable V3.18.2 training code is introduced.
    cfg['data_files'].update(parent['files'])
    run=Path(parent['run']);train=read_json(run/'train_rows.json');dev=read_json(run/'dev_rows.json')
    from .sampling import Plan
    Plan(train,16,cfg['seed'],args.group_quotas)
    verify_inputs(cfg)
    return cfg,train,dev


def verify_inputs(cfg):
    if cfg.get('version')!='3.18.3':raise ValueError('Not V3.18.3')
    configured_spec(cfg)
    for key in ('data_files','code_fingerprints','augmentation_files'):verify_files(cfg[key])
    verify_files({cfg['omni_checkpoint']:cfg['omni_sha256'],cfg['ffmpeg']:cfg['ffmpeg_sha256']})
    if cfg['runtime_versions']!=runtime_versions():raise ValueError('Pinned runtime changed')
    verify_files(cfg['initialization']['files'])


if __name__=='__main__':
    args=parser().parse_args();validate(args);validate_locked_cli(args,sys.argv[1:])
