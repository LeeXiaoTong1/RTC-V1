"""V3.18.3-R2: full-layer fusion, explicit parent transfer, and coordinated learning rates."""
import argparse
import copy
import math
from pathlib import Path
import sys
import torch
from w2v_v318.config import runtime_versions
from .common import ROOT, read_json, digest, verify_files
from .assets import configured_spec
from .parent import inspect_parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',help='Completed V3.18.3 data/weight source; default exp/.latest_v3183_run')
    p.add_argument('--parent-tag',help='Optional assertion for the selected parent best tag')
    p.add_argument('--init-mode',choices=('parent-transfer','pretrained'),default='parent-transfer')
    p.add_argument('--resume');p.add_argument('--baseline-only','--preflight-only',dest='baseline_only',action='store_true',
        help='Verify initialization and cached parent reference; save the initial state without training')
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int,default=6,help='New-run epochs, including head/fusion warmup')
    p.add_argument('--warm-epochs',type=int,default=1,help='Keep inherited LoRA active but frozen while training fusion and head')
    p.add_argument('--microbatch',type=int);p.add_argument('--frame-budget',type=int);p.add_argument('--eval-batch',type=int)
    p.add_argument('--group-quotas',type=int,nargs=4,default=[4,2,7,3],metavar=('EN_FAKE','EN_REAL','ZH_FAKE','ZH_REAL'))
    p.add_argument('--rtc-stress-probability',type=float,default=.35,help='Fraction of Noisy B; unchanged from V3.18.3')
    p.add_argument('--rtc-stress-ramp-epochs',type=float,default=2.)
    p.add_argument('--lora-lr',type=float,default=2e-5);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--evidence-lr',type=float,default=1e-4);p.add_argument('--fusion-lr',type=float,default=3e-4)
    p.add_argument('--layer-mix-lr',type=float,default=1e-3)
    p.add_argument('--warm-lr',type=float,default=1e-4);p.add_argument('--warm-fusion-lr',type=float,default=3e-4)
    p.add_argument('--warm-layer-mix-lr',type=float,default=1e-3);p.add_argument('--warm-evidence-lr',type=float,default=1e-4)
    p.add_argument('--lr-warmup-updates',type=int,default=200,help='Ramp at the beginning of head warmup and of joint adaptation')
    p.add_argument('--lr-floor',type=float,default=.1)
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp');p.add_argument('--upload-temp',action='store_true')
    p.add_argument('--submit-best',action='store_true',help='After completion, export this run best to submission.zip')
    p.add_argument('--upload-submission',action='store_true',help='With --submit-best, upload only submission.zip to temp.sh')
    p.add_argument('--submission-protocol',help='Unlabeled ID list; default EVAL_PROTOCOL or the existing Progress path')
    p.add_argument('--submission-audio-root',help='Default EVAL_AUDIO_ROOT or the existing Progress audio path')
    p.add_argument('--submission-out',help='Default: download-dir/RUN_submission_best; saved for resume')
    return p


def validate(args):
    if args.upload_submission and not args.submit_best:raise ValueError('--upload-submission requires --submit-best')
    if not args.submit_best and any(getattr(args,k) for k in ('submission_protocol','submission_audio_root','submission_out')):
        raise ValueError('Submission paths require --submit-best')
    if args.epochs<1 or not 0<=args.warm_epochs<args.epochs or args.workers<0 or min(args.group_quotas)<1 or sum(args.group_quotas)!=16:
        raise ValueError('Positive epochs, 0 <= warm epochs < epochs, four positive quotas summing to 16, and nonnegative workers required')
    for name,minimum in (('microbatch',2),('frame_budget',12),('eval_batch',1)):
        if getattr(args,name) is not None and getattr(args,name)<minimum:raise ValueError('Invalid '+name)
    for key in ('lora_lr','head_lr','evidence_lr','fusion_lr','layer_mix_lr','warm_lr','warm_fusion_lr',
                'warm_layer_mix_lr','warm_evidence_lr','rtc_stress_ramp_epochs'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0:raise ValueError('Invalid '+key)
    if not 0<=args.rtc_stress_probability<=.5 or not 0<args.lr_floor<=1 or args.lr_warmup_updates<1:
        raise ValueError('Invalid RTC or LR schedule')


def validate_locked_cli(args,argv):
    if not args.resume:return
    allowed={'--resume','--baseline-only','--preflight-only','--download-dir','--upload-temp','--help','--submit-best',
             '--upload-submission','--submission-protocol','--submission-audio-root','--submission-out'}
    changed={s.split('=')[0] for s in argv if s.startswith('--')}-allowed
    if changed:raise ValueError('Resume uses the saved configuration; remove overrides: '+', '.join(sorted(changed)))


def configuration(args):
    validate(args)
    source=args.parent_run
    if not source:
        pointer=ROOT/'exp'/'.latest_v3183_run'
        if not pointer.is_file():raise FileNotFoundError('Specify --parent-run for a completed V3.18.3 run')
        source=pointer.read_text(encoding='utf-8').strip()
    print('[Startup] inspecting completed V3.18.3 best checkpoint and frozen data manifests',flush=True)
    old,weights,parent=inspect_parent(source,args.parent_tag);del weights
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production Omni training requires BF16 CUDA; CPU fixtures are tests only')
    if old.get('omni_arch')!='3b':raise ValueError('Production R2 requires the completed Omni-3B parent')
    cfg=copy.deepcopy(old)
    cfg.pop('fusion_max_residual',None)
    cfg.update(version='3.18.3-R2',epochs=args.epochs,warm_epochs=args.warm_epochs,workers=args.workers,device=args.device,
        stream_sources=16,group_quotas=args.group_quotas,fusion_layers=list(range(1,old['encoder_layers']+1)),fusion_dim=128,
        feature_layer='all transformer blocks; per-frame LayerNorm, shared projection and SwiGLU, learned softmax mixture',
        rtc_stress_probability=args.rtc_stress_probability,rtc_stress_ramp_epochs=args.rtc_stress_ramp_epochs,
        lora_lr=args.lora_lr,head_lr=args.head_lr,evidence_lr=args.evidence_lr,fusion_lr=args.fusion_lr,layer_mix_lr=args.layer_mix_lr,
        warm_lr=args.warm_lr,warm_fusion_lr=args.warm_fusion_lr,warm_evidence_lr=args.warm_evidence_lr,
        warm_layer_mix_lr=args.warm_layer_mix_lr,lr_warmup_updates=args.lr_warmup_updates,lr_floor=args.lr_floor,patience=0,
        initialization=dict(parent,mode=args.init_mode,old_detector_loaded=args.init_mode=='parent-transfer',
            old_optimizer_loaded=False,transfer='LoRA, AASIST excluding LL, evidence, LL to shared projection' if args.init_mode=='parent-transfer' else 'public Omni only',
            discarded='parent bounded fusion; parent optimizer; parent BN running statistics'),
        bn_momentum=.01,bn_policy='reset parent running statistics once; update running statistics in all training epochs with momentum .01; saved statistics at evaluation')
    for key in ('microbatch','frame_budget','eval_batch'):
        if getattr(args,key) is not None:cfg[key]=getattr(args,key)
    cfg['code_fingerprints'].update({str(p):digest(p) for p in (ROOT/'w2v_v3183r2').glob('*.py') if not p.name.startswith('test_')})
    cfg['data_files'].update(parent['files'])
    run=Path(parent['run']);train=read_json(run/'train_rows.json');dev=read_json(run/'dev_rows.json')
    from .sampling import Plan
    Plan(train,16,cfg['seed'],args.group_quotas)
    print('[Startup] verifying pinned code, audio manifests, RTC assets, and Omni checkpoint; large-file hashing can take time',flush=True)
    verify_inputs(cfg)
    print('[Startup] input verification complete',flush=True)
    return cfg,train,dev


def verify_inputs(cfg):
    if cfg.get('version')!='3.18.3-R2':raise ValueError('Not V3.18.3-R2')
    configured_spec(cfg)
    verify_unique(cfg['data_files'],cfg['code_fingerprints'],cfg['augmentation_files'],
                  {cfg['omni_checkpoint']:cfg['omni_sha256'],cfg['ffmpeg']:cfg['ffmpeg_sha256']},
                  cfg['initialization']['files'])
    if cfg['runtime_versions']!=runtime_versions():raise ValueError('Pinned runtime changed')


def verify_unique(*groups):
    merged={}
    for values in groups:
        for path,expected in values.items():
            name=str(Path(path).resolve())
            if name in merged and merged[name]!=expected:raise ValueError('Conflicting input fingerprints: '+name)
            merged[name]=expected
    verify_files(merged)


if __name__=='__main__':
    args=parser().parse_args();validate(args);validate_locked_cli(args,sys.argv[1:])
