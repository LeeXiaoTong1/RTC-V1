"""Explicit V3.18.6 contract; receiver-only classification and two PCL edges."""
import argparse
import copy
import math
from pathlib import Path
import sys
import torch
from w2v_v318.config import runtime_versions
from w2v_v3184.assets import configured_spec
from w2v_v3185.config import verify_unique
from .common import ROOT, RECIPE, schema_for, read_json, digest
from .parent import inspect_parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',help='Completed V3.18.5 best; default exp/.latest_v3185_run')
    p.add_argument('--parent-tag',help='Optional assertion for the exact parent best tag')
    p.add_argument('--noise-manifest',default=str(ROOT/'assets/noise_v3184_rnnoise/manifest.json'))
    p.add_argument('--phoneme-manifest',default=str(ROOT/'assets/phonemes_v3186/manifest.json'))
    p.add_argument('--consistency',choices=('dual','none'),default='dual',help='none is the matched CE-only control')
    p.add_argument('--pcl-platform-weight',type=float,default=.15)
    p.add_argument('--pcl-noise-weight',type=float,default=.15)
    p.add_argument('--pcl-ramp-updates',type=int,default=300)
    p.add_argument('--platform-sources-per-group',type=int,default=2)
    p.add_argument('--gradient-audit-interval',type=int,default=100)
    p.add_argument('--scene-probability',type=float,default=1.)
    p.add_argument('--rtc-policy',choices=('legacy','three-profile'),default='three-profile')
    p.add_argument('--view-factors',type=int,nargs=4,default=[5,5,1,3],metavar=('EN_FAKE','EN_REAL','ZH_FAKE','ZH_REAL'))
    p.add_argument('--refresh-base-variants',action='store_true')
    p.add_argument('--group-quotas',type=int,nargs=4,default=[4,3,5,4])
    p.add_argument('--scene-panel-per-group',type=int,default=32)
    p.add_argument('--resume')
    p.add_argument('--baseline-only','--preflight-only',dest='baseline_only',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--warm-epochs',type=int,choices=(0,),default=0,help='All epochs are joint; PCL uses an update-level ramp')
    p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--frame-budget',type=int,default=12000)
    p.add_argument('--eval-batch',type=int,default=16)
    p.add_argument('--whole-max-frames',type=int,default=3000,help='Explicit full-utterance resource guard; never truncates audio')
    p.add_argument('--whole-graph-budget',type=int,default=1000000)
    p.add_argument('--whole-batch',type=int,default=8)
    p.add_argument('--rtc-stress-probability',type=float,default=.35)
    p.add_argument('--rtc-stress-ramp-epochs',type=float,default=2.)
    for name,value in (('lora',1e-5),('head',3e-5),('fusion',1e-4),('layer-mix',3e-4),('classifier',1e-4)):
        p.add_argument('--'+name+'-lr',type=float,default=value)
    p.add_argument('--lr-warmup-updates',type=int,default=200)
    p.add_argument('--lr-floor',type=float,default=.1)
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp',action='store_true',help='Upload metrics-only report ZIP to temp.sh')
    p.add_argument('--submit-best',action='store_true')
    p.add_argument('--upload-submission',action='store_true',help='Upload submission and report ZIP links after training')
    p.add_argument('--submission-protocol')
    p.add_argument('--submission-audio-root')
    p.add_argument('--submission-out')
    return p


def validate(args):
    if tuple(args.view_factors)!=(5,5,1,3):
        raise ValueError('V3.18.6 retains V3.18.5 fixed version budgets: 5 5 1 3')
    if tuple(args.group_quotas)!=(4,3,5,4):
        raise ValueError('V3.18.6 retains classification quotas: 4 3 5 4')
    if args.epochs<1 or args.warm_epochs!=0 or args.workers<0:
        raise ValueError('Require positive epochs, zero warm epochs, and nonnegative workers')
    for key,minimum in (('microbatch',4),('frame_budget',48),('eval_batch',1),('whole_max_frames',12),
                        ('whole_graph_budget',16),('whole_batch',1),('pcl_ramp_updates',1),
                        ('gradient_audit_interval',1),('lr_warmup_updates',1),('scene_panel_per_group',1)):
        if getattr(args,key)<minimum:raise ValueError('Invalid '+key)
    if not 1<=args.platform_sources_per_group<=min(args.group_quotas):
        raise ValueError('Platform source quota must fit every classification group')
    for key in ('lora_lr','head_lr','fusion_lr','layer_mix_lr','classifier_lr','rtc_stress_ramp_epochs'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<=0:raise ValueError('Invalid '+key)
    for key in ('pcl_platform_weight','pcl_noise_weight'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<0:raise ValueError('Invalid '+key)
    if args.consistency=='dual' and min(args.pcl_platform_weight,args.pcl_noise_weight)<=0:
        raise ValueError('dual requires both positive PCL weights; use --consistency none for CE-only')
    if not 0<=args.scene_probability<=1 or not 0<=args.rtc_stress_probability<=.5 or not 0<args.lr_floor<=1:
        raise ValueError('Invalid augmentation probability or LR floor')
    if args.upload_submission and not args.submit_best:raise ValueError('--upload-submission requires --submit-best')
    if not args.submit_best and any(getattr(args,k) for k in ('submission_protocol','submission_audio_root','submission_out')):
        raise ValueError('Submission paths require --submit-best')


def validate_locked_cli(args,argv):
    if not args.resume:return
    allowed={'--resume','--baseline-only','--preflight-only','--download-dir','--upload-temp','--help',
             '--submit-best','--upload-submission','--submission-protocol','--submission-audio-root','--submission-out'}
    changed={a.split('=')[0] for a in argv if a.startswith('--')}-allowed
    if changed:raise ValueError('Resume uses the saved configuration; remove overrides: '+', '.join(sorted(changed)))


def configuration(args):
    validate(args)
    source=args.parent_run
    if not source:
        pointer=ROOT/'exp/.latest_v3185_run'
        if not pointer.is_file():raise FileNotFoundError('Specify a completed V3.18.5 --parent-run')
        source=pointer.read_text(encoding='utf-8').strip()
    print('[Startup] inspecting completed V3.18.5 BEST',flush=True)
    old,weights,parent=inspect_parent(source,args.parent_tag);del weights
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production training requires BF16 CUDA; CPU fixtures are tests only')
    if old.get('omni_arch')!='3b':raise ValueError('Production V3.18.6 requires an Omni-3B parent')
    cfg=copy.deepcopy(old)
    for key in ('pool_lr','evidence_lr','warm_lr','warm_fusion_lr','warm_layer_mix_lr','warm_evidence_lr','view_factor'):
        cfg.pop(key,None)
    keys=('epochs','workers','device','microbatch','frame_budget','eval_batch','whole_max_frames','whole_graph_budget',
          'whole_batch','consistency','pcl_platform_weight','pcl_noise_weight','pcl_ramp_updates','platform_sources_per_group',
          'gradient_audit_interval','scene_probability','rtc_policy','view_factors','refresh_base_variants','group_quotas',
          'scene_panel_per_group','rtc_stress_probability','rtc_stress_ramp_epochs','lora_lr','head_lr','fusion_lr',
          'layer_mix_lr','classifier_lr','lr_warmup_updates','lr_floor')
    cfg.update({key:getattr(args,key) for key in keys})
    cfg.update(version='3.18.6',recipe=RECIPE,pooling='whole',local_evidence=False,warm_epochs=0,
        base_original_fraction=.2,stream_sources=16,patience=0,fusion_dim=128,
        fusion_layers=list(range(1,old['encoder_layers']+1)),
        initialization=dict(parent,mode='parent-transfer',old_optimizer_loaded=False,
            transfer='LoRA, all-layer fusion, AASIST; initialize 160d classifier from parent mean columns',
            discarded='external window pooling, std classifier columns, optimizer',initial_function_preserved=False,
            exact_continuation=False),
        bn_momentum=old.get('bn_momentum',.01),
        bn_policy='inherit parent buffers; update on unpadded classification utterances only, never auxiliary references',
        inference_requires_phonemes=False,phoneme_manifest=None,phoneme_files={})
    if args.consistency=='none':cfg.update(pcl_platform_weight=0.,pcl_noise_weight=0.)
    from w2v_v3184.noise import bind_manifest as bind_noise
    cfg.update(bind_noise(args.noise_manifest))
    run=Path(parent['run']);train=read_json(run/'train_rows.json');dev=read_json(run/'dev_rows.json')
    from .resources import audit_training
    cfg['whole_utterance_resource_audit']=audit_training(train,dev,cfg)
    if args.consistency=='dual':
        from .phonemes import bind_manifest
        phonemes=bind_manifest(args.phoneme_manifest,train)
        cfg['phoneme_manifest']=str(Path(args.phoneme_manifest).resolve())
        cfg['phoneme_provenance']=phonemes
        cfg['phoneme_files']={cfg['phoneme_manifest']:digest(cfg['phoneme_manifest'])}
    cfg['data_files'].update(parent['files'])
    # Include production modules only; new tests can be added without invalidating a run.
    for name in ('w2v_v3186','w2v_v3185','w2v_v3183r2','w2v_v3184'):
        cfg['code_fingerprints'].update({str(p):digest(p) for p in (ROOT/name).glob('*.py') if not p.name.startswith('test_')})
    from .sampling import Plan
    plan=Plan(train,16,cfg['seed'],cfg['group_quotas'],cfg['view_factors'],cfg['refresh_base_variants'],cfg=cfg)
    cfg['augmentation_budget_audit']=plan.audit_budget(cfg['epochs'])
    print('[Startup] verifying pinned code, manifests, RTC assets, and Omni checkpoint',flush=True)
    verify_inputs(cfg)
    print('[Startup] input verification complete',flush=True)
    return cfg,train,dev


def verify_inputs(cfg,*,training=True):
    schema_for(cfg);configured_spec(cfg)
    verify_unique(cfg['data_files'],cfg['code_fingerprints'],cfg['augmentation_files'],
        {cfg['omni_checkpoint']:cfg['omni_sha256'],cfg['ffmpeg']:cfg['ffmpeg_sha256']},
        cfg['initialization']['files'],cfg['scene_noise_files'],cfg.get('phoneme_files',{}) if training else {})
    if cfg['runtime_versions']!=runtime_versions():raise ValueError('Pinned runtime changed')
    if training and cfg['consistency']=='dual':
        from .phonemes import bind_manifest
        rows=read_json(Path(cfg['initialization']['run'])/'train_rows.json')
        if bind_manifest(cfg['phoneme_manifest'],rows)!=cfg['phoneme_provenance']:
            raise ValueError('Phoneme inventory changed')


if __name__=='__main__':
    args=parser().parse_args();validate(args);validate_locked_cli(args,sys.argv[1:])
