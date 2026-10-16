"""Activity-conditioned RTC continuation of the complete V3.18.8 best detector."""
import argparse
import ast
import copy
import math
from pathlib import Path
import torch
from w2v_v318.config import runtime_versions
from w2v_v3184.assets import configured_spec
from w2v_v3185.config import verify_unique
from .common import ROOT, RECIPE, schema_for, read_json, digest


def code_inventory(root=ROOT):
    root=Path(root); pending=list((root/'w2v_v3189').glob('*.py')); seen={}
    while pending:
        path=pending.pop()
        if path in seen or not path.is_file() or path.name.startswith('test_'):
            continue
        seen[path]=digest(path); package=list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import): names=[a.name.split('.') for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                target=prefix+(node.module.split('.') if node.module else [])
                names=[target]+[target+a.name.split('.') for a in node.names if a.name!='*']
            for parts in names:
                for depth in range(1,len(parts)+1):
                    candidate=root.joinpath(*parts[:depth])
                    pending.extend((candidate.with_suffix('.py'),candidate/'__init__.py'))
    return {str(path):value for path,value in seen.items()}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run','--init-run',dest='parent_run',help='Stopped V3.18.8 run; defaults to its latest-run marker')
    p.add_argument('--resume',help='Resume an existing V3.18.9 run without recipe overrides')
    p.add_argument('--epochs',type=int,default=6,help='Additional epochs, starting from the parent best')
    p.add_argument('--warm-epochs',type=int,default=0,help='Must be zero: complete trained detector and frozen PCL scale are inherited')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=None,help='Auto: half available CPU cores, max 8, reduced for host RAM; 0 disables multiprocessing')
    p.add_argument('--eval-workers',type=int,default=None,help='Default min(4, train workers)')
    p.add_argument('--prefetch-factor',type=int,default=1,help='Queued batches per worker, default 1')
    p.add_argument('--backend-precision',choices=('fp16','fp32'),default='fp16',help='AASIST mixed precision; parameters, pooling and losses remain FP32')
    p.add_argument('--microbatch',type=int,default=12,help='Encoded views, INCLUDING PCL references; logical CE batch remains 48')
    p.add_argument('--frame-budget',type=int,default=9000)
    p.add_argument('--eval-batch',type=int,default=12)
    p.add_argument('--window-batch',type=int,default=16)
    p.add_argument('--rtc-activity',choices=('mixed','off'),default='mixed',help='Mixed pause attenuation, or an otherwise identical control')
    p.add_argument('--baseline-only','--preflight-only',dest='baseline_only',action='store_true')
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp',action='store_true')
    p.add_argument('--submit-best',action='store_true')
    p.add_argument('--upload-submission',action='store_true')
    p.add_argument('--submission-protocol')
    p.add_argument('--submission-audio-root')
    p.add_argument('--submission-out')
    return p


def validate(args):
    if args.epochs<1 or args.warm_epochs!=0:
        raise ValueError('Use positive additional epochs and --warm-epochs 0')
    if args.parent_run and args.resume:
        raise ValueError('Choose --parent-run for a new run or --resume for an existing V3.18.9 run')
    if any(v is not None and v<0 for v in (args.workers,args.eval_workers)) or args.prefetch_factor<1 or args.microbatch<12 or min(args.frame_budget,args.eval_batch,args.window_batch)<1:
        raise ValueError('Require microbatch >=12 for four three-view platform units; other budgets must be positive')
    if args.upload_submission and not args.submit_best:
        raise ValueError('--upload-submission requires --submit-best')
    if not args.submit_best and any(getattr(args,k) for k in ('submission_protocol','submission_audio_root','submission_out')):
        raise ValueError('Submission paths require --submit-best')


def validate_locked_cli(args,argv):
    if not args.resume:return
    allowed={'--resume','--baseline-only','--preflight-only','--download-dir','--upload-temp','--help',
             '--submit-best','--upload-submission','--submission-protocol','--submission-audio-root','--submission-out'}
    changed={a.split('=')[0] for a in argv if a.startswith('--')}-allowed
    if changed:raise ValueError('Resume uses saved settings; remove overrides: '+', '.join(sorted(changed)))


def end_learning_rates(old,epoch,steps):
    """Reconstruct the verified parent schedule at its SELECTED best boundary.

    Never load the last epoch's optimizer together with an older best model.
    The new optimizer is fresh, initialized at these learning rates.
    """
    from types import SimpleNamespace
    from w2v_v3188.step import schedule as parent_schedule
    keys=('lora_early_lr','lora_middle_lr','lora_lr','head_lr','fusion_lr','layer_mix_lr',
          'classifier_lr','evidence_lr','pool_lr')
    groups=[dict(name=key,role='lora' if key.startswith('lora') else 'head',
                 initial_lr=old[key],lr=old[key]) for key in keys]
    opt=SimpleNamespace(param_groups=groups)
    if epoch > 0:
        parent_schedule(opt,old,epoch-1,steps-1,steps)
    # If the best was the head-warmup endpoint, adapters have not started yet.
    # Use their configured initial rates rather than freezing them forever.
    return {g['name']:float(g['lr'] or g['initial_lr']) for g in groups}


def configuration(args):
    from .parent import metadata
    old,origin,train,dev=metadata(args.parent_run)
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production training requires BF16 CUDA; CPU fixtures are tests only')
    cfg=copy.deepcopy(old)
    for key in ('saved_manifests','warm_start','augmentation_budget_audit','window_resource_audit'):
        cfg.pop(key,None)
    cfg.update(version='3.18.9',recipe=RECIPE,epochs=args.epochs,warm_epochs=0,
        device=args.device,backend_precision=args.backend_precision,microbatch=args.microbatch,frame_budget=args.frame_budget,
        eval_batch=args.eval_batch,window_batch=args.window_batch,checkpoint_layers=old['encoder_layers'],
        checkpointing=True,fusion_checkpointing=True,noisy_views_per_source=1,
        classification_objective='balanced_single_view_ce',risk_coefficient=0.,
        lr_warmup_updates=1,pcl_ramp_updates=1,
        rtc_activity_policy=args.rtc_activity,
        rtc_activity=dict(policy='energy_hysteresis_v1',profile_probabilities={'off':.25,'mild':.45,'strong':.30},
            placement='after NS/AGC, before Opus',pcl_mask_db=12.,pcl_codec_guard_samples=320,pcl_rf_guard_samples=200,
            detector='Processed-signal energy with utterance quantile initialization; no class/phone/clean oracle',
            inference_applied=False),
        source_epoch_offset=int(old.get('source_epoch_offset',0))+origin['epoch'],
        parent_init=origin,initialization=dict(mode='v3188_best',run=origin['run'],
            selected=origin['tag'],checkpoint_loaded=True,old_detector_loaded=True,old_optimizer_loaded=False,
            transfer='Complete best detector, all LoRA/fusion/AASIST/ASP/classifier tensors and frozen Train PCL scale',
            fresh='Optimizer moments only; no structural conversion'),
        data_source=dict(run=origin['run'],files=origin['manifest_files'],checkpoint_loaded=True),
        data_files=origin['manifest_files'],runtime_versions=runtime_versions())
    from .workers import settings
    cfg.update(settings(args,old))
    cfg.update(end_learning_rates(old,origin['epoch'],origin['steps']))
    cfg['learning_rate_origin']=dict(epoch=origin['epoch'],tag=origin['tag'],
        policy='Selected best endpoint rates; fresh optimizer; cosine decay over additional epochs; no repeated warmup')
    if cfg.get('consistency')!='dual':
        raise ValueError('The V3.18.9 parent must use the verified dual-PCL recipe')
    if cfg['microbatch']<3*cfg.get('pcl_variance_min_sources',4):
        raise ValueError('Encoded-view cap cannot fit the parent minimum number of platform variance sources')
    from .resources import audit_training
    cfg['window_resource_audit']=audit_training(train,dev,cfg)
    cfg['code_fingerprints']=code_inventory()
    from .sampling import Plan
    plan=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas'],cfg['view_factors'],cfg['refresh_base_variants'],cfg=cfg)
    cfg['augmentation_budget_audit']=plan.audit_budget(cfg['epochs'])
    print('[Startup] verifying inherited assets, one-Noisy budgets and full-recording memory bounds',flush=True)
    verify_inputs(cfg)
    return cfg,train,dev


def verify_inputs(cfg,*,training=True):
    schema_for(cfg);configured_spec(cfg)
    verify_unique(cfg['code_fingerprints'],{cfg['omni_checkpoint']:cfg['omni_sha256']},
        cfg['data_files'] if training else {},cfg['augmentation_files'] if training else {},
        {cfg['ffmpeg']:cfg['ffmpeg_sha256']} if training else {},
        cfg['scene_noise_files'] if training else {},cfg.get('phoneme_files',{}) if training else {})
    if cfg['runtime_versions']!=runtime_versions():raise ValueError('Pinned runtime changed')
    if training and cfg['consistency']=='dual':
        from .phonemes import bind_manifest
        rows=read_json(Path(cfg['data_source']['run'])/'train_rows.json')
        if bind_manifest(cfg['phoneme_manifest'],rows)!=cfg['phoneme_provenance']:
            raise ValueError('Phoneme inventory changed')


if __name__=='__main__':
    import sys
    args=parser().parse_args();validate(args);validate_locked_cli(args,sys.argv[1:])
