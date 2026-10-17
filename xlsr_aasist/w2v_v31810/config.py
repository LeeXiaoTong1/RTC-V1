"""Fresh official Omni + E0/E1/E2 matched 15-epoch first-order MetaDG recipe."""
import argparse
import ast
import copy
from pathlib import Path
import torch
from w2v_v318.config import runtime_versions
from w2v_v3184.assets import configured_spec
from w2v_v3185.config import verify_unique
from .common import ROOT,RECIPE,schema_for,read_json,digest,fingerprint
from .source import resolve_data_run,metadata
from .workers import settings


def code_inventory(root=ROOT):
    root=Path(root);pending=list((root/'w2v_v31810').glob('*.py'));seen={}
    while pending:
        path=pending.pop()
        if path in seen or not path.is_file() or path.name.startswith('test_'):continue
        seen[path]=digest(path);package=list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import):names=[a.name.split('.') for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                target=prefix+(node.module.split('.') if node.module else [])
                names=[target]+[target+a.name.split('.') for a in node.names if a.name!='*']
            for parts in names:
                for depth in range(1,len(parts)+1):
                    candidate=root.joinpath(*parts[:depth]);pending.extend((candidate.with_suffix('.py'),candidate/'__init__.py'))
    return {str(p):v for p,v in seen.items()}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-run',help='Reuse manifests/assets ONLY; never load a detector checkpoint')
    p.add_argument('--experiment',choices=('E0','E1','E2'),default='E2')
    p.add_argument('--resume',help='Resume only this V3.18.10 experiment at a committed outer-update boundary')
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=None)
    p.add_argument('--eval-workers',type=int,default=None)
    p.add_argument('--prefetch-factor',type=int,default=1)
    p.add_argument('--microbatch',type=int,default=12)
    p.add_argument('--frame-budget',type=int,default=9000)
    p.add_argument('--whole-max-frames',type=int,default=3000)
    p.add_argument('--eval-batch',type=int,default=12)
    p.add_argument('--window-batch',type=int,default=16)
    p.add_argument('--backend-precision',choices=('fp16','fp32'),default='fp16')
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
    if args.epochs<4:raise ValueError('Use at least 4 epochs; matched protocol is 15')
    if min(args.microbatch,args.frame_budget,args.whole_max_frames,args.eval_batch,args.window_batch,args.prefetch_factor)<1:raise ValueError('Positive resource budgets required')
    if any(x is not None and x<0 for x in (args.workers,args.eval_workers)):raise ValueError('workers must be >=0')
    if args.upload_submission and not args.submit_best:raise ValueError('--upload-submission requires --submit-best')
    if args.resume and args.data_run:raise ValueError('Choose --data-run or --resume')


def validate_locked_cli(args,argv):
    if not args.resume:return
    allowed={'--resume','--baseline-only','--preflight-only','--download-dir','--upload-temp','--help',
        '--submit-best','--upload-submission','--submission-protocol','--submission-audio-root','--submission-out'}
    changed={a.split('=')[0] for a in argv if a.startswith('--')}-allowed
    if changed:raise ValueError('Resume uses saved recipe; remove overrides: '+', '.join(sorted(changed)))


def training_configuration(old,args,source):
    validate(args);cfg=copy.deepcopy(old)
    for key in ('saved_manifests','parent_init','warm_start','augmentation_budget_audit','window_resource_audit',
                'learning_rate_origin','warm_lr','warm_fusion_lr','warm_layer_mix_lr','warm_evidence_lr','phoneme_provenance'):
        cfg.pop(key,None)
    cfg.update(version='3.18.10',recipe=RECIPE,experiment=args.experiment,seed=3181001,
        epochs=args.epochs,warm_epochs=1,source_epoch_offset=0,stream_sources=24,
        source_draws_per_stream=[9668,8000,8000,8000],group_quotas=[6,6,6,6],
        view_factors=[5,5,1,3],refresh_base_variants=True,base_original_fraction=.2,
        pooling='asp',local_evidence=True,temporal_stats=True,window=128,hop=64,
        fusion_dim=128,fusion_layers=list(range(1,old['encoder_layers']+1)),
        lora_layers=old['encoder_layers'],lora_rank=16,lora_alpha=32.,lora_dropout=.05,
        lora_early_lr=5e-6,lora_middle_lr=1e-5,lora_lr=2e-5,
        fusion_lr=1e-4,layer_mix_lr=1e-4,head_lr=1e-4,pool_lr=1e-4,classifier_lr=1e-4,evidence_lr=1e-4,
        weight_decay=1e-4,max_grad_norm=1.,label_smoothing=.02,lr_floor=.1,lr_warmup_updates=200,
        meta_beta=.5,meta_beta_ramp_epochs=2,inner_grad_norm=1.,
        inner_candidates=[.001,.003,.01,.03,.1],calibration_tasks=12,
        gradient_audit_interval=200,diagnostic_interval=1,
        consistency='none',pcl_noise_weight=0.,pcl_platform_weight=0.,pcl_variance_weight=0.,
        platform_sources_per_group=0,en_real_confidence_weight=0.,pcl_objective=None,
        phoneme_manifest=None,phoneme_files={},inference_requires_phonemes=False,
        noisy_views_per_source=1,risk_coefficient=0.,classification_objective='source_isolated_metadg_ce',
        scene_probability=1.,rtc_policy='three-profile',rtc_activity_policy='mixed',
        checkpoint_layers=old['encoder_layers'],checkpointing=True,fusion_checkpointing=True,
        short_input_policy='reflect_to_3920',normalization='group_layer',patience=0,
        device=args.device,microbatch=args.microbatch,frame_budget=args.frame_budget,
        whole_max_frames=args.whole_max_frames,
        eval_batch=args.eval_batch,window_batch=args.window_batch,backend_precision=args.backend_precision,
        data_source=source,data_files=source['files'],selection='0.30 online + 0.35 regular noisy + 0.35 heldout anlmdn macro F1; fixed .5',
        initialization=dict(mode='pretrained',checkpoint_loaded=False,old_detector_loaded=False,old_optimizer_loaded=False,
            transfer='Official Omni SSL backbone only; all detector adapters/reader/backend/classifier initialized afresh'),
        runtime_versions=runtime_versions())
    cfg.update(settings(args,old));schema_for(cfg)
    return cfg


def configuration(args):
    source=resolve_data_run(args.data_run);old,provenance=metadata(source)
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production training requires BF16 CUDA; CPU fixtures are tests only')
    cfg=training_configuration(old,args,provenance)
    train=read_json(source/'train_rows.json');dev=read_json(source/'dev_rows.json')
    from .validation import build_protocol
    protocol=build_protocol(cfg,train,dev,source)
    from .resources import audit_training
    cfg['window_resource_audit']=audit_training(train,protocol['select']['rows']+protocol['audit']['rows'],cfg)
    from .sampling import Plan
    plan=Plan(train,cfg=cfg,seed=cfg['seed'])
    cfg['augmentation_budget_audit']=plan.audit_budget(cfg['epochs'])
    cfg['validation_protocol_hash']=fingerprint(protocol)
    cfg['validation_input_files']=protocol['input_files']
    cfg['code_fingerprints']=code_inventory()
    print('[Startup] verifying fresh-model assets, source-isolated tasks and locked validation protocol',flush=True)
    verify_inputs(cfg)
    return cfg,train,dev,protocol


def verify_inputs(cfg,*,training=True):
    schema_for(cfg);configured_spec(cfg)
    verify_unique(cfg['code_fingerprints'],{cfg['omni_checkpoint']:cfg['omni_sha256']},
        cfg['data_files'] if training else {},cfg['augmentation_files'] if training else {},
        {cfg['ffmpeg']:cfg['ffmpeg_sha256']} if training else {},cfg['scene_noise_files'] if training else {},
        cfg.get('validation_input_files',{}) if training else {})
    if cfg['runtime_versions']!=runtime_versions():raise ValueError('Pinned runtime changed')


if __name__=='__main__':
    import sys
    args=parser().parse_args();validate(args);validate_locked_cli(args,sys.argv[1:])
