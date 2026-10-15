"""Fresh Omni detector training with explicit source exposure budgets."""
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
from .source import resolve_data_run, metadata


def code_inventory(root=ROOT):
    """Follow repository Python imports; bind production dependencies only."""
    root=Path(root); pending=list((root/'w2v_v3187').glob('*.py')); seen={}
    def module_paths(parts):
        path=root.joinpath(*parts)
        return [path.with_suffix('.py'),path/'__init__.py']
    while pending:
        path=pending.pop()
        if path in seen or not path.is_file() or path.name.startswith('test_'):
            continue
        seen[path]=digest(path)
        package=list(path.relative_to(root).parts[:-1])
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):
                names=[alias.name.split('.') for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                target=prefix+(node.module.split('.') if node.module else [])
                names=[target]+[target+alias.name.split('.') for alias in node.names if alias.name!='*']
            for parts in names:
                for depth in range(1,len(parts)+1):
                    pending.extend(module_paths(parts[:depth]))
    return {str(path):value for path,value in seen.items()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-run', '--parent-run', dest='data_run', help='Reuse manifests/assets only; no detector checkpoint is loaded')
    p.add_argument('--noise-manifest', default=str(ROOT/'assets/noise_v3184_rnnoise/manifest.json'))
    p.add_argument('--phoneme-manifest', default=str(ROOT/'assets/phonemes_v3186/manifest.json'))
    p.add_argument('--consistency', choices=('dual','none'), default='dual')
    p.add_argument('--pcl-platform-weight', type=float, default=.045)
    p.add_argument('--pcl-noise-weight', type=float, default=.15)
    p.add_argument('--pcl-variance-weight', type=float, default=.01)
    p.add_argument('--pcl-variance-target', type=float, default=.5)
    p.add_argument('--pcl-max-phones-per-source', type=int, default=8)
    p.add_argument('--pcl-variance-min-sources', type=int, default=4)
    p.add_argument('--pcl-variance-min-phones', type=int, default=32)
    p.add_argument('--pcl-ramp-updates', type=int, default=300)
    p.add_argument('--platform-sources-per-group', type=int, default=3)
    p.add_argument('--gradient-audit-interval', type=int, default=200)
    p.add_argument('--scene-probability', type=float, default=1.)
    p.add_argument('--rtc-policy', choices=('legacy','three-profile'), default='three-profile')
    p.add_argument('--source-draws-per-stream', type=int, nargs=4, default=[9668,8000,8000,8000], metavar=('EN_FAKE','EN_REAL','ZH_FAKE','ZH_REAL'))
    p.add_argument('--refresh-base-variants', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--scene-panel-per-group', type=int, default=32)
    p.add_argument('--train-probe-per-group', type=int, default=128)
    p.add_argument('--diagnostic-interval', type=int, default=5, help='Extra RTC/scene panels every N epochs and final epoch; full Dev each epoch')
    p.add_argument('--resume')
    p.add_argument('--warm-start-run', help='New CE-only or PCL branch from a saved identical Train warmup')
    p.add_argument('--baseline-only','--preflight-only', dest='baseline_only', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--warm-epochs', type=int, default=1)
    p.add_argument('--microbatch', type=int, default=16, help='Encoded view limit including auxiliary references; logical CE batch is 72')
    p.add_argument('--frame-budget', type=int, default=12000)
    p.add_argument('--eval-batch', type=int, default=24)
    p.add_argument('--window-batch', type=int, default=32)
    p.add_argument('--checkpoint-layers', type=int, default=60)
    p.add_argument('--temporal-stats', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fusion-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--whole-max-frames', type=int, default=3000)
    p.add_argument('--rtc-stress-probability', type=float, default=.35)
    p.add_argument('--rtc-stress-ramp-epochs', type=float, default=2.)
    for name, value in (('lora-early',5e-6),('lora-middle',1e-5),('lora',2e-5),('head',1e-4),('fusion',3e-4),('layer-mix',1e-3),('classifier',1e-4),('evidence',1e-4),('pool',1e-4)):
        p.add_argument('--'+name+'-lr', type=float, default=value)
    p.add_argument('--lr-warmup-updates', type=int, default=200)
    p.add_argument('--lr-floor', type=float, default=.1)
    p.add_argument('--en-real-confidence-weight', type=float, default=0.)
    p.add_argument('--en-real-confidence-margin', type=float, default=3.5)
    p.add_argument('--download-dir', default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp', action='store_true')
    p.add_argument('--submit-best', action='store_true')
    p.add_argument('--upload-submission', action='store_true')
    p.add_argument('--submission-protocol')
    p.add_argument('--submission-audio-root')
    p.add_argument('--submission-out')
    return p


def validate(args):
    if args.epochs < 20 or not 0 <= args.warm_epochs < args.epochs:
        raise ValueError('V3.18.7 requires at least 20 epochs and fewer warm epochs')
    if args.workers < 0:
        raise ValueError('workers must be nonnegative')
    if args.resume and args.warm_start_run:
        raise ValueError('Choose resume or a new warm-start branch, not both')
    if args.consistency == 'dual' and args.warm_epochs < 1 and not args.resume:
        raise ValueError('Fixed-scale PCL requires at least one Train-only CE warmup epoch')
    for key, minimum in (('microbatch',4),('frame_budget',48),('eval_batch',1),('window_batch',1),
                         ('whole_max_frames',12),('pcl_ramp_updates',1),('platform_sources_per_group',1),
                         ('gradient_audit_interval',1),('lr_warmup_updates',1),('scene_panel_per_group',1),
                         ('train_probe_per_group',1),('diagnostic_interval',1),
                         ('pcl_max_phones_per_source',1),('pcl_variance_min_sources',2),('pcl_variance_min_phones',2)):
        if getattr(args,key) < minimum:
            raise ValueError('Invalid '+key)
    if not 0 <= args.checkpoint_layers <= 60:
        raise ValueError('checkpoint-layers must be within 0..60')
    for key in ('lora_early_lr','lora_middle_lr','lora_lr','head_lr','fusion_lr','layer_mix_lr','classifier_lr','evidence_lr','pool_lr','rtc_stress_ramp_epochs','en_real_confidence_margin','pcl_variance_target'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            raise ValueError('Invalid '+key)
    for key in ('pcl_platform_weight','pcl_noise_weight','pcl_variance_weight','en_real_confidence_weight'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) < 0:
            raise ValueError('Invalid '+key)
    if args.consistency == 'dual' and min(args.pcl_platform_weight,args.pcl_noise_weight) <= 0:
        raise ValueError('dual requires both positive PCL weights; use --consistency none for CE-only')
    if not 0 <= args.scene_probability <= 1 or not 0 <= args.rtc_stress_probability <= .5 or not 0 < args.lr_floor <= 1:
        raise ValueError('Invalid augmentation probability or LR floor')
    from .sampling import quota_schedule
    quota_schedule(tuple(args.source_draws_per_stream),24)
    if args.upload_submission and not args.submit_best:
        raise ValueError('--upload-submission requires --submit-best')
    if not args.submit_best and any(getattr(args,k) for k in ('submission_protocol','submission_audio_root','submission_out')):
        raise ValueError('Submission paths require --submit-best')


def validate_locked_cli(args, argv):
    if args.warm_start_run:
        allowed={'--warm-start-run','--consistency','--baseline-only','--preflight-only','--download-dir','--upload-temp',
                 '--submit-best','--upload-submission','--submission-protocol','--submission-audio-root','--submission-out','--help'}
        changed={a.split('=')[0] for a in argv if a.startswith('--')}-allowed
        if changed:
            raise ValueError('Warm-start control keeps the original recipe; remove overrides: '+', '.join(sorted(changed)))
    if not args.resume:
        return
    allowed={'--resume','--baseline-only','--preflight-only','--download-dir','--upload-temp','--help',
             '--submit-best','--upload-submission','--submission-protocol','--submission-audio-root','--submission-out'}
    changed={a.split('=')[0] for a in argv if a.startswith('--')}-allowed
    if changed:
        raise ValueError('Resume uses saved settings; remove overrides: '+', '.join(sorted(changed)))


def training_configuration(old,args,source):
    validate(args)
    cfg=copy.deepcopy(old)
    for key in ('warm_lr','warm_fusion_lr','warm_layer_mix_lr','warm_evidence_lr','view_factor','fusion_max_residual',
                'whole_graph_budget','whole_batch','whole_utterance_resource_audit','saved_manifests'):
        cfg.pop(key,None)
    excluded={'data_run','noise_manifest','phoneme_manifest','resume','warm_start_run','baseline_only','download_dir','upload_temp',
              'submit_best','upload_submission','submission_protocol','submission_audio_root','submission_out'}
    cfg.update({key:value for key,value in vars(args).items() if key not in excluded})
    cfg.update(version='3.18.7',recipe=RECIPE,pooling='asp',local_evidence=True,
        base_original_fraction=.2,stream_sources=24,group_quotas=[6,6,6,6],view_factors=[5,5,1,3],
        patience=0,fusion_dim=128,fusion_layers=list(range(1,old['encoder_layers']+1)),
        lora_layers=old['encoder_layers'],lora_rank=16,lora_alpha=32,lora_dropout=.05,
        checkpoint_layers=min(args.checkpoint_layers,old['encoder_layers']),
        normalization='group_layer',pcl_objective='fixed_scale_mse_variance_v2',
        pcl_scale_floor_fraction=.1,pcl_scale_min_floor=1e-4,
        data_source=source,initialization=dict(mode='pretrained',run=source['run'],files={},
            checkpoint_loaded=False,old_detector_loaded=False,old_optimizer_loaded=False,
            transfer='Official Omni SSL weights only',
            fresh='All-layer LoRA, fusion, GN/LN AASIST, ASP, temporal statistics and complete classifier'),
        bn_policy='No BatchNorm; per-window GroupNorm and per-node LayerNorm',
        label_smoothing=.02,inference_requires_phonemes=False,phoneme_manifest=None,phoneme_files={},
        short_input_policy='reflect_to_3920_original_region_pcl_v1',checkpointing=True,
        runtime_versions=runtime_versions())
    # Earlier recipes recorded ancestry checkpoints as inputs. Fresh training
    # needs only current manifests, never a historical last.pt or its optimizer.
    cfg['data_files']=dict(source['files'])
    cfg['code_fingerprints']={}
    if args.consistency=='none':
        cfg.update(pcl_platform_weight=0.,pcl_noise_weight=0.,pcl_variance_weight=0.)
    schema_for(cfg)
    return cfg


def configuration(args):
    if args.warm_start_run:
        return warm_start_configuration(args)
    source=resolve_data_run(args.data_run)
    old,provenance=metadata(source)
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production training requires BF16 CUDA; CPU fixtures are tests only')
    cfg=training_configuration(old,args,provenance)
    from w2v_v3184.noise import bind_manifest as bind_noise
    cfg.update(bind_noise(args.noise_manifest))
    train=read_json(source/'train_rows.json');dev=read_json(source/'dev_rows.json')
    from .resources import audit_training
    cfg['window_resource_audit']=audit_training(train,dev,cfg)
    if args.consistency=='dual':
        from .phonemes import bind_manifest
        cfg['phoneme_provenance']=bind_manifest(args.phoneme_manifest,train)
        cfg['phoneme_manifest']=str(Path(args.phoneme_manifest).resolve())
        cfg['phoneme_files']={cfg['phoneme_manifest']:digest(cfg['phoneme_manifest'])}
    cfg['code_fingerprints']=code_inventory()
    from .sampling import Plan
    plan=Plan(train,24,cfg['seed'],cfg['group_quotas'],cfg['view_factors'],cfg['refresh_base_variants'],cfg=cfg)
    cfg['augmentation_budget_audit']=plan.audit_budget(cfg['epochs'])
    print('[Startup] verifying manifests, prepared noise/phonemes and official Omni weights',flush=True)
    verify_inputs(cfg)
    return cfg,train,dev


def warm_start_configuration(args):
    """Branch only the auxiliary objective; replay the identical saved warmup."""
    from .common import verify_files, fingerprint
    source=Path(args.warm_start_run).expanduser().resolve()
    old=read_json(source/'config.json');schema_for(old)
    verify_inputs(old)
    verify_files({str(source/name):value for name,value in old['saved_manifests'].items()})
    checkpoint=source/'warmup.pt'
    if not checkpoint.is_file():
        raise ValueError('Warmup snapshot not ready; finish the first CE warmup epoch before branching')
    cfg=copy.deepcopy(old);cfg.pop('saved_manifests',None);cfg.pop('warm_start',None)
    cfg['consistency']=args.consistency
    if args.consistency=='none':
        cfg.update(pcl_platform_weight=0.,pcl_noise_weight=0.,pcl_variance_weight=0.)
    elif old['consistency']!='dual':
        raise ValueError('A PCL branch requires the calibrated dual-recipe warmup snapshot')
    cfg['warm_start']=dict(run=str(source),checkpoint=str(checkpoint),sha256=digest(checkpoint),parent_identity=fingerprint(old))
    cfg['initialization']=dict(mode='shared_warmup',run=str(source),
        files={str(checkpoint):cfg['warm_start']['sha256']},checkpoint_loaded=True,
        old_detector_loaded=True,old_optimizer_loaded=True,
        transfer='Identical V3.18.7 v2 CE warmup weights, optimizer, PCL scale and RNG',
        official_backbone_origin=True)
    return cfg,read_json(source/'train_rows.json'),read_json(source/'dev_rows.json')


def verify_inputs(cfg, *, training=True):
    schema_for(cfg);configured_spec(cfg)
    verify_unique(cfg['code_fingerprints'],
        {cfg['omni_checkpoint']:cfg['omni_sha256']},
        cfg['data_files'] if training else {}, cfg['augmentation_files'] if training else {},
        {cfg['ffmpeg']:cfg['ffmpeg_sha256']} if training else {},
        cfg['scene_noise_files'] if training else {},cfg.get('phoneme_files',{}) if training else {})
    if cfg['runtime_versions']!=runtime_versions():
        raise ValueError('Pinned runtime changed')
    if training and cfg['consistency']=='dual':
        from .phonemes import bind_manifest
        rows=read_json(Path(cfg['data_source']['run'])/'train_rows.json')
        if bind_manifest(cfg['phoneme_manifest'],rows)!=cfg['phoneme_provenance']:
            raise ValueError('Phoneme inventory changed')


if __name__=='__main__':
    import sys
    args=parser().parse_args()
    validate(args)
    validate_locked_cli(args,sys.argv[1:])
