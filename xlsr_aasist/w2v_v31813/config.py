"""New-run warm start, immutable parent model and unchanged source partitions."""
import argparse
import copy
from pathlib import Path
import torch
from w2v_v31810.common import (read_json,atomic_json,atomic_save,fingerprint,digest,verify_files)
from w2v_v31810.config import verify_inputs as verify_parent
from w2v_v31810.evaluate import selected
from w2v_v31810.identity import verify_registry
from .source import resolve_parent
from .phonemes import build_pairs
from .prior import bind
from . import VERSION,SCHEMA


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run');p.add_argument('--resume');p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--mode',choices=('full','baseline','random'),default='full')
    p.add_argument('--seed',type=int,default=3181301);p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--eval-workers',type=int,default=2)
    p.add_argument('--microbatch',type=int,default=4);p.add_argument('--frame-budget',type=int,default=6000)
    p.add_argument('--phoneme-manifest',default='assets/phonemes_v3186/manifest.json')
    p.add_argument('--xeus-manifest',default='assets/xeus_v31813/manifest.json')
    p.add_argument('--hf-endpoint',help='Optional Hugging Face endpoint, e.g. https://hf-mirror.com')
    p.add_argument('--submit-best',action='store_true');p.add_argument('--upload-submission',action='store_true')
    p.add_argument('--upload-temp',action='store_true')
    root='/home/ubuntu/LXT/RTC/xlsr_aasist/dataset'
    p.add_argument('--submission-protocol',default=root+'/progress.txt');p.add_argument('--submission-audio-root',default=root+'/wav/progress')
    return p


def prepare_missing(args):
    """Preparation subprocesses exit before the detector allocates GPU memory."""
    parent=resolve_parent(args.parent_run)
    # Reject an unavailable/incomplete parent before a costly teacher pass.
    old,weights,_=selected(parent,'best');del weights
    from w2v_v318.config import runtime_versions
    if old['runtime_versions']!=runtime_versions():raise ValueError('Activate the unchanged parent Omni runtime before preparation')
    if not Path(args.phoneme_manifest).exists():
        from .prepare_phoneme_environment import parser as phone_parser,execute
        command=['--data-run',str(parent),'--output-dir',str(Path(args.phoneme_manifest).resolve().parent),'--device',args.device]
        if args.hf_endpoint:command+=['--hf-endpoint',args.hf_endpoint]
        execute(phone_parser().parse_args(command))
    if args.mode=='full' and not Path(args.xeus_manifest).exists():
        import subprocess,sys
        command=[sys.executable,'-u','-m','w2v_v31813.prepare_teacher','--parent-run',str(parent),
            '--output',str(Path(args.xeus_manifest).resolve().parent),'--device',args.device]
        if args.hf_endpoint:command+=['--hf-endpoint',args.hf_endpoint]
        subprocess.run(command,check=True)


def initialize(args,run):
    if args.epochs<3 or not 2<=args.microbatch<=24 or args.frame_budget<6000 or min(args.workers,args.eval_workers)<0:
        raise ValueError('Need epochs >= 3, microbatch 2..24, frame_budget >= 6000 and nonnegative workers')
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production requires BF16 CUDA; local tests use explicit tiny model fixtures')
    parent=resolve_parent(args.parent_run);old,weights,meta=selected(parent,'best');verify_parent(old)
    files=('train_rows.json','dev_rows.json','validation_protocol.json','source_registry.json')
    verify_files({str(parent/name):old['saved_manifests'][name] for name in files})
    data={name:read_json(parent/name) for name in files}
    rows=data['train_rows.json'];protocol=data['validation_protocol.json']
    verify_registry(rows,data['source_registry.json'],old,protocol)
    cfg=copy.deepcopy(old)
    parent_epoch=next(e['epoch'] for e in torch.load(parent/'last.pt',map_location='cpu',weights_only=True)['history'] if e['tag']==meta['selected'])
    run=Path(run);run.mkdir(parents=True,exist_ok=False)
    for name,value in data.items():atomic_json(run/name,value)
    atomic_json(run/'parent_config.json',old)
    atomic_save(run/'init.pt',weights)
    phones=build_pairs(rows,Path(args.phoneme_manifest).resolve(),run/'original_phone_pairs.json')
    prior=bind(Path(args.xeus_manifest).resolve(),rows) if args.mode=='full' else None
    if args.mode=='random':
        from .prior import random_neighbors
        random_pairs=random_neighbors(rows,args.seed)
    else:random_pairs={}
    atomic_json(run/'xeus_neighbors.json',prior['neighbors'] if prior else random_pairs)
    atomic_json(run/'prior_quality.json',prior['quality'] if prior else dict(mode=args.mode,xeus_used=False))
    cfg.update(version=VERSION,recipe=SCHEMA,experiment='E0',mode=args.mode,epochs=args.epochs,warm_epochs=1,
        seed=args.seed,source_epoch_offset=parent_epoch,meta_beta=0.,consistency='none',
        classification_objective='eight_cell_balanced_ce',auxiliary_objective='coupled_xeus_margin_and_official_original_phoneme_mse',
        prior_margin=.5,grad_clip=1.,device=args.device,workers=args.workers,eval_workers=args.eval_workers,
        microbatch=args.microbatch,frame_budget=args.frame_budget,eval_batch=24,prefetch_factor=1,
        initialization=dict(mode='warm_start',parent=str(parent),tag=meta['selected'],parent_checkpoint_sha256=meta['checkpoint_sha256'],
            detector_weights_loaded=True,optimizer_loaded=False,source='V3.18.10 detector inspected by V3.18.11'),
        lr_floor=.1,lr_warmup_updates=200,
        learning_rate_origin=dict(policy='0.25 of original V3.18.10 initial rates; gentle restart, new optimizer/cosine schedule',parent_epoch=parent_epoch),
        auxiliary_protocol=dict(pcl='same-source official original Offline/Online only; both sides receive gradients',
            xeus='cross-source same-language opposite-label Train neighbors; teacher absent during training/inference',
            warmup_epochs=1,ramp_epochs=2,calibration='Train shared-fusion gradient norms; 10% prior and 5% PCL caps',
            reference_sources_per_group=1),teacher_binding={k:v for k,v in prior.items() if k!='neighbors'} if prior else None,
        phoneme_pair_coverage=phones['coverage'],inference_requires_phonemes=False,inference_requires_xeus=False)
    for key in ('lora_early_lr','lora_middle_lr','lora_lr','head_lr','classifier_lr','evidence_lr','pool_lr','fusion_lr','layer_mix_lr'):
        cfg[key]=float(old[key])*.25
    cfg['saved_manifests']={name:digest(run/name) for name in (*files,'parent_config.json','original_phone_pairs.json','xeus_neighbors.json','prior_quality.json','init.pt')}
    cfg['version_code']={str(p):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    atomic_json(run/'config.json',cfg)
    return cfg,rows,protocol


def verify(run,training=True):
    run=Path(run);cfg=read_json(run/'config.json')
    if cfg.get('recipe')!=SCHEMA or cfg.get('version')!=VERSION:raise ValueError('Not a V3.18.13 run')
    verify_files({str(run/name):sha for name,sha in cfg['saved_manifests'].items()})
    verify_files(cfg['version_code'])
    verify_parent(read_json(run/'parent_config.json'),training=training)
    if training:
        verify_registry(read_json(run/'train_rows.json'),read_json(run/'source_registry.json'),cfg,read_json(run/'validation_protocol.json'))
    return cfg
