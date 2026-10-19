"""V4 new experiment: immutable parent weights, rebuilt data and compact priors."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import torch
from w2v_v31813.config import verify as verify_parent
from w2v_v31813.evaluate import selected
from w2v_v31813.phonemes import build_pairs
from w2v_v31810.identity import verify_registry
from .common import ROOT, read_json, atomic_json, atomic_save, fingerprint, digest, verify_files
from .source import resolve_parent, resolve_data_run
from .dataset import prepare as prepare_data
from . import VERSION, SCHEMA


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run');p.add_argument('--resume')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=4001);p.add_argument('--split-seed',type=int,default=4001)
    p.add_argument('--dev-real-train-fraction',type=float,default=.75)
    p.add_argument('--assets-root',default=str(ROOT/'assets/v4'))
    p.add_argument('--mode',choices=('full','baseline','random'),default='full')
    p.add_argument('--adapter-layers',type=int,choices=(0,8,16),default=16)
    p.add_argument('--adapter-dim',type=int,default=64);p.add_argument('--adapter-lr',type=float,default=1e-4)
    p.add_argument('--gradient-coordination',choices=('pcgrad','none'),default='pcgrad')
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--eval-workers',type=int,default=2);p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--frame-budget',type=int,default=6000);p.add_argument('--hf-endpoint')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--submit-best',action='store_true');p.add_argument('--upload-submission',action='store_true')
    p.add_argument('--upload-temp',action='store_true')
    root='/home/ubuntu/LXT/RTC/xlsr_aasist/dataset'
    p.add_argument('--submission-protocol',default=root+'/progress.txt')
    p.add_argument('--submission-audio-root',default=root+'/wav/progress')
    return p


def validate(args):
    if args.epochs<4 or not 2<=args.microbatch<=24 or args.frame_budget<6000 or min(args.workers,args.eval_workers)<0:
        raise ValueError('V4 needs >=4 epochs (1 CE + 2 auxiliary ramp + full-weight training), microbatch 2..24, frame budget >=6000')
    if not 0<=args.dev_real_train_fraction<1 or args.adapter_dim<4 or args.adapter_dim%4 or not 0<args.adapter_lr<=.01:
        raise ValueError('Invalid source split or adapter parameters')


def prepare_missing(args):
    validate(args)
    data,old=prepare_data(args)
    parent=resolve_parent(args.parent_run)
    phones=data/'phonemes/manifest.json';teacher=data/'xeus/manifest.json'
    if not phones.is_file():
        binding=read_json(parent/'original_phone_pairs.json')['owner']['manifest']
        previous=binding['path'];verify_files({previous:binding['sha256']})
        command=[sys.executable,'-u','-m','w2v_v4.prepare_phoneme_environment','--data-run',str(data),
            '--output-dir',str(phones.parent),'--device',args.device,'--reuse-manifest',previous]
        if args.hf_endpoint:command+=['--hf-endpoint',args.hf_endpoint]
        subprocess.run(command,check=True)
    if args.mode=='full' and not teacher.is_file():
        command=[sys.executable,'-u','-m','w2v_v4.prepare_teacher','--parent-run',str(data),
            '--output',str(teacher.parent),'--device',args.device]
        previous=old.get('teacher_binding')
        if previous:
            verify_files({previous['path']:previous['sha256']})
            command+=['--reuse-run',str(parent),'--reuse-manifest',previous['path']]
        if args.hf_endpoint:command+=['--hf-endpoint',args.hf_endpoint]
        subprocess.run(command,check=True)
    return data


def initialize(args,run,prepared):
    validate(args)
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production V4 requires BF16 CUDA; CPU fixtures are tests only')
    data=resolve_data_run(prepared);parent=resolve_parent(args.parent_run)
    old=verify_parent(parent);_,weights,done=selected(parent)
    staging=read_json(data/'config.json')
    if staging['data_owner']['parent_checkpoint']!=done['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed during data preparation; safely pause it first')
    cfg=copy.deepcopy(old);cfg.update({k:staging[k] for k in ('source_registry','validation_protocol_hash','validation_input_files')})
    run=Path(run);run.mkdir(parents=True,exist_ok=False)
    names=('train_rows.json','dev_rows.json','validation_protocol.json','source_registry.json','source_transfer.json')
    for name in names:atomic_json(run/name,read_json(data/name))
    rows=read_json(run/'train_rows.json');protocol=read_json(run/'validation_protocol.json')
    atomic_json(run/'parent_config.json',old);atomic_save(run/'init.pt',weights)
    phones=build_pairs(rows,data/'phonemes/manifest.json',run/'original_phone_pairs.json')
    from .prior import bind, random_neighbors
    prior=bind(data/'xeus/manifest.json',rows) if args.mode=='full' else None
    atomic_json(run/'xeus_neighbors.json',prior['neighbors'] if prior else random_neighbors(rows,args.seed) if args.mode=='random' else {})
    atomic_json(run/'prior_quality.json',prior['quality'] if prior else dict(mode=args.mode,xeus_used=False))
    parent_epoch=int(done['best_tag'].split('_')[-1]) if done['best_tag'].startswith('epoch_') else 0
    cfg.update(version=VERSION,recipe=SCHEMA,mode=args.mode,experiment='E0',epochs=args.epochs,warm_epochs=1,
        seed=args.seed,device=args.device,workers=args.workers,eval_workers=args.eval_workers,
        microbatch=args.microbatch,frame_budget=args.frame_budget,adapter_layers=args.adapter_layers,adapter_dim=args.adapter_dim,
        adapter_lr=args.adapter_lr,gradient_coordination=args.gradient_coordination,grad_clip=1.,
        source_epoch_offset=old.get('source_epoch_offset',0)+parent_epoch,meta_beta=0.,consistency='none',
        source_transfer=read_json(run/'source_transfer.json'),data_preparation=str(data),
        classification_objective='eight_cell_balanced_ce_with_two_stream_pcgrad',
        lora_trainable=False,lr_floor=.1,lr_warmup_updates=200,paired_probe_per_cell=2,
        initialization=dict(mode='warm_start',parent=str(parent),tag=done['best_tag'],
            parent_checkpoint_sha256=done['checkpoint_sha256'],optimizer_loaded=False,
            adapters='zero output projection',inherited_lora='enabled, frozen; no dropout'),
        learning_rate_origin=dict(policy='adapter 1e-4 default; fusion/head 0.5 times parent configured initial rates; fresh cosine and 200-update warmup',parent_epoch=parent_epoch),
        teacher_binding={k:v for k,v in prior.items() if k!='neighbors'} if prior else None,
        phoneme_pair_coverage=phones['coverage'],selection_protocol='V4 fixed retained sources and equal EN/ZH panel metrics')
    for key in ('head_lr','classifier_lr','evidence_lr','pool_lr','fusion_lr','layer_mix_lr'):
        cfg[key]=float(old[key])*.5
    from w2v_v31810.resources import audit_training
    from w2v_v31810.sampling import Plan
    cfg['window_resource_audit']=audit_training(rows,protocol['select']['rows']+protocol['audit']['rows'],cfg)
    largest=cfg['window_resource_audit']['maxima']['classification_frames']
    if 2*largest>cfg['frame_budget']:
        raise ValueError('XEUS/PCL pairs need a frame budget of at least '+str(2*largest)+'; full audio is never truncated')
    cfg['window_resource_audit']['auxiliary_policy']='Atomic XEUS pair and official original PCL pair; <=2*longest training view frames'
    cfg['window_resource_audit']['maximum_pair_padded_frames']=2*largest
    cfg['augmentation_budget_audit']=Plan(rows,cfg=cfg,seed=cfg['seed']).audit_budget(cfg['epochs'])
    cfg['saved_manifests']={name:digest(run/name) for name in (*names,'parent_config.json','init.pt','original_phone_pairs.json','xeus_neighbors.json','prior_quality.json')}
    cfg['version_code']={str(p.resolve()):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    atomic_json(run/'config.json',cfg)
    verify_registry(rows,read_json(run/'source_registry.json'),cfg,protocol)
    print(f'V4: {args.epochs} epochs; adapters last {args.adapter_layers}; frozen LoRA; {args.gradient_coordination}; full-length overlap ASP',flush=True)
    return cfg,rows,protocol


def verify(run,training=True):
    run=Path(run);cfg=read_json(run/'config.json')
    if cfg.get('version')!=VERSION or cfg.get('recipe')!=SCHEMA:raise ValueError('Not a V4 run')
    verify_files({str(run/name):sha for name,sha in cfg['saved_manifests'].items()})
    verify_files(cfg['version_code'])
    verify_parent(Path(cfg['initialization']['parent']),training=training)
    if training:
        verify_files(cfg['validation_input_files'])
        verify_registry(read_json(run/'train_rows.json'),read_json(run/'source_registry.json'),cfg,read_json(run/'validation_protocol.json'))
    return cfg
