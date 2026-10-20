"""Fresh V3.20 detector with V4 data/PCGrad and dual phoneme consistency."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import torch
from w2v_v31810.identity import verify_registry
from .common import ROOT, read_json, atomic_json, fingerprint, digest, verify_files
from .source import resolve_parent, resolve_data_run, verify_source
from .dataset import prepare as prepare_data
from .assets import prepare_assets, verify_assets
from .phonemes import build_pairs
from . import VERSION, SCHEMA


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-run','--parent-run',dest='parent_run',help='Reuse data only; no Omni detector weights')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--resume');mode.add_argument('--continue-from',help='Migrate a safely paused V4 last.pt, preserving its total epoch budget')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=4001);p.add_argument('--split-seed',type=int,default=4001)
    p.add_argument('--dev-real-train-fraction',type=float,default=.75)
    p.add_argument('--assets-root',default=str(ROOT/'assets/v4'))
    p.add_argument('--bert-model-dir',default='/home/ubuntu/LXT/v318-runtime/hf-w2v-bert-2.0')
    p.add_argument('--bert-revision');p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--reuse-phoneme-manifest');p.add_argument('--hf-endpoint')
    p.add_argument('--pcl',choices=('dual','official','none'),default='dual')
    p.add_argument('--pcl-gradient-target',type=float,default=.05,help='Per-branch Train calibration gradient ratio')
    p.add_argument('--pcl-max-weight',type=float,default=1.)
    p.add_argument('--pcl-ramp-updates',type=int,default=300)
    p.add_argument('--gradient-coordination',choices=('pcgrad','none'),default='pcgrad')
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int)
    p.add_argument('--eval-workers',type=int);p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--feature-cache-mib',type=int,default=64)
    p.add_argument('--checkpoint-policy',choices=('auto','fixed'),default='auto')
    p.add_argument('--checkpoint-layers',type=int,default=24)
    p.add_argument('--checkpoint-margin-mib',type=int,default=2048)
    p.add_argument('--frame-budget',type=int,default=9000);p.add_argument('--whole-max-frames',type=int,default=3000)
    p.add_argument('--backend-precision',choices=('fp16','fp32'),default='fp16')
    p.add_argument('--lora-lr',type=float,default=2e-5);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--submit-best',action='store_true');p.add_argument('--upload-submission',action='store_true')
    p.add_argument('--upload-temp',action='store_true')
    root='/home/ubuntu/LXT/RTC/xlsr_aasist/dataset'
    p.add_argument('--submission-protocol',default=root+'/progress.txt')
    p.add_argument('--submission-audio-root',default=root+'/wav/progress')
    return p


def validate(a):
    if a.workers is None:a.workers=4
    if a.eval_workers is None:a.eval_workers=2
    if not 0<=a.checkpoint_layers<=24 or a.feature_cache_mib<0 or a.checkpoint_margin_mib<512:
        raise ValueError('Invalid execution cache/checkpoint settings')
    if a.epochs<2 or not 1<=a.microbatch<=24 or a.frame_budget<1 or a.whole_max_frames<11 or min(a.workers,a.eval_workers)<0:
        raise ValueError('Use >=2 epochs, microbatch 1..24 and positive resource limits')
    if a.pcl!='none' and a.microbatch<3:
        raise ValueError('Original Online PCL needs capacity for one classification + two reference views (microbatch >=3)')
    if not 0<=a.dev_real_train_fraction<1 or not 0<a.pcl_gradient_target<=.5 or not 0<a.pcl_max_weight<=10 or a.pcl_ramp_updates<1:
        raise ValueError('Invalid source split or PCL calibration settings')
    if not 0<a.lora_lr<=.001 or not 0<a.head_lr<=.01:raise ValueError('Invalid learning rate')


def prepare_missing(args):
    validate(args)
    data,old=prepare_data(args)
    parent=resolve_parent(args.parent_run)
    if args.hf_endpoint:
        os.environ['HF_ENDPOINT']=args.hf_endpoint
    assets_file=data/'bert_assets.json'
    if assets_file.is_file():
        assets=read_json(assets_file);verify_assets(assets)
        if Path(args.bert_model_dir).resolve()!=Path(assets['bert_model_dir']) or args.bert_revision!=assets['bert_revision']:
            raise ValueError('Prepared official model selection changed; use a separate --assets-root')
    else:
        print('Preparing official w2v-BERT 2.0 assets (no detector or XEUS preparation)',flush=True)
        assets=prepare_assets(args.bert_model_dir,download=not args.local_files_only,revision=args.bert_revision)
        atomic_json(assets_file,assets)
    phones=data/'phonemes/manifest.json'
    if args.pcl!='none' and not phones.is_file():
        previous=args.reuse_phoneme_manifest
        if not previous and (parent/'original_phone_pairs.json').is_file():
            binding=read_json(parent/'original_phone_pairs.json')['owner']['manifest']
            if Path(binding['path']).is_file():
                verify_files({binding['path']:binding['sha256']});previous=binding['path']
        if not previous and old.get('phoneme_manifest') and Path(old['phoneme_manifest']).is_file():
            previous=old['phoneme_manifest']
        command=[sys.executable,'-u','-m','w2v_v41.prepare_phoneme_environment','--data-run',str(data),
            '--output-dir',str(phones.parent),'--device',args.device]
        if previous:command+=['--reuse-manifest',previous]
        if args.hf_endpoint:command+=['--hf-endpoint',args.hf_endpoint]
        if args.local_files_only:command+=['--local-files-only']
        subprocess.run(command,check=True)
    return data


def initialize(args,run,prepared):
    validate(args)
    if not args.device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Production V4 requires BF16 CUDA; CPU fixtures are tests only')
    import transformers
    if transformers.__version__!='4.38.2':raise RuntimeError('Activate the isolated setup_w2v_v41.sh environment (transformers 4.38.2)')
    data=resolve_data_run(prepared);parent=resolve_parent(args.parent_run)
    old=verify_source(parent);staging=read_json(data/'config.json')
    if staging['data_owner']['parent_config']!=digest(parent/'config.json'):
        raise ValueError('Source data configuration changed during preparation')
    cfg=copy.deepcopy(old)
    for key in list(cfg):
        if any(word in key for word in ('omni','xeus','teacher','adapter','window','pcl','phoneme')) or key in (
            'version_code','saved_manifests','initialization','learning_rate_origin','prior_margin','mode','local_evidence','pooling','hop'):
            cfg.pop(key,None)
    cfg.update({k:staging[k] for k in ('source_registry','validation_protocol_hash','validation_input_files')})
    cfg.update(read_json(data/'bert_assets.json'));verify_assets(cfg)
    run=Path(run);run.mkdir(parents=True,exist_ok=False)
    names=('train_rows.json','dev_rows.json','validation_protocol.json','source_registry.json','source_transfer.json')
    for name in names:atomic_json(run/name,read_json(data/name))
    rows=read_json(run/'train_rows.json');protocol=read_json(run/'validation_protocol.json')
    atomic_json(run/'data_source_config.json',old)
    if args.pcl!='none':
        phones=build_pairs(rows,data/'phonemes/manifest.json',run/'pcl_pairs.json')
    else:
        phones=dict(official={},noise={},coverage={},owner=None)
        atomic_json(run/'pcl_pairs.json',phones)
    cfg.update(version=VERSION,recipe=SCHEMA,architecture='w2v_bert_2_mhasp_mlp',experiment='E0',
        epochs=args.epochs,warm_epochs=1,seed=args.seed,device=args.device,workers=args.workers,eval_workers=args.eval_workers,
        microbatch=args.microbatch,frame_budget=args.frame_budget,whole_max_frames=args.whole_max_frames,
        backend_precision=args.backend_precision,gradient_coordination=args.gradient_coordination,grad_clip=1.,
        encoder_dim=1024,encoder_layers=24,lora_layers=24,lora_rank=16,lora_alpha=32.,lora_dropout=.05,
        checkpointing=True,checkpoint_layers=24,fusion_checkpointing=True,fusion_layers=list(range(1,25)),
        pooling_heads=4,mlp_dims=[1536,1024,512],classifier_dropout=.1,
        lora_lr=args.lora_lr,lora_early_lr=args.lora_lr*.25,lora_middle_lr=args.lora_lr*.5,
        head_lr=args.head_lr,pool_lr=args.head_lr,classifier_lr=args.head_lr,layer_mix_lr=args.head_lr,
        weight_decay=.01,label_smoothing=.02,lr_floor=.1,lr_warmup_updates=200,
        source_epoch_offset=0,meta_beta=0.,consistency=args.pcl,pcl_ramp_updates=args.pcl_ramp_updates,
        pcl_gradient_target=args.pcl_gradient_target,pcl_max_weight=args.pcl_max_weight,
        source_transfer=read_json(run/'source_transfer.json'),data_preparation=str(data),
        classification_objective='eight_cell_balanced_ce_with_two_stream_pcgrad',paired_probe_per_cell=0,
        initialization=dict(mode='official_pretrained_fresh_detector',data_source=str(parent),detector_weights_loaded=False,
            optimizer_loaded=False,base_model='facebook/w2v-bert-2.0'),
        learning_rate_origin=dict(policy='fresh 24-layer LoRA 0.25/0.5/1 rates; new MHASP/MLP; head warmup then cosine'),
        phoneme_pair_coverage=phones['coverage'],selection_protocol='V4 fixed retained sources and equal EN/ZH panel metrics')
    cfg.update(execution_settings(args))
    from .resources import audit_training
    from .sampling import Plan
    cfg['resource_audit']=audit_training(rows,protocol['select']['rows']+protocol['audit']['rows'],cfg)
    cfg['augmentation_budget_audit']=Plan(rows,cfg,phones).audit_budget(cfg['epochs'])
    cfg['saved_manifests']={name:digest(run/name) for name in (*names,'data_source_config.json','pcl_pairs.json')}
    # Freeze the actual reused augmentation/sampling/identity code as well as
    # this architecture. A resume must not silently change its data generator.
    packages=('w2v_v41','w2v_v4','w2v_v31810','w2v_v3189','w2v_v3188','w2v_v3187','w2v_v3186',
              'w2v_v3185','w2v_v3184','w2v_v3183r2','w2v_v318','w2v_v315')
    code={p.resolve() for package in packages for p in (ROOT/package).glob('*.py') if not p.name.startswith('test_')}
    code.add((ROOT/'w2v_v31813/diagnostics.py').resolve())
    cfg['version_code']={str(p):digest(p) for p in sorted(code)}
    atomic_json(run/'config.json',cfg)
    verify_registry(rows,read_json(run/'source_registry.json'),cfg,protocol)
    print(f'V4: w2v-BERT 24-layer LoRA / 1024 fusion / MHASP / MLP; {args.pcl} PCL; {args.gradient_coordination}; {args.epochs} epochs',flush=True)
    return cfg,rows,protocol


def verify(run,training=True):
    run=Path(run);cfg=read_json(run/'config.json')
    if cfg.get('version')!=VERSION or cfg.get('recipe')!=SCHEMA:
        raise ValueError('Not the new w2v-BERT V4 recipe; old Omni V4 runs cannot be resumed here')
    verify_files({str(run/name):sha for name,sha in cfg['saved_manifests'].items()})
    verify_files(cfg['version_code']);verify_assets(cfg)
    if training:
        verify_files(cfg['validation_input_files'])
        rows=read_json(run/'train_rows.json')
        verify_registry(rows,read_json(run/'source_registry.json'),cfg,read_json(run/'validation_protocol.json'))
        phones=read_json(run/'pcl_pairs.json')
        if phones['owner']:
            from w2v_v3187.phonemes import bind_manifest
            binding=bind_manifest(phones['owner']['manifest']['path'],rows)
            if binding!=phones['owner']['manifest']:raise ValueError('PCL metadata changed')
    return cfg


def execution_settings(args):
    if not 0<=args.checkpoint_layers<=24 or args.feature_cache_mib<0 or args.checkpoint_margin_mib<512:
        raise ValueError('Invalid execution cache/checkpoint settings')
    return dict(worker_features=True,feature_cache_mib=args.feature_cache_mib,
        checkpoint_policy=args.checkpoint_policy,checkpoint_layers=args.checkpoint_layers,
        checkpoint_margin_mib=args.checkpoint_margin_mib)
