"""Independent V3.18.2 contract: two warmup + eight joint epochs by default."""
import argparse
from importlib.metadata import version
import math
import sys
from pathlib import Path
import torch
from .common import ROOT,atomic_json,read_json,digest,verify_files
from .records import data_inputs,inherited
from .assets import DEFAULT_ARCH,MODELS,spec,default_assets,validate_assets,configured_spec


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-run');p.add_argument('--resume')
    p.add_argument('--omni-size',choices=tuple(MODELS),default=DEFAULT_ARCH)
    p.add_argument('--omni-assets',help='Matching prepared assets; default follows --omni-size')
    p.add_argument('--dev-pairs');p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--warm-epochs',type=int,default=2)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--microbatch',type=int)
    p.add_argument('--frame-budget',type=int);p.add_argument('--eval-batch',type=int)
    p.add_argument('--stream-sources',type=int,default=16)
    p.add_argument('--train-probe-per-group',type=int,default=512)
    p.add_argument('--panel-per-group',type=int,default=64)
    p.add_argument('--lora-layers',type=int,default=16);p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--lora-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=3e-5)
    p.add_argument('--warm-lr',type=float,default=1e-4);p.add_argument('--evidence-lr',type=float,default=1e-4)
    p.add_argument('--label-smoothing',type=float,default=.02)
    p.add_argument('--patience',type=int,default=0,help='Default 0 runs all 10 epochs; positive values enable joint-phase early stopping')
    p.add_argument('--variant',choices=('C1',),default='C1',help='Fixed local AASIST; two-noisy risk disabled')
    p.add_argument('--source-sampling-power',type=float,default=.5)
    p.add_argument('--contrastive-weight',type=float,default=.05)
    p.add_argument('--contrastive-temperature',type=float,default=.1)
    p.add_argument('--contrastive-ramp-epochs',type=float,default=1.)
    p.add_argument('--ce-only',action='store_true',help='A control: identical three views, no pair loss')
    p.add_argument('--warmup-only',action='store_true',help='Save shared warmup and stop before joint training')
    p.add_argument('--init-warmup',help='New A/B run from the immutable warmup of this V3.18.2 run')
    p.add_argument('--seed',type=int,default=31801)
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp');p.add_argument('--upload-temp',action='store_true')
    return p


def validate(args):
    if (not 0 <= args.source_sampling_power <= 1 or not math.isfinite(args.contrastive_weight)
            or args.contrastive_weight < 0 or not math.isfinite(args.contrastive_temperature)
            or args.contrastive_temperature <= 0 or not math.isfinite(args.contrastive_ramp_epochs)
            or args.contrastive_ramp_epochs <= 0 or args.warm_epochs < 1):
        raise ValueError('Invalid source sampling, pair loss or warmup setting')
    if args.init_warmup and (args.resume or args.warmup_only):
        raise ValueError('--init-warmup starts a new joint run; incompatible with --resume/--warmup-only')
    item=spec(args.omni_size)
    for name in ('microbatch','frame_budget','eval_batch'):
        if getattr(args,name) is None:setattr(args,name,item[name])
    if (args.epochs<1 or not 0<=args.warm_epochs<args.epochs or args.workers<0 or args.microbatch<2 or
        args.frame_budget<12 or args.stream_sources<4 or args.stream_sources%4 or args.train_probe_per_group<1 or
        args.panel_per_group<1 or not 1<=args.lora_layers<=item['encoder_layers'] or not 1<=args.lora_rank<=128 or
        not 0<=args.label_smoothing<.2 or args.patience<0 or args.eval_batch<1):
        raise ValueError('Invalid epoch, sampling, worker, LoRA or regularization setting')
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k)<=0 for k in ('lora_lr','head_lr','warm_lr','evidence_lr')):
        raise ValueError('Learning rates must be finite and positive')


def runtime_versions():
    return {k:version(k) for k in ('torch','numpy','soundfile','scipy','fairseq2','fairseq2n','omnilingual-asr','webrtc-audio-processing','imageio-ffmpeg')}


def validate_locked_cli(args, argv):
    """Never silently ignore an attempted training override during resume/fork."""
    if args.resume:
        allowed={'--resume','--download-dir','--upload-temp','--warmup-only'}
    elif args.init_warmup:
        allowed={'--init-warmup','--ce-only','--contrastive-weight','--download-dir','--upload-temp'}
    else:
        return
    provided={word.split('=',1)[0] for word in argv if word.startswith('--')}
    unexpected=provided-allowed
    if unexpected:
        raise ValueError('Saved training configuration is locked; unsupported overrides: '+', '.join(sorted(unexpected)))


def configuration(args):
    validate(args)
    source=args.data_run
    if not source:
        for name in ('.latest_v318_run','.latest_v317_run','.latest_v316_tfcl_run'):
            p=ROOT/'exp'/name
            if p.is_file():source=p.read_text().strip();break
    if not source:raise ValueError('Specify --data-run for an existing V3.15–V3.18 run')
    old,train,dev,files=data_inputs(source)
    assets_path=Path(args.omni_assets or default_assets(args.omni_size)).expanduser().resolve();assets=read_json(assets_path)
    item=validate_assets(assets,args.omni_size)
    pair=args.dev_pairs
    if not pair:
        candidates=[p for p in old.get('data_fingerprints',{}) if Path(p).suffix=='.csv' and 'dev' in Path(p).name.lower() and 'pair' in Path(p).name.lower()]
        root=Path(inherited(old,'dev_data_path'))
        candidates += [str(parent/'meta'/'dev_offline_online_pairs.csv') for parent in (root,*root.parents)]
        pair=next((p for p in candidates if Path(p).is_file()),None)
    if not pair:raise FileNotFoundError('Specify --dev-pairs path/to/dev_offline_online_pairs.csv for source-isolated calibration')
    pair=str(Path(pair).resolve());files[pair]=digest(pair);files[str(assets_path)]=digest(assets_path)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Production Omni training requires CUDA; unit tests use a small CPU encoder')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 support required by the pinned Omni execution')
    from .augment import augmentation_runtime
    processing=augmentation_runtime()
    from .runtime import execution_profile,native_abi
    runtime_profile=execution_profile();abi=native_abi(runtime_profile)
    print('V3182_FFMPEG='+processing['ffmpeg_path']+'; '+processing['ffmpeg'],flush=True)
    cfg=dict(version='3.18.2',variant=args.variant,data_run=str(Path(source).resolve()),data_files=files,
        omni_checkpoint=assets['checkpoint'],omni_sha256=assets['sha256'],omni_provenance=assets,
        omni_arch=args.omni_size,encoder_dim=item['encoder_dim'],encoder_layers=item['encoder_layers'],
        feature_layer='final_only',waveform_normalization='full-wave layer_norm',
        epochs=args.epochs,warm_epochs=args.warm_epochs,workers=args.workers,device=args.device,seed=args.seed,
        microbatch=args.microbatch,frame_budget=args.frame_budget,eval_batch=args.eval_batch,stream_sources=args.stream_sources,
        train_probe_per_group=args.train_probe_per_group,panel_per_group=args.panel_per_group,
        lora_layers=args.lora_layers,lora_rank=args.lora_rank,lora_alpha=2.*args.lora_rank,lora_dropout=.05,
        lora_lr=args.lora_lr,head_lr=args.head_lr,warm_lr=args.warm_lr,evidence_lr=args.evidence_lr,
        local_evidence=True,risk_coefficient=0.,
        source_sampling_power=args.source_sampling_power,
        contrastive_weight=0. if args.ce_only else args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,contrastive_ramp_epochs=args.contrastive_ramp_epochs,
        negative_mass=4.,view_ce_weights=[.1,.45,.45],group_loss_mass=.25,
        contrastive_scope='whole logical source batch; symmetric Offline/processed pairs; opposite-label negatives only',
        negative_policy='positive view slot; equal language mass; total negative mass=4',
        gradient_replay='two passes, exact RNG replay, frozen BN in joint phase',
        window=128,hop=64,window_batch=32,label_smoothing=args.label_smoothing,weight_decay=.01,max_grad_norm=1.,
        bn_policy='warmup running stats; frozen running stats with trainable affine in joint phase',
        patience=args.patience,min_joint_epochs=3,min_delta=.0002,calibration_fraction=.2,dev_pairs=pair,
        selection='raw weighted F1 on source-separated Dev select; full Dev historical reporting only; no old-model fallback',
        checkpointing=True,amp='bf16',raw_audio_cache_mib=128,noise_cache_mib=64,
        free_reserve_bytes=10*1024**3,disk_margin_bytes=128*1024**2,
        noise_records=old['noise_records'],augmentation_files=old['augmentation_files'],
        ffmpeg=processing['ffmpeg_path'],ffmpeg_sha256=processing['ffmpeg_sha256'],augmentation_runtime=processing,
        official_dev_protocol=inherited(old,'dev_protocol'),official_dev_root=inherited(old,'dev_data_path'),
        initialization=dict(mode='public_omni_w2v'+args.omni_size+'_fresh_lora_and_ssl_aasist',old_detector_loaded=False,old_optimizer_loaded=False),
        runtime_versions=runtime_versions(),execution_profile=runtime_profile,native_build=abi)
    # Pin only code actually used by V3.18.2, not code of historical model producers.
    code=list((ROOT/'w2v_v3182').glob('*.py'))+list((ROOT/'w2v_v318').glob('*.py'))
    for folder in ('w2v_v317_monitor','w2v_v316_tfcl','w2v_v39','w2v_v36','w2v_v315','w2v_v313','w2v_aasist','rtc_noisy','utils'):
        names={'w2v_v317_monitor':('metrics.py','render.py','history.py'), 'w2v_v316_tfcl':('metrics.py',),
            'w2v_v39':('common.py','metrics.py'), 'w2v_v36':('metrics.py',), 'w2v_v315':('augment.py',),
            'w2v_v313':('state.py',), 'w2v_aasist':('data.py','runtime.py','composition.py','launch.py','full_workflow.py','evaluate.py'),
            'rtc_noisy':('simulator.py',),'utils':('env_noise.py',)}[folder]
        code.extend(ROOT/folder/n for n in names)
    cfg['code_fingerprints']={str(p):digest(p) for p in code if not p.name.startswith('test_')}
    verify_inputs(cfg)
    return cfg,train,dev


def verify_inputs(cfg):
    if cfg.get('version')!='3.18.2':raise ValueError('Not a V3.18.2 configuration')
    configured_spec(cfg)
    verify_files(cfg['data_files']);verify_files(cfg['code_fingerprints']);verify_files(cfg['augmentation_files'])
    verify_files({cfg['omni_checkpoint']:cfg['omni_sha256'],cfg['ffmpeg']:cfg['ffmpeg_sha256']})
    if cfg['runtime_versions']!=runtime_versions():raise ValueError('V3.18.2 runtime versions changed')


if __name__=='__main__':
    arguments=parser().parse_args();validate(arguments);validate_locked_cli(arguments,sys.argv[1:])
