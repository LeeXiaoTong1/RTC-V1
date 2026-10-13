"""Pin the reported 93.6341 V3.15 parent and coordinate the V3.19 recipe."""
import argparse
import math
from pathlib import Path
import sys
import torch
from w2v_v3151.config import source_configuration,verify_parent
from w2v_v315.config import code_fingerprints as historical_code
from w2v_v315.augment import bind_augmentation,runtime
from w2v_v39.common import ROOT,digest,verify_files
from w2v_v39.config import runtime_versions
from w2v_v3184.noise import bind_manifest
from .selection import DEFAULTS

PARENT_RUN='w2v_v315_20261008_075905_5663'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',help='Completed V3.15 run; defaults to the historical 93.6341 run, never latest')
    p.add_argument('--parent-checkpoint',choices=('best_guarded',),default=None)
    p.add_argument('--parent-tag',help='Optional additional assertion of the selected parent tag')
    p.add_argument('--resume');p.add_argument('--device',default='auto')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--microbatch',type=int,default=48,help='Autotune ceiling, not logical source count')
    p.add_argument('--frame-budget',type=int,default=24000)
    p.add_argument('--no-autotune',action='store_true')
    p.add_argument('--group-quotas',type=int,nargs=4,default=[4,3,5,4])
    p.add_argument('--view-factors',type=int,nargs=4,default=[6,6,1,3],help='TOTAL fixed versions, including original')
    p.add_argument('--noise-manifest',default=str(ROOT/'assets/noise_v3184_rnnoise/manifest.json'))
    p.add_argument('--scene-probability',type=float,default=1.)
    p.add_argument('--rtc-policy',choices=('three-profile','legacy'),default='three-profile')
    p.add_argument('--consistency',choices=('matched-bridge','matched','none'),default='matched-bridge')
    p.add_argument('--encoder-lr',type=float,default=1e-6)
    p.add_argument('--head-lr',type=float,default=1e-5)
    p.add_argument('--adapter-lr',type=float,default=2e-5)
    p.add_argument('--output-lr',type=float,default=2e-5)
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp')
    p.add_argument('--upload-temp',action='store_true',help='Upload report to temp.sh')
    p.add_argument('--submit-best',action='store_true')
    p.add_argument('--upload-submission',action='store_true')
    p.add_argument('--submission-checkpoint',choices=('best_weighted','best_guarded'),default=None)
    p.add_argument('--submission-protocol');p.add_argument('--submission-audio-root');p.add_argument('--submission-out')
    return p


def validate(args):
    if args.epochs<1 or args.workers<0 or args.microbatch<3 or args.frame_budget<3:
        raise ValueError('Positive epochs, physical batch >=3, positive frame budget, nonnegative workers required')
    if len(args.group_quotas)!=4 or min(args.group_quotas)<1 or sum(args.group_quotas)!=16:
        raise ValueError('Four positive source quotas must sum to 16')
    if args.view_factors!=[6,6,1,3]:raise ValueError('This release fixes version budgets to EN 6/6, ZH 1/3')
    if not math.isfinite(args.scene_probability) or not 0<=args.scene_probability<=1:
        raise ValueError('scene-probability must be finite and between zero and one')
    for name in ('encoder_lr','head_lr','adapter_lr','output_lr'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:raise ValueError('Invalid '+name)
    if args.upload_submission and not args.submit_best:raise ValueError('--upload-submission requires --submit-best')
    if not args.submit_best and any(getattr(args,n) for n in ('submission_protocol','submission_audio_root','submission_out')):
        raise ValueError('Submission paths require --submit-best')


def validate_locked_cli(args,argv):
    if not args.resume:return
    allowed={'--resume','--download-dir','--upload-temp','--submit-best','--upload-submission',
             '--submission-checkpoint','--submission-protocol','--submission-audio-root','--submission-out','--help'}
    extra={s.split('=')[0] for s in argv if s.startswith('--')}-allowed
    if extra:raise ValueError('Resume uses saved training parameters; remove overrides: '+', '.join(sorted(extra)))


def code_fingerprints():
    paths=historical_code()
    # These modules are reused as immutable implementation, not as Omni models.
    for folder in ('w2v_v3151','w2v_v318','w2v_v3183','w2v_v3183r2','w2v_v3184','w2v_v3185','w2v_v319'):
        paths.update({str(p.resolve()):digest(p) for p in (ROOT/folder).glob('*.py') if not p.name.startswith('test_')})
    catalog=ROOT/'w2v_v3184/noise_catalog.json';paths[str(catalog)]=digest(catalog)
    return paths


def configuration(args):
    validate(args)
    parent=Path(args.parent_run).expanduser().resolve() if args.parent_run else ROOT/'exp'/PARENT_RUN
    if not parent.is_dir():raise FileNotFoundError('Historical V3.15 parent missing: '+str(parent)+'; specify its relocated --parent-run')
    cfg=source_configuration(parent,'best_guarded')
    if cfg['parent_baseline_fallback']:raise ValueError('Expected trained V3.15 best_guarded, not its V3.12 fallback')
    if args.parent_tag and args.parent_tag!=cfg['parent_selected_tag']:raise ValueError('Parent selected tag differs')
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device!='cpu' and not device.startswith('cuda'):raise ValueError('Use cpu or cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    cfg.update(DEFAULTS)
    cfg.update(version='3.19',recipe='v319_received_tfcl_en6_v1',seed=31901,device=device,epochs=args.epochs,
        workers=args.workers,feature_workers=args.workers,feature_batch=16,eval_batch=16,
        microbatch=args.microbatch,frame_budget=args.frame_budget,source_batch=16,
        group_quotas=args.group_quotas,view_factors=args.view_factors,
        exposure_coverage_policy='all EN six fixed versions on both available streams within configured epochs',
        trainable_layers=8,checkpointing=True,encoder_lr=args.encoder_lr,head_lr=args.head_lr,
        adapter_lr=args.adapter_lr,output_lr=args.output_lr,tfcl_lr=0.,layer_decay=.8,
        lr_warmup_fraction=0.,objective_ramp_epochs=0.,warm_epochs=0,lr_floor=.1,
        weight_decay=.01,max_grad_norm=1.,amp='bf16' if device.startswith('cuda') else 'none',eval_amp='none',
        consistency=args.consistency,tfcl_time_weight=0. if args.consistency=='none' else .15,
        tfcl_structure_weight=0. if args.consistency=='none' else .045,
        tfcl_bridge_mass=.2 if args.consistency=='matched-bridge' else 0.,
        tfcl_matched_mass=.8 if args.consistency=='matched-bridge' else 1.,tfcl_heads=8,tfcl_bins=201,
        alignment_max_bins=192,alignment_local_band_fraction=.2,alignment_min_frames=4,alignment_temperature=.07,
        online_ce_mass=.3,reference_ce_mass=.1,initial_noisy_ce_mass=.6,main_noisy_ce_mass=.6,
        scene_probability=args.scene_probability,rtc_policy=args.rtc_policy,rtc_stress_probability=0.,rtc_stress_ramp_epochs=0.,
        raw_audio_cache_mib=128,noise_cache_mib=64,prefetch_factor=2,rolling_cache_bytes=0,
        minimum_epochs=args.epochs,patience=0,checks_per_epoch=1,progress_min_delta=.0002,
        catastrophic_weighted_drop=.05,matched_fake_recall=.99,threshold=.5,
        disk_margin_bytes=128*1024**2,free_reserve_bytes=10*1024**3,maximum_new_peak_bytes=20*1024**3,
        gpu_reserve_bytes=4*1024**3,autotune=not args.no_autotune,panel_seed=3190901,panel_sources_per_group=192,
        code_fingerprints=code_fingerprints(),runtime_versions=runtime_versions(),
        effective_objective='balanced CE + parameter-free matched received-audio and weak aligned official-Online consistency',
        effective_views='augmented official Online (or independent clean RTC fallback) / matched clean RTC / input-noise-before-RTC',
        source_exclusion_policy='keep missing Online sources; disable only their unavailable bridge',
        historical_progress=dict(weighted=93.6341,source='user report documented in README_W2V_V3151.md',run=PARENT_RUN,selector='best_guarded'),
        selection_policy='best_weighted uses historical fixed Dev; best_guarded additionally uses a held-out scene panel; parent fallback explicit',
        external_teacher_at_inference=False,generated_audio_disk_cache=False)
    cfg.update(bind_augmentation(cfg))
    cfg.update(bind_manifest(args.noise_manifest))
    verify_inputs(cfg)
    return cfg


def verify_inputs(cfg):
    if cfg.get('version')!='3.19' or cfg.get('recipe')!='v319_received_tfcl_en6_v1':raise ValueError('Expected V3.19 configuration')
    files={}
    for key in ('code_fingerprints','source_fingerprints','data_fingerprints','augmentation_files','scene_noise_files'):
        for path,sha in cfg[key].items():
            name=str(Path(path).resolve())
            if name in files and files[name]!=sha:raise ValueError('Conflicting input fingerprint: '+name)
            files[name]=sha
    for key in ('base_checkpoint','starting_checkpoint','parent_checkpoint'):
        path=str(Path(cfg[key]).resolve());sha=cfg[key+'_sha256']
        if path in files and files[path]!=sha:raise ValueError('Conflicting checkpoint fingerprint')
        files[path]=sha
    verify_files(files);verify_parent(cfg)
    if cfg['runtime_versions']!=runtime_versions():raise ValueError('Runtime changed; resume must preserve the recorded environment')
    if cfg['augmentation_runtime']!=runtime(cfg['augmentation_runtime'].get('ffmpeg_path')):
        raise ValueError('FFmpeg/WebRTC runtime changed')


if __name__=='__main__':
    args=parser().parse_args();validate(args);validate_locked_cli(args,sys.argv[1:])
