"""Prepare two full noisy views, retire unused caches, then start one new training run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .launch import BASELINE, BASELINE_SHA256, ROOT, configuration, find_source, parser as train_parser, run_lock
from .runtime import atomic_json, sha256
from .full_cache import prepare


def ensure_idle():
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Run the production workflow on the Linux training server')
    active = []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if int(entry.name) == os.getpid() or entry.stat().st_uid != os.getuid(): continue
            args = (entry/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if not args or not Path(args[0]).name.startswith('python'): continue
            # Training, evaluation, audit or another cache writer can still depend on retired files.
            if any(any(tag in a.lower() for tag in ('w2v_', 'rtc_noisy', 'main_train')) for a in args[1:]):
                active.append({'pid':int(entry.name),'command':' '.join(args)})
        except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    if active: raise RuntimeError('Finish/stop the existing job before changing caches: '+json.dumps(active))


def find_ffmpeg(source, explicit=None):
    from rtc_noisy.simulator import LocalRTC
    expected = json.loads((Path(source['dev_noisy_cache'])/'config.json').read_text(encoding='utf-8'))['ffmpeg_version']
    candidates = [explicit] if explicit else [os.environ.get('FFMPEG_BIN'),source.get('ffmpeg'), shutil.which('ffmpeg'),
        '/home/ubuntu/LXT/RTC/tools/ffmpeg61/ffmpeg','/home/ubuntu/LXT/RTC/tools/ffmpeg61/bin/ffmpeg']
    if not explicit:
        tool_root=Path('/home/ubuntu/LXT/RTC/tools/ffmpeg61')
        if tool_root.is_dir(): candidates.extend(str(p) for p in tool_root.rglob('ffmpeg') if p.is_file())
    for candidate in candidates:
        if not candidate: continue
        try:
            engine = LocalRTC(candidate)
            if engine.version == expected: return engine.ffmpeg
        except (OSError,RuntimeError): pass
    raise RuntimeError('Pass --ffmpeg with the original executable matching Dev: '+expected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config')
    p.add_argument('--ffmpeg')
    p.add_argument('--noise-manifest')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--prepare-only',action='store_true',help='Prepare and clean caches; start training separately')
    p.add_argument('--upload-temp',action='store_true')
    args=p.parse_args()
    if args.workers < 1: p.error('workers must be positive')
    ensure_idle()
    with run_lock(ROOT/'exp'/'.full2-workflow.lock'):
        with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
            ensure_idle()
            source_path,source=find_source(ROOT/'exp',args.source_config)
            print('Verifying the protected original 91.68 best...',flush=True)
            if sha256(BASELINE) != BASELINE_SHA256:
                raise RuntimeError('Protected original best differs; no cache was changed')
            output=ROOT/'data'/'rtc_noisy_full2_v1'/'train'
            manifest=args.noise_manifest or source.get('train_noise_manifest') or '/home/ubuntu/LXT/RTC/external_noise/noise_split/train.jsonl'
            print('SOURCE_CONFIG='+str(source_path),flush=True)
            ffmpeg=find_ffmpeg(source,args.ffmpeg)
            print('FFMPEG='+ffmpeg,flush=True)
            prepare(source,output,manifest,ffmpeg,args.workers)
            training_args=train_parser().parse_args(['--full-noisy-cache',str(output)])
            cfg=configuration(source,training_args)
            # Exercise the actual training reader and fixed Dev role/noise isolation checks before deleting anything.
            from .data import build_data
            plan,validation,_,_,fingerprints=build_data(cfg)
            if not plan.full or len(plan.banks)!=1: raise RuntimeError('New recipe still depends on a legacy Train cache')
            print('FULL_CACHE_TRAINING_READER_VALIDATED=True',flush=True)
            del plan,validation
            saved=ROOT/'exp'/'full2_source_config.json'
            atomic_json(saved,source)
            atomic_json(ROOT/'exp'/'full2_training_config.json',cfg)
            from .cache_retirement import retire
            retire(source,cfg,ROOT,ensure_idle)
            if sha256(BASELINE)!=BASELINE_SHA256 or any(sha256(path)!=digest for path,digest in fingerprints.items()):
                raise RuntimeError('Protected weights or active cache metadata changed')
            print('ORIGINAL_BEST_PRESERVED=True FIXED_DEV_CACHES_PRESERVED=True',flush=True)
            print(f'FREE_GiB={shutil.disk_usage(ROOT).free/1024**3:.2f}',flush=True)
        if not args.prepare_only:
            ensure_idle()
            # exec keeps the same PID, avoiding a hidden duplicate launcher while preserving the outer log.
            command=[sys.executable,'-u','-m','w2v_aasist.launch','--run','--source-config',str(saved),
                     '--full-noisy-cache',str(output)]
            if args.upload_temp: command.append('--upload-temp')
            os.execv(sys.executable,command)
        print('PREPARATION_COMPLETE=True',flush=True)


if __name__=='__main__': main()
