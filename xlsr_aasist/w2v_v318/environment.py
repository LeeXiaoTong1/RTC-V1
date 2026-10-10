"""Independent startup checks; collect every failure before returning a report."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def check_versions():
    from .runtime import require_version
    errors=[];actual={}
    for line in (ROOT/'requirements_w2v_v318.txt').read_text(encoding='utf8').splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        name,wanted=line.split('==',1)
        try:actual[name]=require_version(name,wanted)
        except Exception as exc:errors.append(str(exc))
    print(json.dumps(actual,sort_keys=True))
    if errors:raise RuntimeError('\n'.join(errors))


def check_scientific():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pandas as pd
    import polars as pl
    import numba
    import llvmlite
    import librosa
    from scipy.signal import butter,sosfilt
    from sklearn.metrics import roc_auc_score
    x=np.arange(16,dtype=np.float32);table=pa.table({'x':x})
    sink=pa.BufferOutputStream();pq.write_table(table,sink)
    loaded=pq.read_table(pa.BufferReader(sink.getvalue()))
    if not np.array_equal(loaded.to_pandas()['x'].to_numpy(),x):raise RuntimeError('Arrow/Pandas/NumPy roundtrip failed')
    if pl.from_arrow(loaded)['x'].sum()!=120 or pc.sum(loaded['x']).as_py()!=120:raise RuntimeError('Arrow/Polars compute failed')
    @numba.njit
    def total(values):return values.sum()
    if total(x)!=120:raise RuntimeError('Numba JIT failed')
    resampled=librosa.resample(x,orig_sr=8000,target_sr=16000)
    filtered=sosfilt(butter(2,2000,fs=16000,output='sos'),resampled)
    if len(filtered)!=32 or not np.isfinite(filtered).all():raise RuntimeError('Audio resampling/filter failed')
    if roc_auc_score([0,1],[.1,.9])!=1.:raise RuntimeError('Metric backend failed')
    print('Scientific imports, parquet roundtrip, JIT, resampling and metrics passed')


def check_native():
    import ctypes
    import platform
    if platform.system()!='Linux':raise RuntimeError('Production V3.18 requires Linux')
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Use Python 3.11 in sdd-v318')
    prefix=Path(os.environ['CONDA_PREFIX']).resolve()
    if Path(sys.prefix).resolve()!=prefix:raise RuntimeError('Python and active Conda environment differ')
    for name in ('libsndfile.so.1','libtbb.so.12'):
        ctypes.CDLL(str(prefix/'lib'/name),mode=ctypes.RTLD_GLOBAL)
    print('Conda libsndfile and oneTBB loaded')


def cuda_diagnostics(torch,runner=subprocess.run):
    """Separate wheel, driver/visibility and BF16 errors without changing devices."""
    data=dict(python=sys.executable,torch=str(torch.__version__),torch_cuda=torch.version.cuda,
        visibility={name:os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES')})
    try:
        result=runner(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
            capture_output=True,text=True,timeout=15)
        data['nvidia_smi']=dict(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
    except Exception as exc:data['nvidia_smi']=dict(error=str(exc))
    try:
        data['cuda_available']=torch.cuda.is_available();data['visible_device_count']=torch.cuda.device_count()
        if not data['cuda_available']:
            # is_available() reduces initialization failures to False. Preserve
            # the driver's actual error here instead of blaming BF16 hardware.
            try:torch.cuda.init()
            except Exception as exc:data['initialization_error']=str(exc)
            data['failure']='CUDA unavailable: '+data.get('initialization_error','no visible CUDA device')
        else:
            data['device']=torch.cuda.get_device_name()
            data['capability']=list(torch.cuda.get_device_capability())
            data['bf16_supported']=torch.cuda.is_bf16_supported()
            if not data['bf16_supported']:data['failure']='CUDA is available but BF16 is unsupported on the selected device: '+data['device']
    except Exception as exc:data['failure']='CUDA initialization/device query failed: '+str(exc)
    if torch.version.cuda!='12.6':data['failure']='Expected CUDA 12.6 Torch wheel; installed CUDA runtime='+str(torch.version.cuda)
    return data


def check_cuda():
    import torch
    import torchaudio
    import fairseq2n  # This executes the upstream Torch/CUDA ABI checks.
    status=cuda_diagnostics(torch)
    print('V318_CUDA_DIAGNOSTICS='+json.dumps(status,ensure_ascii=False),flush=True)
    if 'failure' in status:raise RuntimeError(status['failure'])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        x=torch.randn(32,32,device='cuda',requires_grad=True);product=x@x
        if product.dtype!=torch.bfloat16:raise RuntimeError('CUDA autocast did not execute BF16 matrix multiplication')
        y=product.float().square().mean()
    y.backward();torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all():raise RuntimeError('CUDA gradient invalid')
    print('CUDA/BF16 and native ABI passed: '+torch.cuda.get_device_name())


def check_omni():
    import omnilingual_asr
    import kenlm
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from .assets import spec
    hub=get_wav2vec2_model_hub()
    for name in ('1b','3b'):
        cfg=hub.get_arch_config(name).encoder_config;item=spec(name)
        if (cfg.model_dim,cfg.num_encoder_layers)!=(item['encoder_dim'],item['encoder_layers']):raise RuntimeError('Wrong '+name+' SSL architecture')
    print('Omni full import and official 1B/3B registration passed')


def check_augmentation(family):
    import numpy as np
    from .augment import Engines,recipe,PANEL_FAMILIES
    engine=Engines()
    print('V318_FFMPEG='+engine.rtc.ffmpeg+'; '+engine.rtc.version,flush=True)
    # Include a non-10-ms-aligned length to check WebRTC padding/codec trimming.
    for length in (16000,16123):
        wave=(.08*np.sin(np.arange(length)*2*np.pi*240/16000)).astype(np.float32)
        for codec in (('none',) if family in PANEL_FAMILIES else ('none','opus')):
            r=recipe(31801,'environment','synthetic',0,0,family=family);r['codec']=codec
            y=engine(wave,r)
            if y.shape!=wave.shape or not np.isfinite(y).all():raise RuntimeError(f'Invalid {family}/{codec}/{length}')
    print(family+' full duration and finite output passed')


def plans():
    stages=['versions','scientific','native','cuda','omni']
    stages+=['augment:'+name for name in ('ffmpeg','webrtc','light','bypass','g711_mulaw','anlmdn')]
    stages+=['interface']
    return [(name,[sys.executable,'-m','w2v_v318.environment','--stage',name]) for name in stages]+[
        ('pip-check',[sys.executable,'-m','pip','check'])]


def run_checks(checks,report,runner=subprocess.run,timeout=600):
    results=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    for name,command in checks:
        print('[Check] '+name,flush=True);start=time.monotonic()
        try:
            r=runner(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=timeout)
            result=dict(name=name,ok=r.returncode==0,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except Exception as exc:
            result=dict(name=name,ok=False,returncode=None,stdout='',stderr=repr(exc))
        result['seconds']=round(time.monotonic()-start,3);results.append(result)
        print('  '+('PASS' if result['ok'] else 'FAIL'),flush=True)
    data=dict(created=datetime.now().isoformat(),python=sys.executable,checks=results,ok=all(r['ok'] for r in results))
    path=Path(report);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print('V318_ENV_REPORT='+str(path.resolve()),flush=True)
    for r in results:
        if not r['ok']:
            lines=(r['stderr'] or r['stdout']).strip().splitlines()
            print('[FAIL] '+r['name']+': '+(lines[-1][-400:] if lines else 'no output'),flush=True)
            if r['name']=='cuda' and r['stdout'].strip():print(r['stdout'].strip(),flush=True)
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',help=argparse.SUPPRESS)
    p.add_argument('--report',default=str(ROOT/'exp'/'v318_environment_report.json'))
    args=p.parse_args()
    if args.stage:
        if args.stage.startswith('augment:'):check_augmentation(args.stage.split(':',1)[1])
        elif args.stage=='interface':
            from .interface_smoke import main as smoke
            smoke()
        else:{'versions':check_versions,'scientific':check_scientific,'native':check_native,'cuda':check_cuda,'omni':check_omni}[args.stage]()
    elif not run_checks(plans(),args.report)['ok']:
        raise SystemExit(1)


if __name__=='__main__':main()
