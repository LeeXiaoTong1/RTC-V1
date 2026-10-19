"""Install old ESPnet dependencies in a separate, owned preparation environment."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
from w2v_v31810.common import ROOT,atomic_json,read_json
from .source import resolve_parent
from .teacher import ESPNET_REVISION

PINS=['numpy==1.23.5','scipy==1.10.1','scikit-learn==1.3.2','numba==0.57.1','llvmlite==0.40.1',
      'librosa==0.9.2','typeguard==2.13.3','humanfriendly==10.0','huggingface-hub==0.23.5',
      'importlib-metadata==4.13.0','setuptools<74','packaging','configargparse>=1.2.1',
      'filelock','jamo==0.4.1','PyYAML','soundfile','h5py','kaldiio>=2.18.0','torch_complex',
      'nltk','protobuf==3.20.3','hydra-core','opt-einsum','sentencepiece==0.2.0','lightning==2.5.1',
      'ctc-segmentation>=1.6.6','pyworld>=0.3.4','pypinyin==0.44.0','espnet_tts_frontend',
      'ci_sdr','fast-bss-eval==0.1.3','asteroid_filterbanks==0.4.0','editdistance']


def snapshot():
    return {d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')}


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-run')
    p.add_argument('--environment-root',default='/home/ubuntu/LXT/v318-runtime/xeus-env-v31813')
    p.add_argument('--cache-root',default='/home/ubuntu/LXT/v318-runtime/xeus-prep-cache')
    p.add_argument('--model-cache',default='/home/ubuntu/LXT/v318-runtime/hf-xeus')
    p.add_argument('--output',default=str(ROOT/'assets/xeus_v31813'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--hf-endpoint');p.add_argument('--local-files-only',action='store_true')
    return p


def execute(a):
    parent=resolve_parent(a.parent_run)
    # A completed, bound index needs neither dependency installation nor a GPU.
    if (Path(a.output)/'manifest.json').is_file():
        from .prior import bind
        bound=bind(Path(a.output)/'manifest.json',read_json(parent/'train_rows.json'))
        print('XEUS_MANIFEST='+bound['path']);return
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Use the existing Python 3.11 Omni training environment to prepare XEUS')
    private=Path(a.environment_root).expanduser().resolve();cache=Path(a.cache_root).expanduser().resolve()
    if private in (Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()):raise ValueError('XEUS needs a separate environment')
    base=snapshot();owner=dict(schema='v31813_xeus_env_v1',base_python=sys.executable,torch=metadata.version('torch'),espnet=ESPNET_REVISION,pins=PINS)
    if private.exists() and (not (private/'owner.json').exists() or read_json(private/'owner.json')!=owner):
        raise ValueError('Environment path is not owned by this XEUS preparation; choose a new path')
    private.mkdir(parents=True,exist_ok=True);atomic_json(private/'owner.json',owner)
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1'
    for key in ('PIP_TARGET','PIP_PREFIX','PIP_USER','PYTHONHOME'):env.pop(key,None)
    for name in ('pip','tmp'): (cache/name).mkdir(parents=True,exist_ok=True)
    env.update(PIP_CACHE_DIR=str(cache/'pip'),TMPDIR=str(cache/'tmp'),TEMP=str(cache/'tmp'),TMP=str(cache/'tmp'))
    python=private/'bin/python'
    if not python.exists():venv.EnvBuilder(with_pip=True,system_site_packages=True).create(private)
    stamp=private/'installed.json'
    if not stamp.exists():
        constraints=cache/'constraints.txt'
        constraints.write_text('torch=='+metadata.version('torch')+'\ntorchaudio=='+metadata.version('torchaudio')+'\n',encoding='utf-8')
        command=[str(python),'-m','pip','install','--disable-pip-version-check','-c',str(constraints),*PINS]
        if a.local_files_only:command+=['--no-index']
        subprocess.run(command,env=env,cwd=ROOT,check=True)
        if a.local_files_only:raise ValueError('Pinned ESPnet source install requires network on its first preparation')
        subprocess.run([str(python),'-m','pip','install','--no-deps',
            'espnet @ git+https://github.com/espnet/espnet.git@'+ESPNET_REVISION],env=env,cwd=ROOT,check=True)
    check='from espnet2.tasks.ssl import SSLTask; import sklearn, torch, json, importlib.metadata as m; '+\
          'assert m.version("torch")=='+repr(owner['torch'])+'; '+\
          'assert json.loads(m.distribution("espnet").read_text("direct_url.json"))["vcs_info"]["commit_id"]=='+repr(ESPNET_REVISION)+'; '+\
          'assert m.version("numpy")=="1.23.5" and m.version("scikit-learn")=="1.3.2"; print("XEUS runtime imports and revision verified")'
    subprocess.run([str(python),'-c',check],env=env,cwd=ROOT,check=True)
    atomic_json(stamp,owner)
    if snapshot()!=base:raise RuntimeError('The base training environment unexpectedly changed')
    cmd=[str(python),'-u','-m','w2v_v31813.prior','--parent-run',str(parent),'--output',a.output,
         '--model-cache',a.model_cache,'--device',a.device]
    if a.hf_endpoint:cmd+=['--hf-endpoint',a.hf_endpoint]
    if a.local_files_only:cmd+=['--local-files-only']
    subprocess.run(cmd,env=env,cwd=ROOT,check=True)


def main():
    from w2v_aasist.launch import run_lock
    a=parser().parse_args()
    with run_lock(Path(a.cache_root)/'prepare.lock'):execute(a)

if __name__=='__main__':main()
