"""Install pinned ESPnet in an owned environment, with actionable build logs."""
import argparse
from collections import deque
from datetime import datetime,timezone
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

ESPNET_ARCHIVE='https://codeload.github.com/espnet/espnet/tar.gz/'+ESPNET_REVISION
# Official archive: tar PAX comment equals ESPNET_REVISION; version.txt=202506.
ESPNET_SHA256='610886085a9a133e029b1fc14bb309e68a60584058979b0a200879f26f40ef5b'
BUILD_PINS=['setuptools==73.0.1','wheel==0.45.1','pytest-runner==6.0.1']

PINS=['numpy==1.23.5','scipy==1.10.1','scikit-learn==1.3.2','numba==0.57.1','llvmlite==0.40.1',
      'librosa==0.9.2','typeguard==2.13.3','humanfriendly==10.0','huggingface-hub==0.23.5',
      'importlib-metadata==4.13.0','setuptools<74','packaging','configargparse>=1.2.1',
      'filelock','jamo==0.4.1','PyYAML','soundfile','h5py','kaldiio>=2.18.0','torch_complex',
      'nltk','protobuf==3.20.3','hydra-core','opt-einsum','sentencepiece==0.2.0','lightning==2.5.1',
      'ctc-segmentation>=1.6.6','pyworld>=0.3.4','pypinyin==0.44.0','espnet_tts_frontend',
      'ci_sdr','fast-bss-eval==0.1.3','asteroid_filterbanks==0.4.0','editdistance']


def snapshot():
    return {d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')}


def run_logged(command,env,log,cwd=ROOT):
    """Stream the child output, persist it, and retain the actual error on failure."""
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);tail=deque(maxlen=40)
    print('XEUS_PREP_LOG='+str(log),flush=True)
    with log.open('a',encoding='utf-8') as stream:
        stream.write('\nAttempt '+datetime.now(timezone.utc).isoformat()+'\n');stream.flush()
        with subprocess.Popen(command,env=env,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',bufsize=1) as process:
            try:
                for line in process.stdout:
                    stream.write(line);stream.flush();print(line,end='',flush=True);tail.append(line)
                code=process.wait()
            except BaseException:
                if process.poll() is None:process.terminate();process.wait()
                raise
    if code:
        raise RuntimeError('XEUS preparation failed (exit '+str(code)+'). Full log: '+str(log)+
                           '\nLast child output:\n'+''.join(tail))


def espnet_command(python):
    # No Git clone; --no-deps alone does NOT pin PEP 517 build dependencies.
    return [str(python),'-m','pip','install','--disable-pip-version-check','--no-deps',
            '--no-build-isolation','--retries','3','--timeout','30',
            'espnet @ '+ESPNET_ARCHIVE+'#sha256='+ESPNET_SHA256]


def runtime_check(private,torch_version):
    # Accept already successful v1 Git installs as well as this exact archive.
    return ('import sys,json,pathlib,importlib.metadata as m; '
            'assert pathlib.Path(sys.prefix).resolve()==pathlib.Path('+repr(str(private))+').resolve(), "Wrong XEUS environment"; '
            'd=m.distribution("espnet"); '
            'assert pathlib.Path(d.locate_file("")).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()), "ESPnet must be installed in the private environment"; '
            'p=json.loads(d.read_text("direct_url.json") or "{}"); '
            'a=p.get("archive_info",{}); '
            'git=(p.get("url")=="https://github.com/espnet/espnet.git" and p.get("vcs_info",{}).get("commit_id")=='+repr(ESPNET_REVISION)+'); '
            'archive=(p.get("url")=='+repr(ESPNET_ARCHIVE)+' and (a.get("hashes",{}).get("sha256")=='+repr(ESPNET_SHA256)+
            ' or a.get("hash")=='+repr('sha256='+ESPNET_SHA256)+')); '
            'assert git or archive, "ESPnet source does not match the pinned official revision"; '
            'assert m.version("espnet")=="202506"; '
            'from espnet2.tasks.ssl import SSLTask; import sklearn,torch; '
            'assert m.version("torch")=='+repr(torch_version)+'; '
            'assert m.version("numpy")=="1.23.5" and m.version("scikit-learn")=="1.3.2"; '
            'print("XEUS runtime imports and revision verified")')


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
    for key in ('PIP_TARGET','PIP_PREFIX','PYTHONHOME'):env.pop(key,None)
    env.update(PIP_CONFIG_FILE=os.devnull,PIP_USER='0')
    for name in ('pip','tmp'): (cache/name).mkdir(parents=True,exist_ok=True)
    env.update(PIP_CACHE_DIR=str(cache/'pip'),TMPDIR=str(cache/'tmp'),TEMP=str(cache/'tmp'),TMP=str(cache/'tmp'))
    python=private/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists() or not (private/'pyvenv.cfg').exists():
        saved=dict(os.environ)
        try:
            os.environ.clear();os.environ.update(env)
            venv.EnvBuilder(with_pip=True,system_site_packages=True).create(private)
        finally:os.environ.clear();os.environ.update(saved)
    stamp=private/'installed.json'
    if stamp.exists() and read_json(stamp)!=owner:raise ValueError('XEUS installation receipt differs from its environment owner')
    if not stamp.exists():
        if a.local_files_only:raise ValueError('First XEUS environment installation needs network; rerun without --local-files-only')
        constraints=cache/'constraints.txt'
        constraints.write_text('torch=='+metadata.version('torch')+'\ntorchaudio=='+metadata.version('torchaudio')+'\n',encoding='utf-8')
        command=[str(python),'-m','pip','install','--disable-pip-version-check','-c',str(constraints),*PINS]
        run_logged(command,env,cache/'dependency_install.log')
        run_logged([str(python),'-m','pip','install','--disable-pip-version-check','--no-deps',*BUILD_PINS],
                   env,cache/'build_tools_install.log')
        run_logged(espnet_command(python),env,cache/'espnet_install.log')
    run_logged([str(python),'-c',runtime_check(private,owner['torch'])],env,cache/'runtime_check.log')
    if snapshot()!=base:raise RuntimeError('The base training environment unexpectedly changed')
    atomic_json(stamp,owner)
    cmd=[str(python),'-u','-m','w2v_v31813.prior','--parent-run',str(parent),'--output',a.output,
         '--model-cache',a.model_cache,'--device',a.device]
    if a.hf_endpoint:cmd+=['--hf-endpoint',a.hf_endpoint]
    if a.local_files_only:cmd+=['--local-files-only']
    subprocess.run(cmd,env=env,cwd=ROOT,check=True)


def main():
    from w2v_aasist.launch import run_lock
    a=parser().parse_args()
    try:
        with run_lock(Path(a.cache_root)/'prepare.lock'):execute(a)
    except RuntimeError as exc:
        print('XEUS_PREPARATION_FAILED='+str(exc)+
              '\nCompleted preparation files are retained. Fix the error and rerun the same command.',file=sys.stderr,flush=True)
        raise SystemExit(1) from None

if __name__=='__main__':main()
