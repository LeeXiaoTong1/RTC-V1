"""Standard-library-only guards used before repairing the Python environment."""
import argparse
from importlib.metadata import PackageNotFoundError,version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

SOURCE_COMMIT='6fa0aaf178db437bde0fae125b36105dc123119d'


def select_profile(requested,installed,drivers):
    if requested not in ('auto','11.8','12.6'):raise ValueError('Unknown CUDA profile')
    if requested!='auto':return 'cu'+requested.replace('.','')
    # A repair must not undo a working downgrade, even on a newer host.
    if installed=='2.6.0+cu118':return 'cu118'
    if not drivers:raise RuntimeError('Cannot query the driver; select --cuda 11.8 or --cuda 12.6 explicitly')
    if any(int(d.split('.')[0])<525 for d in drivers):return 'cu118'
    return 'cu126'


def profile(requested):
    try:installed=version('torch')
    except PackageNotFoundError:installed=''
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
            capture_output=True,text=True,timeout=15,check=True)
        drivers=[x.strip() for x in result.stdout.splitlines() if re.fullmatch(r'\d+(\.\d+)+',x.strip())]
    except (OSError,subprocess.SubprocessError):drivers=[]
    return select_profile(requested,installed,drivers)


def active_jobs(proc=Path('/proc'),current=None,uid=None):
    current=os.getpid() if current is None else current
    uid=os.getuid() if uid is None else uid
    jobs=[]
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name)==current:continue
        try:
            if entry.stat().st_uid!=uid:continue
            args=[x.decode(errors='replace') for x in (entry/'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not args or 'python' not in Path(args[0]).name.lower():continue
        if any(re.search(r'w2v_[\w]+\.(workflow|train|validate|inference|evaluate|preflight|environment)$',a)
               or re.search(r'(main_train|main_eval|start_w2v)[\w]*\.py$',a) for a in args):
            jobs.append(dict(pid=int(entry.name),command=' '.join(args)))
    return jobs


def guard():
    if platform.system()!='Linux' or platform.machine()!='x86_64':raise RuntimeError('This installer requires Linux x86_64')
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Use Python 3.11 in sdd-v318')
    prefix=Path(os.environ.get('CONDA_PREFIX','/missing')).resolve()
    if prefix.name!='sdd-v318' or prefix!=Path(sys.prefix).resolve():raise RuntimeError('Activate the isolated sdd-v318 environment')
    jobs=active_jobs()
    if jobs:raise RuntimeError('Wait for active training/evaluation before changing packages: '+json.dumps(jobs))
    for path in (Path.cwd(),prefix,Path(os.environ.get('TMPDIR','/tmp'))):
        free=shutil.disk_usage(path).free/2**30
        if free<12:raise RuntimeError(f'Environment repair needs at least 12 GiB temporary headroom at {path}; free={free:.2f} GiB')


def record_build(destination):
    from .runtime import native_abi
    import fairseq2n
    import hashlib
    libraries={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(fairseq2n.get_lib().glob('*.so*')) if p.is_file()}
    if not libraries:raise RuntimeError('Installed fairseq2n shared library not found')
    data=dict(source_commit=SOURCE_COMMIT,python=sys.version,abi=native_abi('cu118'),
        custom_cuda_kernels=False,image_decoder=False,libraries=libraries)
    Path(destination).write_text(json.dumps(data,indent=2),encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('guard','profile','record-build'))
    p.add_argument('--cuda',default='auto',choices=('auto','11.8','12.6'))
    p.add_argument('--output')
    args=p.parse_args()
    if args.action=='guard':guard()
    elif args.action=='profile':print(profile(args.cuda))
    else:
        if not args.output:p.error('record-build requires --output')
        record_build(args.output)


if __name__=='__main__':main()
