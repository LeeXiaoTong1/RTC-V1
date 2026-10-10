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


def host_guard():
    if platform.system()!='Linux' or platform.machine()!='x86_64':raise RuntimeError('This installer requires Linux x86_64')
    jobs=active_jobs()
    if jobs:raise RuntimeError('Wait for active training/evaluation before changing packages: '+json.dumps(jobs))


def storage_status(paths):
    records=[]
    for role,path in paths:
        path=Path(path).expanduser().resolve();existing=path
        while not existing.exists():
            if existing.parent==existing:raise FileNotFoundError(path)
            existing=existing.parent
        records.append(dict(role=role,path=str(path),filesystem=existing.stat().st_dev,
            free_GiB=round(shutil.disk_usage(existing).free/2**30,2)))
    return records


def require_storage(records,minimum):
    print('V318_STORAGE='+json.dumps(records),file=sys.stderr,flush=True)
    insufficient=[r for r in records if r['free_GiB']<minimum]
    if insufficient:
        details='; '.join(f'{r["role"]}={r["path"]}: {r["free_GiB"]:.2f} GiB free' for r in insufficient)
        raise RuntimeError(f'Need {minimum:g} GiB temporary headroom on the affected filesystem: '+details+
            '. Use repair_w2v_v318.sh --runtime-root /absolute/path/on/a/larger/filesystem to place the isolated environment, temporary files and Conda cache there; freeing files on a different filesystem will not help.')


def guard():
    host_guard()
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Use Python 3.11 in sdd-v318')
    prefix=Path(os.environ.get('CONDA_PREFIX','/missing')).resolve()
    if prefix.name!='sdd-v318' or prefix!=Path(sys.prefix).resolve():raise RuntimeError('Activate the isolated sdd-v318 environment by name or its full prefix path')
    paths=[('project',Path.cwd()),('environment',prefix),('temporary',Path(os.environ.get('TMPDIR','/tmp')))]
    for path in os.environ.get('CONDA_PKGS_DIRS','').split(os.pathsep):
        if path:paths.append(('conda_cache',Path(path)))
    require_storage(storage_status(paths),12)


def prepare_runtime(root):
    host_guard()
    supplied=Path(root).expanduser()
    if not supplied.is_absolute():raise ValueError('--runtime-root must be an absolute path')
    root=supplied.resolve();prefix=root/'envs'/'sdd-v318'
    if root.exists() and not root.is_dir():raise ValueError('Runtime root is not a directory: '+str(root))
    if prefix.exists() and not ((prefix/'conda-meta'/'history').is_file() and (prefix/'bin'/'python').is_file()):
        raise RuntimeError('Existing destination is not a complete Conda environment; it was not modified: '+str(prefix))
    # New environment + downloads/build tools require more headroom than repair.
    require_storage(storage_status([('runtime_root',root),('environment',prefix),
        ('temporary',root/'tmp'),('conda_cache',root/'pkgs')]),12 if prefix.exists() else 20)
    for path in (root/'tmp',root/'pkgs',root/'envs',root/'logs'):path.mkdir(parents=True,exist_ok=True)
    return root


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
    p.add_argument('action',choices=('guard','profile','record-build','prepare-runtime'))
    p.add_argument('--cuda',default='auto',choices=('auto','11.8','12.6'))
    p.add_argument('--output')
    p.add_argument('--root')
    args=p.parse_args()
    if args.action=='guard':guard()
    elif args.action=='profile':print(profile(args.cuda))
    elif args.action=='prepare-runtime':
        if not args.root:p.error('prepare-runtime requires --root')
        print(prepare_runtime(args.root))
    else:
        if not args.output:p.error('record-build requires --output')
        record_build(args.output)


if __name__=='__main__':main()
