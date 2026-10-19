"""Reuse the isolated phoneme runtime and cache with V3.18.7 short-input rules."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
from w2v_v3186.prepare_environment import (
    ROOT, SCHEMA, PINS, CORE, private_environment, pip_command, run_capture,
    probe, check_private_environment, installed_pins, environment_lock, _json,
)
from w2v_v3186.phoneme_download import add_download_arguments
from .source import resolve_data_run


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-run','--parent-run',dest='data_run',help='Reuse Train manifests, never detector weights')
    p.add_argument('--output-dir',default=str(ROOT/'assets/phonemes_v3186'),
        help='Reuse compatible V3.18.6 metadata; no second normal cache is needed')
    p.add_argument('--model-cache',default='/home/ubuntu/LXT/v318-runtime/hf-phonemes')
    p.add_argument('--environment-root',default='/home/ubuntu/LXT/v318-runtime/phoneme-env-v3186')
    p.add_argument('--cache-root',default='/home/ubuntu/LXT/v318-runtime/phoneme-prep-cache')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--local-files-only',action='store_true',help='Never download model or dependency wheels')
    add_download_arguments(p)
    return p


def preparation_command(python,args,data_run):
    command=[str(python),'-u','-m','w2v_v31813.prepare_phonemes',
        '--data-run',str(data_run),'--output-dir',str(Path(args.output_dir).expanduser().resolve()),
        '--model-cache',str(Path(args.model_cache).expanduser().resolve()),'--device',args.device,
        '--hf-endpoint',args.hf_endpoint,'--download-retries',str(args.download_retries)]
    if args.local_files_only:command.append('--local-files-only')
    return command


def execute(args):
    data_run=resolve_data_run(args.data_run)
    output=Path(args.output_dir).expanduser().resolve()
    cache=Path(args.cache_root).expanduser().resolve()
    model_cache=Path(args.model_cache).expanduser().resolve()
    private=Path(args.environment_root).expanduser().resolve()
    base_python=Path(sys.executable).absolute()
    if private in (Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()):
        raise ValueError('--environment-root must not be the current training environment')
    for path in (cache,cache/'tmp',cache/'pip',cache/'wheels',model_cache):path.mkdir(parents=True,exist_ok=True)
    env=private_environment(cache,model_cache)
    with environment_lock(cache/'prepare.lock'):
        if (output/'manifest.json').is_file():
            print('Existing phonemes: verify original bytes and metadata; no recognizer or installation.',flush=True)
            subprocess.run(preparation_command(base_python,args,data_run),cwd=ROOT,env=env,check=True)
            return
        base=probe(base_python,env)
        # Reuse the exact managed V3.18.6 runtime ownership contract. This is
        # deliberately not renamed: no dependency upgrade/new model is needed.
        owner=dict(schema=SCHEMA,base_python=str(base_python),python=base['python'],
            base_core={name:base['packages'].get(name) for name in CORE},
            core_imports=base['core_imports'],pins=PINS)
        owner_path=private/'v3186_environment.json'
        if private.exists():
            if not owner_path.is_file() or json.loads(owner_path.read_text(encoding='utf-8'))!=owner:
                raise ValueError('Existing environment is not the managed phoneme runtime; choose another empty path')
        else:
            private.mkdir(parents=True);_json(owner_path,owner)
        python=private/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if not (private/'pyvenv.cfg').is_file() or not python.is_file():
            print('Creating isolated phoneme environment: '+str(private),flush=True)
            saved=dict(os.environ)
            try:
                os.environ.clear();os.environ.update(env)
                venv.EnvBuilder(system_site_packages=True,with_pip=True).create(private)
            finally:os.environ.clear();os.environ.update(saved)
        candidate=probe(python,env);check_private_environment(base,candidate,private)
        if not installed_pins(candidate):
            print('Installing pinned dependencies only inside the phoneme runtime.',flush=True)
            output_log=run_capture(pip_command(python,cache,args.local_files_only),env)
            (cache/'dependency_install.log').write_text(output_log,encoding='utf-8')
            candidate=probe(python,env);check_private_environment(base,candidate,private)
            if not installed_pins(candidate):raise RuntimeError('Pinned phoneme dependencies did not install correctly')
        if probe(base_python,env)!=base:
            raise RuntimeError('Original training environment changed during isolated preparation')
        subprocess.run(preparation_command(python,args,data_run),cwd=ROOT,env=env,check=True)


def main():
    try:execute(parser().parse_args())
    except subprocess.CalledProcessError as exc:
        print('PHONEME_PREPARATION_FAILED=Child exited '+str(exc.returncode)+
            '; completed environment/model/metadata retained. Fix the error and rerun the same command.',
            file=sys.stderr,flush=True)
        raise SystemExit(exc.returncode if exc.returncode>0 else 1) from None


if __name__=='__main__':main()
