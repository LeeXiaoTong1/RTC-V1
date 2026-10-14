"""One-command isolated phoneme preparation; the Omni environment is read-only.

Only the small dependency environment is bootstrapped here. The public CTC
model download remains pinned by prepare_phonemes.py. Dependencies, pip cache,
temporary files and model weights all use explicit data-disk directories.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 'v3186-phoneme-preparation-environment-v1'
# Install with --no-deps/--ignore-installed into our own venv. Never let pip's
# resolver replace torch, numpy, scipy, soundfile, fairseq or the original HF
# packages. This list is the complete runtime dependency closure needed here.
PINS = {
    'transformers': '4.38.2', 'tokenizers': '0.15.2',
    'huggingface-hub': '0.23.5', 'safetensors': '0.4.5',
    'filelock': '3.16.1', 'fsspec': '2024.10.0',
    'packaging': '24.2', 'PyYAML': '6.0.2', 'regex': '2024.11.6',
    'requests': '2.32.3', 'tqdm': '4.67.1', 'typing-extensions': '4.12.2',
    'charset-normalizer': '3.4.1', 'idna': '3.10',
    'urllib3': '2.3.0', 'certifi': '2024.12.14',
}
CORE = ('torch', 'numpy', 'scipy', 'soundfile', 'fairseq2', 'fairseq2n')
PROBE = r'''
import importlib.metadata as m, json, pathlib, sys
import numpy, soundfile, torch
names = json.loads(sys.argv[1])
result = {}
for name in names:
    try:
        d = m.distribution(name)
        result[name] = {'version': d.version, 'location': str(pathlib.Path(d.locate_file('')).resolve())}
    except m.PackageNotFoundError:
        result[name] = None
print(json.dumps({'packages': result, 'python': sys.version,
                 'base_prefix': str(pathlib.Path(sys.base_prefix).resolve()),
                 'prefix': str(pathlib.Path(sys.prefix).resolve()),
                 'core_imports': {k: str(pathlib.Path(v.__file__).resolve()) for k, v in
                                 [('torch', torch), ('numpy', numpy), ('soundfile', soundfile)]}}))
'''


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', help='Completed V3.18.5 parent; auto-discover when the latest-run pointer is missing')
    p.add_argument('--output-dir', default=str(ROOT / 'assets/phonemes_v3186'))
    p.add_argument('--model-cache', default='/home/ubuntu/LXT/v318-runtime/hf-phonemes')
    p.add_argument('--environment-root', default='/home/ubuntu/LXT/v318-runtime/phoneme-env-v3186')
    p.add_argument('--cache-root', default='/home/ubuntu/LXT/v318-runtime/phoneme-prep-cache')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--local-files-only', action='store_true',
                   help='No network: require prepared dependency wheels and pinned model already cached')
    return p


def resolve_parent(value):
    from .parent_discovery import resolve_parent as discover
    return discover(value,root=ROOT)


def private_environment(cache_root, model_cache):
    cache_root = Path(cache_root); model_cache = Path(model_cache)
    result = dict(os.environ)
    # Avoid a user's PYTHONPATH or pip config sending installation into the
    # training environment, enabling --user, or overriding our fixed versions.
    for name in ('PYTHONHOME', 'PYTHONPATH', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_REQUIRE_VIRTUALENV'):
        result.pop(name, None)
    result.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull, PIP_USER='0',
                  PIP_DISABLE_PIP_VERSION_CHECK='1', TOKENIZERS_PARALLELISM='false',
                  TMPDIR=str(cache_root / 'tmp'), TMP=str(cache_root / 'tmp'), TEMP=str(cache_root / 'tmp'),
                  PIP_CACHE_DIR=str(cache_root / 'pip'), HF_HOME=str(model_cache),
                  HF_HUB_CACHE=str(model_cache), TRANSFORMERS_CACHE=str(model_cache))
    return result


def preparation_command(python, args, parent):
    command = [str(python), '-u', '-m', 'w2v_v3186.prepare_phonemes',
               '--parent-run', str(parent), '--output-dir', str(Path(args.output_dir).expanduser().resolve()),
               '--model-cache', str(Path(args.model_cache).expanduser().resolve()), '--device', args.device]
    if args.local_files_only: command.append('--local-files-only')
    return command


def pip_command(python, cache_root, local_files_only=False):
    result = [str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
              '--no-deps', '--ignore-installed', '--only-binary=:all:',
              '--cache-dir', str(Path(cache_root) / 'pip')]
    if local_files_only:
        # Offline mode never asks pip to contact a package index. An existing
        # compatible environment needs no install; otherwise place wheels here.
        result += ['--no-index', '--find-links', str(Path(cache_root) / 'wheels')]
    return result + [name + '==' + version for name, version in PINS.items()]


def run_capture(command, env):
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding='utf-8', errors='replace',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        tail = (result.stderr or result.stdout)[-4000:]
        raise RuntimeError('Preparation subprocess failed (exit ' + str(result.returncode) + '):\n' + tail)
    return result.stdout


def probe(python, env):
    output = run_capture([str(python), '-I', '-c', PROBE, json.dumps(list(CORE) + list(PINS))], env)
    return json.loads(output.strip().splitlines()[-1])


def check_private_environment(base, candidate, environment_root):
    if candidate['prefix'] != str(Path(environment_root).resolve()) or candidate['prefix'] == candidate['base_prefix']:
        raise ValueError('Refusing pip outside the dedicated phoneme virtual environment')
    for name in CORE:
        if candidate['packages'].get(name) != base['packages'].get(name):
            raise ValueError('Phoneme environment changed protected dependency: ' + name)
    if candidate['core_imports'] != base['core_imports']:
        raise ValueError('Protected torch/numpy/soundfile import paths changed')


def installed_pins(snapshot):
    return all(snapshot['packages'].get(name) is not None and
               snapshot['packages'][name]['version'] == version for name, version in PINS.items())


def _json(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2); f.write('\n')
    tmp.replace(path)


@contextmanager
def environment_lock(path):
    # Kernel file locks release automatically on Ctrl+C, reboot, or exceptions.
    # A leftover lock FILE is harmless; no stale-PID deletion is required.
    path = Path(path)
    with path.open('a+b') as f:
        if os.name == 'posix':
            import fcntl
            try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('Another phoneme preparation is already running') from None
        else:
            import msvcrt
            if path.stat().st_size == 0: f.write(b'0'); f.flush()
            f.seek(0)
            try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError: raise RuntimeError('Another phoneme preparation is already running') from None
        try: yield
        finally:
            if os.name == 'posix': fcntl.flock(f, fcntl.LOCK_UN)
            else: f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def execute(args):
    parent = resolve_parent(args.parent_run)
    output = Path(args.output_dir).expanduser().resolve()
    cache = Path(args.cache_root).expanduser().resolve()
    model_cache = Path(args.model_cache).expanduser().resolve()
    private = Path(args.environment_root).expanduser().resolve()
    # Preserve the executable symlink: resolving a venv's bin/python can point
    # outside the activated Omni environment and silently lose its packages.
    base_python = Path(sys.executable).absolute()
    if private == Path(sys.prefix).resolve() or private == Path(sys.base_prefix).resolve():
        raise ValueError('--environment-root must not be the current training environment')
    for path in (cache, cache / 'tmp', cache / 'pip', cache / 'wheels', model_cache): path.mkdir(parents=True, exist_ok=True)
    env = private_environment(cache, model_cache)
    with environment_lock(cache / 'prepare.lock'):
        # A finished cache is verified in the original Python using only
        # existing numpy; it never loads the recognizer or installs packages.
        if (output / 'manifest.json').is_file():
            print('Phoneme metadata exists; verifying original bytes and metadata only.', flush=True)
            subprocess.run(preparation_command(base_python, args, parent), cwd=ROOT, env=env, check=True)
            return
        base = probe(base_python, env)
        owner = dict(schema=SCHEMA, base_python=str(base_python), python=base['python'],
                     base_core={name: base['packages'].get(name) for name in CORE},
                     core_imports=base['core_imports'], pins=PINS)
        owner_path = private / 'v3186_environment.json'
        if private.exists():
            if not owner_path.is_file() or json.loads(owner_path.read_text(encoding='utf-8')) != owner:
                raise ValueError('Existing --environment-root is not this managed phoneme environment; choose another empty path')
        else:
            private.mkdir(parents=True)
            _json(owner_path, owner)
        python = private / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not (private / 'pyvenv.cfg').is_file() or not python.is_file():
            print('Creating isolated phoneme environment: ' + str(private), flush=True)
            # venv/ensurepip subprocesses inherit our data-disk temp/cache paths.
            saved = dict(os.environ)
            try:
                os.environ.clear(); os.environ.update(env)
                venv.EnvBuilder(system_site_packages=True, with_pip=True).create(private)
            finally:
                os.environ.clear(); os.environ.update(saved)
        candidate = probe(python, env); check_private_environment(base, candidate, private)
        if not installed_pins(candidate):
            print('Installing pinned phoneme dependencies in the isolated environment.', flush=True)
            output_log = run_capture(pip_command(python, cache, args.local_files_only), env)
            (cache / 'dependency_install.log').write_text(output_log, encoding='utf-8')
            candidate = probe(python, env); check_private_environment(base, candidate, private)
            if not installed_pins(candidate): raise RuntimeError('Pinned phoneme dependencies did not install correctly')
        if probe(base_python, env) != base:
            raise RuntimeError('Original training environment changed during isolated preparation')
        print('Phoneme environment ready; original torch/numpy/fairseq/HF environment unchanged.', flush=True)
        print('Model cache: ' + str(model_cache), flush=True)
        subprocess.run(preparation_command(python, args, parent), cwd=ROOT, env=env, check=True)


def main(): execute(parser().parse_args())


if __name__ == '__main__': main()
