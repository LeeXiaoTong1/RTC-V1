"""Recover the confirmed epoch-0-only run without regenerating audio caches.

Dry run by default. --run starts a NEW experiment. --release-feature-cache
explicitly removes only disposable generated .npy files in this checkout's
data/w2v_feature_cache directory. Existing experiments and all audio stay intact.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import torch
from w2v_rebuild.storage import GIB, MIB, available_bytes, prune_feature_files, require_space, storage_bytes
from w2v_rebuild.train import parser as train_parser

ROOT = Path(__file__).resolve().parent


def training_command(config, out, baseline, preflight):
    values = dict(config)
    values.update(out=str(out), init=None, resume=None, finetune_from=str(baseline),
                  preflight=preflight, check_data=False, profile_steps=1 if preflight else 0,
                  feature_cache=None, no_feature_cache=True)
    command = [sys.executable, '-u', '-m', 'w2v_rebuild.train']
    for action in train_parser()._actions:
        value = values.get(action.dest)
        if value is None or action.dest == 'help':
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(option)
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                command.extend([option, str(item)])
        else:
            command.extend([option, str(value)])
    return command


@contextmanager
def recovery_lock():
    # Linux server only. The parent holds the lock through both child processes.
    import fcntl
    with (ROOT/'exp'/'.storage_recovery.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another storage recovery is already running in this checkout') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-run', required=True)
    parser.add_argument('--release-feature-cache', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    failed = Path(args.failed_run).expanduser().resolve()
    failed.relative_to((ROOT/'exp').resolve())
    config = json.loads((failed/'stage3'/'config.json').read_text(encoding='utf-8'))
    if config.get('stage') != 3 or not config.get('adaptation'):
        raise ValueError('Expected an improved Stage3 fine-tuning experiment')
    if (failed/'stage3'/'last.pt').exists():
        raise ValueError('last.pt exists: inspect it for normal resume instead of restarting')
    saved = torch.load(failed/'stage3'/'best_model.pt', map_location='cpu', weights_only=True, mmap=True)
    if saved.get('epoch') != 0 or saved.get('kind') != 'weights':
        raise ValueError('This helper is only for the confirmed epoch-0-only failure')
    weights = storage_bytes(saved['model'])
    del saved
    baseline = Path(config.get('baseline_path') or config.get('finetune_from') or '').expanduser().resolve()
    if not baseline.is_file():
        raise FileNotFoundError(f'Original baseline is missing: {baseline}')
    for cache in ([config['train_noisy_cache']] + config.get('extra_train_noisy_cache', [])
                  + [config['dev_noisy_cache'], config['dev_heldout_cache']]):
        for name in ('config.json', 'manifest.jsonl'):
            if not (Path(cache)/name).is_file():
                raise FileNotFoundError(str(Path(cache)/name))
    feature_root = Path(config.get('feature_cache') or '').expanduser()
    if args.release_feature_cache:
        if (feature_root.is_symlink() or not feature_root.is_dir()
                or feature_root.resolve() != (ROOT/'data'/'w2v_feature_cache').resolve()):
            raise ValueError('Refusing cleanup outside this checkout/data/w2v_feature_cache')
    print(f'FAILED_RUN={failed}\nBASELINE={baseline}', flush=True)
    print('Audio caches will be reused. Feature caching will be disabled for this recovery.', flush=True)
    # Conservative estimate before loading any model onto the GPU.
    needed = weights + 64*MIB + 2*(3*weights + 128*MIB) + GIB
    print(f'Free={available_bytes(ROOT)/GIB:.2f} GiB; new-run headroom={needed/GIB:.2f} GiB', flush=True)
    if not args.run:
        print('Dry run only. Add --run to start; add --release-feature-cache to reclaim generated features.')
        return
    with recovery_lock():
        if args.release_feature_cache:
            count, size = prune_feature_files(feature_root)
            print(f'Released {count} generated feature files, {size/GIB:.2f} GiB. No audio/checkpoints removed.', flush=True)
        require_space(ROOT/'exp', needed, 'Recovery startup')
        run = ROOT/'exp'/('w2v_recovered_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
        run.mkdir(exist_ok=False)
        print(f'RUN={run}', flush=True)
        (run/'recovery.json').write_text(json.dumps({'failed_run': str(failed), 'baseline': str(baseline),
                                                     'feature_cache_disabled': True}, indent=2), encoding='utf-8')
        env = os.environ.copy()
        env.update({key: str(value) for key, value in config.get('noise_environment', {}).items()})
        for preflight in (True, False):
            print('[1/2] Preflight' if preflight else '[2/2] Training from baseline, epoch 1', flush=True)
            subprocess.run(training_command(config, run/'stage3', baseline, preflight),
                           cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
