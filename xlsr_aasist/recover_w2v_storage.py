"""Recover the confirmed epoch-0-only run without regenerating audio caches.

Dry run by default. --run starts a NEW experiment. --prune-old-checkpoints
removes only the five explicitly listed checkpoint files from the completed
baseline experiment, preserving its Stage3 best, configuration and logs.
--keep-feature-cache preserves cache acceleration. Alternatively,
--release-feature-cache removes generated features from this checkout only.
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


def training_command(config, out, baseline, preflight, keep_feature_cache=False):
    values = dict(config)
    values.update(out=str(out), init=None, resume=None, finetune_from=str(baseline),
                  preflight=preflight, check_data=False, profile_steps=1 if preflight else 0,
                  feature_cache=config.get('feature_cache') if keep_feature_cache else None,
                  no_feature_cache=not keep_feature_cache)
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


def old_checkpoint_plan(baseline):
    """Exact whitelist under the original completed run; never glob other runs."""
    baseline = Path(baseline)
    if baseline.name != 'best_model.pt' or baseline.parent.name != 'stage3' or not baseline.is_file():
        raise ValueError('Cleanup requires the original stage3/best_model.pt')
    original = baseline.parent.parent.resolve()
    if original.parent.name != 'exp':
        raise ValueError('Expected the original baseline under an exp/RUN directory')
    kept = baseline.resolve()
    saved = torch.load(kept, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('stage') != 3 or saved.get('kind') != 'weights' or 'model' not in saved:
        raise ValueError('The retained Stage3 best checkpoint is not readable as expected')
    del saved
    plan = []
    names = [(1, 'best_model.pt'), (1, 'last.pt'), (2, 'best_model.pt'),
             (2, 'last.pt'), (3, 'last.pt')]
    for stage, name in names:
        folder = original/f'stage{stage}'
        path = folder/name
        if folder.is_symlink() or path.is_symlink():
            raise ValueError(f'Refusing checkpoint cleanup through a symlink: {path}')
        if not path.exists():
            continue
        if not path.is_file() or not path.resolve().is_relative_to(original) or path.resolve() == kept:
            raise ValueError(f'Unexpected checkpoint cleanup target: {path}')
        stat = path.stat()
        plan.append((path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)))
    return plan


def prune_old_checkpoints(baseline, expected_plan):
    # Validate the complete list again before deleting any file, including the kept best.
    if old_checkpoint_plan(baseline) != expected_plan:
        raise RuntimeError('Original checkpoint files changed; rerun the preview before cleanup')
    for path, _ in expected_plan:
        path.unlink()
        print(f'Removed old checkpoint: {path}', flush=True)
    return sum(state[2] for _, state in expected_plan)


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
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument('--release-feature-cache', action='store_true')
    cache_mode.add_argument('--keep-feature-cache', action='store_true')
    parser.add_argument('--prune-old-checkpoints', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    failed = Path(args.failed_run).expanduser().resolve()
    failed.relative_to((ROOT/'exp').resolve())
    config = json.loads((failed/'stage3'/'config.json').read_text(encoding='utf-8'))
    if config.get('stage') != 3 or not config.get('adaptation'):
        raise ValueError('Expected an improved Stage3 fine-tuning experiment')
    if args.keep_feature_cache and not config.get('feature_cache'):
        raise ValueError('The specified failed experiment has no configured feature cache')
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
    if args.prune_old_checkpoints and baseline.parent.parent.resolve() == failed:
        raise ValueError('Refusing to classify the failed/current experiment as the old baseline run')
    old_plan = old_checkpoint_plan(baseline) if args.prune_old_checkpoints else []
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
    print('Audio caches will be reused. Feature caching: '+('enabled' if args.keep_feature_cache else 'disabled'), flush=True)
    if args.prune_old_checkpoints:
        print(f'KEEP_BASELINE={baseline}', flush=True)
        for path, state in old_plan:
            print(f'DELETE_OLD_CHECKPOINT={path} ({state[2]/GIB:.2f} GiB)', flush=True)
        print(f'Old checkpoint files to remove: {len(old_plan)}, {sum(s[2] for _, s in old_plan)/GIB:.2f} GiB', flush=True)
    # Conservative estimate before loading any model onto the GPU.
    needed = weights + 64*MIB + 2*(3*weights + 128*MIB) + GIB
    print(f'Free={available_bytes(ROOT)/GIB:.2f} GiB; new-run headroom={needed/GIB:.2f} GiB', flush=True)
    if not args.run:
        print('Dry run only. Add --run to apply the selected cleanup and start a new experiment.')
        return
    with recovery_lock():
        if args.prune_old_checkpoints:
            size = prune_old_checkpoints(baseline, old_plan)
            print(f'Removed old checkpoint files totaling {size/GIB:.2f} GiB; Stage3 best kept.', flush=True)
        if args.release_feature_cache:
            count, size = prune_feature_files(feature_root)
            print(f'Released {count} generated feature files, {size/GIB:.2f} GiB. No audio/checkpoints removed.', flush=True)
        require_space(ROOT/'exp', needed, 'Recovery startup')
        run = ROOT/'exp'/('w2v_recovered_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(2))
        run.mkdir(exist_ok=False)
        print(f'RUN={run}', flush=True)
        (run/'recovery.json').write_text(json.dumps({'failed_run': str(failed), 'baseline': str(baseline),
                                                     'feature_cache_disabled': not args.keep_feature_cache,
                                                     'removed_old_checkpoints': [str(p) for p, _ in old_plan]},
                                                    indent=2), encoding='utf-8')
        env = os.environ.copy()
        env.update({key: str(value) for key, value in config.get('noise_environment', {}).items()})
        for preflight in (True, False):
            print('[1/2] Preflight' if preflight else '[2/2] Training from baseline, epoch 1', flush=True)
            subprocess.run(training_command(config, run/'stage3', baseline, preflight, args.keep_feature_cache),
                           cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
