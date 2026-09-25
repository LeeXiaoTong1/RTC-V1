"""Disk headroom for full Adam checkpoints and disposable feature files."""
import errno
import os
from pathlib import Path
import re
import shutil
import torch

GIB = 1024**3
MIB = 1024**2


def existing_parent(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def available_bytes(path):
    return shutil.disk_usage(existing_parent(path)).free


def storage_bytes(value):
    """Count unique tensor storages, including backing storage for tensor views."""
    seen = set()

    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(item, dict):
            return sum(visit(x) for x in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(x) for x in item)
        return 0

    return visit(value)


def checkpoint_sizes(model):
    weights = storage_bytes(model.state_dict())
    # The engine uses Adam with two moment buffers for every trainable parameter.
    moments = 2 * sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return weights + 64*MIB, weights + moments + 128*MIB


def checkpoint_headroom(out, best_bytes, full_bytes):
    out = Path(out)
    # Keep both the live last.pt and space for its next atomic replacement.
    required = full_bytes + GIB
    if not (out/'last.pt').is_file():
        required += full_bytes
    if not (out/'best_model.pt').is_file():
        required += best_bytes
    return required


def require_space(path, required, purpose):
    free = available_bytes(path)
    if free < required:
        raise OSError(errno.ENOSPC,
                      f'{purpose}: {free/GIB:.2f} GiB free, need at least {required/GIB:.2f} GiB '
                      f'on {existing_parent(path)}. Free disposable feature caches or expand storage; '
                      'do not delete training data or model checkpoints.')
    return free


def protect_checkpoint_space(out, model, feature_cache=None):
    best, full = checkpoint_sizes(model)
    required = checkpoint_headroom(out, best, full)
    free = require_space(out, required, 'Checkpoint space check')
    if feature_cache:
        # Cache files on another volume cannot consume the checkpoint budget.
        same_disk = existing_parent(feature_cache).stat().st_dev == existing_parent(out).stat().st_dev
        floor = required if same_disk else GIB
        previous = int(os.environ.get('W2V_FEATURE_CACHE_MIN_FREE_BYTES', '0'))
        os.environ['W2V_FEATURE_CACHE_MIN_FREE_BYTES'] = str(max(floor, previous))
    print(f'Storage: free={free/GIB:.2f} GiB, checkpoint headroom={required/GIB:.2f} GiB, '
          f'estimated last.pt={full/GIB:.2f} GiB', flush=True)
    return best, full


def prune_feature_files(root):
    """Explicit maintenance only: remove generated hash/hash/hash.npy regular files.

    No directory removal, symlink traversal, model files, audio or metadata deletion.
    The recovery caller additionally restricts root to its own known feature folder.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Expected a real feature-cache directory, not a symlink')
    count = total = 0
    for namespace in root.iterdir():
        if namespace.is_symlink() or not namespace.is_dir() or not re.fullmatch(r'[0-9a-f]{64}', namespace.name):
            continue
        for shard in namespace.iterdir():
            if shard.is_symlink() or not shard.is_dir() or not re.fullmatch(r'[0-9a-f]{2}', shard.name):
                continue
            for path in shard.iterdir():
                if (path.is_symlink() or not path.is_file()
                        or not re.fullmatch(r'[0-9a-f]{64}\.npy', path.name)
                        or not path.name.startswith(shard.name)):
                    continue
                size = path.stat().st_size
                path.unlink()
                count += 1
                total += size
                if count % 10000 == 0:
                    print(f'Released {count} generated feature files ({total/GIB:.2f} GiB)', flush=True)
    return count, total
