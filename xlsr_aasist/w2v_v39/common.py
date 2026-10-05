"""Small atomic artifacts; no large training checkpoints or audio caches."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def save_small(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 32 * 1024**2:
        raise OSError('Need 32 MiB free for compact V3.9 state; no files were deleted')
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_files(values):
    for path, expected in values.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError('Pinned input changed or is missing: ' + path)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def announce(message):
    from live_progress import phase
    print('[Phase] ' + message, flush=True)
    phase(message)
