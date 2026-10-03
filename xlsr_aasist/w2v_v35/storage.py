"""Disk-peak planning and explicitly bounded compatibility for the V3.5 storage fix."""
import json
import os
from pathlib import Path
import shutil

from w2v_aasist.runtime import atomic_json, storage_size

GIB = 1024**3
MARGIN = 512*1024**2
ORIGINAL_TRAIN_SHA256 = '4ab20ba05ee0fa593fb0c0d3171aa8a73f3fc17648c65b6ec9703c4690d46a20'
PATCHED_TRAIN_SHA256 = '7442ce4764b0ee419dc3229f44124a4c3981c8f426d3bde8c420ef1fa17cbd49'


def compatible_code(saved, current):
    if saved == current:
        return True
    old = {k.replace('\\','/'):v for k,v in saved.items()}
    new = {k.replace('\\','/'):v for k,v in current.items()}
    added = set(new)-set(old)
    changed = {k for k in old.keys() & new.keys() if old[k] != new[k]}
    return (not set(old)-set(new) and added == {'w2v_v35/storage.py'}
            and changed == {'w2v_v35/train.py'}
            and old.get('w2v_v35/train.py') == ORIGINAL_TRAIN_SHA256
            and new.get('w2v_v35/train.py') == PATCHED_TRAIN_SHA256)


def checkpoint_peak_bytes(model, ema):
    """Free space needed in addition to the existing checkpoint files.

    Reserve a complete replacement last.pt (raw model + Adam's two moments +
    EMA), plus two possible newly promoted EMA weight files. Old files remain
    until the commit succeeds. Projection/frozen parameters have no Adam state.
    """
    weights = storage_size(model.state_dict())
    trainable = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    last = weights + 2*trainable + storage_size(ema.state_dict())
    # Adam scalar steps and serialization metadata are covered by the margins.
    return last + 2*weights + 3*MARGIN


def state_peak_bytes(state):
    return storage_size(state) + 2*storage_size(state['model']) + 3*MARGIN


def generation_inventory(cfg, epoch):
    """Upper estimate for full FLOAT WAVs, sidecars and allocation overhead."""
    from .cache import _root, FORMAT
    root = _root(cfg,'train')
    owner = json.loads((root/'owner.json').read_text(encoding='utf-8'))
    if owner != {'format':FORMAT,'role':'train','root':str(root)}:
        raise ValueError('Unrecognized training cache owner')
    inventory = json.loads((root/'sources.json').read_text(encoding='utf-8'))
    expected = sum(int(row['samples'])*8 + 32768 for row in inventory.values() if row['domain']=='offline')
    folder = root/f'epoch_{epoch:03d}'
    allocated = 0
    if folder.exists():
        if folder.is_symlink() or folder.resolve().parent != root:
            raise ValueError('Unsafe cache generation path')
        for parent, dirs, files in os.walk(folder,followlinks=False):
            for name in dirs+files:
                path=Path(parent)/name
                if path.is_symlink() or folder.resolve() not in path.resolve().parents:
                    raise ValueError('Cache generation contains unsafe paths')
            for name in files:
                stat=(Path(parent)/name).stat()
                allocated += getattr(stat,'st_blocks',0)*512 if hasattr(stat,'st_blocks') else stat.st_size
    return root,dict(expected_bytes=expected,existing_bytes=allocated,
                     missing_bytes=max(0,expected-allocated),epoch=epoch)


def check_space(cfg, run, epoch, checkpoint_reserve):
    """Fail before expensive training when predicted cache + save peak will not fit."""
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    root,cache = generation_inventory(cfg,epoch) if cfg.get('train_cache_root') else (run,{'missing_bytes':0,'epoch':epoch})
    same_volume = root.stat().st_dev == run.stat().st_dev
    floor=max(int(cfg.get('cache_free_floor_bytes',GIB)),GIB)
    checkpoint_need=checkpoint_reserve+floor+(cache['missing_bytes'] if same_volume else 0)
    cache_need=checkpoint_need if same_volume else cache['missing_bytes']+floor
    result=dict(epoch=epoch,checkpoint_free_bytes=shutil.disk_usage(run).free,
        checkpoint_required_bytes=checkpoint_need,cache_free_bytes=shutil.disk_usage(root).free,
        cache_required_bytes=cache_need,same_volume=same_volume,cache=cache)
    atomic_json(run/'storage_budget.json',result)
    print(f'V35_STORAGE epoch={epoch} checkpoint_free_GiB={result["checkpoint_free_bytes"]/GIB:.2f} '
          f'required_GiB={checkpoint_need/GIB:.2f} cache_missing_GiB={cache["missing_bytes"]/GIB:.2f}',flush=True)
    if result['checkpoint_free_bytes'] < checkpoint_need or result['cache_free_bytes'] < cache_need:
        raise OSError('Insufficient projected V3.5 disk space before training: '
            f'checkpoint volume needs {checkpoint_need/GIB:.2f} GiB free, has {result["checkpoint_free_bytes"]/GIB:.2f}; '
            f'cache volume needs {cache_need/GIB:.2f} GiB, has {result["cache_free_bytes"]/GIB:.2f}. '
            'Existing checkpoints and current-epoch cache are preserved; see storage_budget.json.')
    return result
