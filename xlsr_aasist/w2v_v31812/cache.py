"""Restartable FP32 evidence shards, bound to checkpoint, rows and recipes."""
from pathlib import Path
import numpy as np
import torch
from w2v_v31810.common import atomic_json, read_json, fingerprint, digest
from w2v_v31810.data import Waves, microbatches, close
from w2v_v31810.validation_data import FixedWaves
from w2v_v31810.workers import loader
from .model import evidence

FIELDS = ('id', 'canonical_source', 'language', 'label', 'condition', 'role',
          'validation_panel', 'validation_split', 'occurrence', 'original_samples')


def save_npz(path, **arrays):
    path = Path(path)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez(stream, **arrays)
    path.with_suffix('.tmp').replace(path)


def read_shard(path, expected):
    meta = read_json(Path(path).with_suffix('.json'))
    if meta['identity'] != expected or digest(path) != meta['sha256']:
        raise ValueError('Feature cache identity/content changed: '+str(path))
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in ('global', 'local', 'logits')}
    n = len(meta['rows'])
    for key, width in (('global',256), ('local',384), ('logits',2)):
        if result[key].shape != (n,width) or result[key].dtype != np.float32 or not np.isfinite(result[key]).all():
            raise ValueError('Invalid evidence array: '+key)
    result['rows'] = meta['rows']
    return result


def concatenate(parts):
    if not parts:
        raise ValueError('No feature shards')
    return dict(rows=[r for p in parts for r in p['rows']],
                **{k: np.concatenate([p[k] for p in parts]) for k in ('global','local','logits')})


def read_complete(rows, tickets, cfg, directory, identity, *, fixed=False, stop=None):
    """Read verified completed evidence without loading a model or decoding audio.

    A continuation never silently re-extracts missing/changed evidence. The
    original checkpoint/config identity, recipes, ticket order and every shard
    hash must match, so an interrupted or corrupted cache fails explicitly.
    """
    directory=Path(directory)
    bound=dict(identity=identity,rows=fingerprint(rows),tickets=fingerprint(tickets),
               recipe=fingerprint(cfg),fixed=fixed,dtype='float32',shard_size=64)
    if not (directory/'manifest.json').is_file() or read_json(directory/'manifest.json')!=bound:
        raise ValueError('Completed cache configuration is missing or changed: '+str(directory))
    parts=[]
    for i in range((len(tickets)+63)//64):
        if stop and stop():raise InterruptedError('Stopped while verifying completed features')
        path=directory/f'{i:05d}.npz'
        if not path.is_file() or not path.with_suffix('.json').is_file():
            raise ValueError('Completed cache shard is missing; no re-encoding was started: '+str(path))
        part=read_shard(path,fingerprint([bound,i]))
        if len(part['rows'])!=len(tickets[i*64:(i+1)*64]):
            raise ValueError('Completed cache shard row count changed')
        for row,ticket in zip(part['rows'],tickets[i*64:(i+1)*64]):
            original=rows[ticket['index']]
            for key in ('id','canonical_source','language','label'):
                if row[key]!=original[key]:raise ValueError('Completed cache row metadata changed: '+key)
        parts.append(part)
    result=concatenate(parts)
    print(f'Reused cache {directory.name}: {len(result["rows"])} verified views; no encoder pass',flush=True)
    return result


def extract(parent, rows, tickets, cfg, directory, identity, *, fixed=False, stop=None):
    """FP32 prevents quantization moving fixed-.5 decisions near the boundary.

    Only unfinished shards are re-encoded after Ctrl+C/OOM. No waveform or large
    sequence cache, no second AASIST/Omni pass. Completed shards verify hashes.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    shard_size = 64
    bound = dict(identity=identity, rows=fingerprint(rows), tickets=fingerprint(tickets),
                 recipe=fingerprint(cfg), fixed=fixed, dtype='float32', shard_size=shard_size)
    manifest = directory/'manifest.json'
    if manifest.exists() and read_json(manifest) != bound:
        raise ValueError('Cache configuration changed; use a new run directory')
    atomic_json(manifest, bound)
    pieces = [tickets[i:i+shard_size] for i in range(0,len(tickets),shard_size)]
    pending = []
    for i, piece in enumerate(pieces):
        path = directory/f'{i:05d}.npz'
        key = fingerprint([bound,i])
        if path.is_file() and path.with_suffix('.json').is_file():
            read_shard(path,key)
        else:
            pending.append((i,piece))
    dataset = (FixedWaves if fixed else Waves)(rows,cfg,'dev' if fixed else 'train')
    batches = loader(dataset,cfg,batch_sampler=[t for _,t in pending]) if pending else None
    parent.eval().requires_grad_(False)
    print(f'Cache {directory.name}: {len(tickets)} views, {len(pending)}/{len(pieces)} shards pending',flush=True)
    try:
        with torch.no_grad():
            for (shard,_), units in zip(pending,batches or []):
                if stop and stop():
                    raise InterruptedError('Stop requested; completed evidence shards retained')
                ordered = [r for unit in units for r in unit]
                positions = {id(r):i for i,r in enumerate(ordered)}
                arrays = {k:np.empty((len(ordered),n),np.float32) for k,n in (('global',256),('local',384),('logits',2))}
                for micro in microbatches(units,cfg['microbatch'],cfg['frame_budget']):
                    if stop and stop():
                        raise InterruptedError('Stop requested; unfinished feature shard will be replayed')
                    examples = [r for unit in micro for r in unit]
                    values = evidence(parent,[r['wave'] for r in examples],[r['original_samples'] for r in examples])
                    for key,value in zip(('global','local','logits'),values):
                        arrays[key][[positions[id(r)] for r in examples]] = value.float().cpu().numpy()
                records = [{k:r[k] for k in FIELDS if k in r} for r in ordered]
                path = directory/f'{shard:05d}.npz'
                save_npz(path,**arrays)
                atomic_json(path.with_suffix('.json'),dict(identity=fingerprint([bound,shard]),sha256=digest(path),rows=records))
                print(f'Cache {directory.name} | {min((shard+1)*shard_size,len(tickets))}/{len(tickets)} recordings',flush=True)
    finally:
        if batches is not None: close(batches)
    parts = [read_shard(directory/f'{i:05d}.npz',fingerprint([bound,i])) for i in range(len(pieces))]
    result = concatenate(parts)
    if len(result['rows']) != len(tickets):
        raise ValueError('Feature extraction lost rows')
    for row,ticket in zip(result['rows'],tickets):
        original = rows[ticket['index']]
        if (row['id'],row['canonical_source']) != (original['id'],original['canonical_source']):
            raise ValueError('Feature extraction changed row order')
    return result
