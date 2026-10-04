"""Small, resumable final-classifier inputs; no frame features are saved."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from w2v_aasist.data import AudioDataset, FeatureCollator, worker_init
from w2v_aasist.launch import run_lock
from w2v_aasist.progress import progress
from w2v_aasist.runtime import atomic_json, sha256
from w2v_v3.model import microbatches

FORMAT='rtc_v36_frozen_vectors_v1'
MODE='eval-fp32-exact-length-full-wave-final-linear-input'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


class NumpyCollator:
    """No torch storage file descriptors are sent across the worker queue."""
    def __init__(self, ssl_path): self.collator=FeatureCollator(ssl_path)

    def __call__(self, rows):
        examples=self.collator(rows)
        return [dict(row,features=row['features'].numpy(),mask=row['mask'].numpy()) for row in examples]


class VerifiedAudioDataset(AudioDataset):
    """Reject files replaced while a long extraction is consuming its inventory."""
    def __getitem__(self,index):
        row=self.records[index]
        def verify():
            if 'audio_size' not in row:return
            stat=Path(row['audio']).stat()
            if stat.st_size!=row['audio_size'] or stat.st_mtime_ns!=row['audio_mtime_ns']:
                raise ValueError('Audio changed during frozen feature extraction: '+row['audio'])
        verify()
        output=super().__getitem__(index)
        verify()
        return output


def feature_loader(records,cfg):
    dataset=VerifiedAudioDataset(records,training=False,epoch=0,seed=cfg.get('seed',1234),
                         max_seconds=0.,rawboost=0,raw_config={})
    workers=int(cfg.get('feature_workers',cfg.get('workers',2)))
    if workers<0: raise ValueError('Feature workers must be nonnegative')
    batch=int(cfg.get('feature_batch',cfg.get('eval_batch',8)))
    if batch<1: raise ValueError('Feature batch must be positive')
    kwargs=dict(dataset=dataset,collate_fn=NumpyCollator(cfg['ssl_path']),num_workers=workers,
        pin_memory=False,batch_size=batch,shuffle=False,
        generator=torch.Generator().manual_seed(cfg.get('seed',1234)))
    if workers:
        kwargs.update(multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=1,
                      persistent_workers=False)
    return DataLoader(**kwargs)


def inference_batches(records,cfg):
    """Exact official features using bounded numpy-only worker transport."""
    for batch in feature_loader(records,cfg):
        yield [dict(row,features=torch.from_numpy(row['features']),mask=torch.from_numpy(row['mask']))
               for row in batch]


def _rows(records):
    rows=[]
    for record in records:
        row={k:v for k,v in record.items() if k not in ('wave','features','mask','audibility_mask')}
        path=Path(row['audio']).resolve(); stat=path.stat()
        # build_records already hashes the full files. Bind exact identities and
        # refuse a changed file rather than silently recaching different audio.
        if 'audio_size' in row and (row['audio_size']!=stat.st_size or row['audio_mtime_ns']!=stat.st_mtime_ns):
            raise ValueError('Audio changed after records were built: '+str(path))
        digest=row.get('audio_sha256') or sha256(path)
        row.update(audio=str(path),audio_sha256=digest,audio_size=stat.st_size,audio_mtime_ns=stat.st_mtime_ns)
        if row.get('view','full')!='full': raise ValueError('Only complete full-wave views can be cached')
        rows.append(row)
    if not rows: raise ValueError('Cannot extract an empty split')
    keys=[(r.get('source_id',r['id']),r.get('condition',r.get('domain'))) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate source/condition feature row')
    _digest(rows)  # Require fully serializable, finite metadata before writing.
    return rows


def _chunk_digest(x,logits,start,stop):
    h=hashlib.sha256()
    h.update(np.ascontiguousarray(x[start:stop]).tobytes())
    h.update(np.ascontiguousarray(logits[start:stop]).tobytes())
    return h.hexdigest()


def _validate_partial(out,owner):
    state=_json(out/'cursor.json')
    if state.get('identity_digest')!=owner['identity_digest'] or state.get('rows_sha256')!=sha256(out/'rows.json'):
        raise ValueError('Feature cursor identity or metadata differs')
    n,d=owner['shape']; cursor=state.get('cursor')
    if type(cursor) is not int or not 0<=cursor<=n: raise ValueError('Invalid feature cursor')
    x=np.load(out/'x.npy',mmap_mode='r+'); logits=np.load(out/'logits.npy',mmap_mode='r+')
    if x.shape!=(n,d) or logits.shape!=(n,2) or x.dtype!=np.float32 or logits.dtype!=np.float32:
        raise ValueError('Feature file shape/dtype differs')
    try:
        previous=0
        for chunk in state.get('chunks',[]):
            start,stop=chunk['start'],chunk['stop']
            if start!=previous or not start<stop<=cursor:
                raise ValueError('Feature cursor chunks are incomplete')
            if _chunk_digest(x,logits,start,stop)!=chunk['sha256']:
                raise ValueError('Committed feature cache bytes changed')
            if not np.isfinite(x[start:stop]).all() or not np.isfinite(logits[start:stop]).all():
                raise ValueError('Nonfinite committed feature cache')
            previous=stop
        if previous!=cursor: raise ValueError('Feature cursor is not fully committed')
    except BaseException:
        x._mmap.close();logits._mmap.close()
        raise
    return x,logits,state


def load_cache(out,identity=None):
    """Only a checksummed complete cache can be consumed by classifier fitting."""
    out=Path(out).resolve(); owner=_json(out/'owner.json'); manifest=_json(out/'complete.json')
    if owner.get('format')!=FORMAT or owner.get('mode')!=MODE or manifest.get('format')!=FORMAT:
        raise ValueError('Unknown feature cache format or inference mode')
    if identity is not None and owner.get('identity')!=identity:
        raise ValueError('Feature cache belongs to different base/split identity')
    if owner.get('identity_digest')!=_digest(dict(identity=owner['identity'],records_digest=owner['records_digest'],mode=MODE,
                                                 preprocessing_sha256=owner['preprocessing_sha256'])):
        raise ValueError('Feature cache owner was changed')
    if manifest.get('owner_sha256')!=sha256(out/'owner.json'):
        raise ValueError('Feature completion owner differs')
    expected={'x.npy','logits.npy','rows.json','cursor.json'}
    if set(manifest.get('files',{}))!=expected:
        raise ValueError('Incomplete feature file checksums')
    for name,digest in manifest['files'].items():
        if sha256(out/name)!=digest: raise ValueError('Completed feature file changed: '+name)
    rows=_json(out/'rows.json'); state=_json(out/'cursor.json')
    if _digest(rows)!=owner['records_digest'] or len(rows)!=owner['shape'][0] or state['cursor']!=len(rows):
        raise ValueError('Completed feature count/order differs')
    x=np.load(out/'x.npy',mmap_mode='r'); logits=np.load(out/'logits.npy',mmap_mode='r')
    if x.shape!=tuple(owner['shape']) or logits.shape!=(len(rows),2) or x.dtype!=np.float32 or logits.dtype!=np.float32:
        raise ValueError('Completed feature dimensions differ')
    if not np.isfinite(x).all() or not np.isfinite(logits).all(): raise ValueError('Nonfinite completed features')
    return dict(x=x,logits=logits,rows=rows,manifest=dict(manifest,**{k:owner[k] for k in ('shape','identity','mode','records_digest')}))


def _flush(path,array):
    array.flush()
    with Path(path).open('r+b') as stream: os.fsync(stream.fileno())


def extract_cache(model,records,cfg,out,identity):
    """One forward per view; an interruption replays only an uncommitted interval."""
    out=Path(out).expanduser().resolve(); rows=_rows(records)
    final=model.head.classifier[-1]
    if not isinstance(final,torch.nn.Linear) or final.out_features!=2:
        raise ValueError('Expected final two-class linear classifier')
    d=int(final.in_features); n=len(rows)
    preprocessing_sha256=sha256(Path(cfg['ssl_path'])/'preprocessor_config.json')
    bound=dict(identity=identity,records_digest=_digest(rows),mode=MODE,preprocessing_sha256=preprocessing_sha256)
    owner=dict(format=FORMAT,**bound,identity_digest=_digest(bound),shape=[n,d],dtype='float32')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out/'.extract.lock'):
        if (out/'owner.json').is_file():
            if _json(out/'owner.json')!=owner: raise ValueError('Refusing changed base/split/records feature identity')
            if (out/'complete.json').is_file(): return load_cache(out,identity)
            x,logits,state=_validate_partial(out,owner)
        else:
            if any(p.name!='.extract.lock' for p in out.iterdir()):
                raise ValueError('Refusing an unowned nonempty feature cache directory')
            metadata_bytes=len(json.dumps(rows).encode())
            required=n*(d+2)*4+metadata_bytes*2+int(cfg.get('feature_free_margin_bytes',256*1024**2))
            free=shutil.disk_usage(out).free
            if free<required: raise OSError(f'Feature cache needs {required/1024**3:.2f} GiB free; no existing data deleted')
            print(f'V36_FEATURE_STORAGE rows={n} dim={d} required_GiB={required/1024**3:.3f} free_GiB={free/1024**3:.3f}',flush=True)
            atomic_json(out/'owner.json',owner); atomic_json(out/'rows.json',rows)
            x=np.lib.format.open_memmap(out/'x.npy',mode='w+',dtype=np.float32,shape=(n,d))
            logits=np.lib.format.open_memmap(out/'logits.npy',mode='w+',dtype=np.float32,shape=(n,2))
            _flush(out/'x.npy',x); _flush(out/'logits.npy',logits)
            state=dict(identity_digest=owner['identity_digest'],rows_sha256=sha256(out/'rows.json'),cursor=0,chunks=[])
            atomic_json(out/'cursor.json',state)
        device=torch.device(cfg.get('device','cpu'))
        model.to(device=device,dtype=torch.float32).eval().requires_grad_(False)
        capture=[]
        def hook(_module,args): capture.append(args[0].detach())
        handle=final.register_forward_pre_hook(hook)
        try:
            cursor=state['cursor']; remaining=rows[cursor:]
            commit_rows=int(cfg.get('feature_commit_rows',256))
            if commit_rows<1: raise ValueError('Feature commit interval must be positive')
            if remaining:
                loader=feature_loader(remaining,cfg)
                label='V3.6 frozen features '+str(identity.get('split','') if isinstance(identity,dict) else '')
                print(f'V36_FEATURE_RESUME completed={cursor}/{n} mode={MODE}',flush=True)
                with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
                    for batch in progress(loader,total=len(loader),label=label,every=100):
                        examples=[dict(r,features=torch.from_numpy(r['features']),mask=torch.from_numpy(r['mask'])) for r in batch]
                        bx=np.empty((len(examples),d),dtype=np.float32); bl=np.empty((len(examples),2),dtype=np.float32)
                        done=set()
                        for indices,features,mask in microbatches(examples,cfg.get('feature_microbatch',cfg.get('microbatch',4)),cfg.get('feature_frame_budget',cfg.get('frame_budget',1600))):
                            capture.clear()
                            output=model(features.to(device),mask.to(device))
                            score=output[0] if isinstance(output,(tuple,list)) else output
                            if len(capture)!=1: raise ValueError('Final classifier must execute exactly once per model forward')
                            hidden=capture[0]
                            replay=torch.nn.functional.linear(hidden,final.weight,final.bias)
                            if hidden.shape!=(len(indices),d) or score.shape!=(len(indices),2):
                                raise ValueError('Unexpected frozen feature or logits shape')
                            if not bool(torch.isfinite(hidden).all() and torch.isfinite(score).all()):
                                raise FloatingPointError('Nonfinite frozen features/logits')
                            torch.testing.assert_close(replay,score,rtol=1e-5,atol=2e-6,
                                msg='Cached final-layer inputs do not reproduce original logits')
                            bx[indices]=hidden.cpu().numpy(); bl[indices]=score.cpu().numpy(); done.update(indices)
                        if done!=set(range(len(examples))): raise ValueError('Missing or duplicate feature microbatch index')
                        stop=cursor+len(examples)
                        if [(e['id'],e.get('condition')) for e in examples] != [(r['id'],r.get('condition')) for r in rows[cursor:stop]]:
                            raise ValueError('Feature loader changed row order')
                        x[cursor:stop]=bx; logits[cursor:stop]=bl
                        cursor=stop
                        if cursor-state['cursor']>=commit_rows or cursor==n:
                            _flush(out/'x.npy',x); _flush(out/'logits.npy',logits)
                            state['chunks'].append(dict(start=state['cursor'],stop=cursor,
                                sha256=_chunk_digest(x,logits,state['cursor'],cursor)))
                            state['cursor']=cursor; atomic_json(out/'cursor.json',state)
            if state['cursor']!=n: raise ValueError('Feature extraction ended before full coverage')
            complete=dict(format=FORMAT,owner_sha256=sha256(out/'owner.json'),cursor=n,
                replay_rtol=1e-5,replay_atol=2e-6,
                files={name:sha256(out/name) for name in ('x.npy','logits.npy','rows.json','cursor.json')})
            atomic_json(out/'complete.json',complete)
        finally:
            handle.remove()
            x._mmap.close();logits._mmap.close()
    return load_cache(out,identity)
