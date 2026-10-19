"""Reuse immutable Train preparation; never import retained Dev representations."""
from pathlib import Path
import shutil
import numpy as np
from .common import read_json, digest


def phone_records(rows, output, manifest):
    if not manifest:return 0
    from w2v_v3187.phonemes import record_name, validate_record, SCHEMA, MODEL_ID, MODEL_REVISION, ALGORITHM
    manifest=Path(manifest);data=read_json(manifest)
    if (not data.get('complete') or data.get('schema')!=SCHEMA or data.get('model_id')!=MODEL_ID or
        data.get('model_revision')!=MODEL_REVISION or data.get('algorithm')!=ALGORITHM):
        raise ValueError('Incompatible phoneme reuse manifest')
    reused=0
    for row in rows:
        entry=data['records'].get(row['id'])
        if not entry:continue
        relative=record_name(row)
        if entry['path']!=relative:raise ValueError('Parent phoneme source lineage changed')
        source=manifest.parent/relative;target=Path(output)/relative
        if digest(source)!=entry['sha256']:raise ValueError('Parent phoneme record changed')
        validate_record(read_json(source),row)
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);reused+=1
    print(f'V4 phoneme metadata reused: {reused}; only new originals need recognition',flush=True)
    return reused


def teacher_cache(rows, proof, parent, manifest):
    if not parent and not manifest:return None
    if not parent or not manifest:raise ValueError('XEUS reuse requires both parent rows and manifest')
    from w2v_v31813.prior import bind, originals
    manifest=Path(manifest);old_rows=read_json(Path(parent)/'train_rows.json')
    bind(manifest,old_rows)
    data=read_json(manifest);owner=data['owner']
    if owner['teacher']!=proof or owner['clusters']!=128 or owner['chunk_samples']!=320000 or owner['context_samples']!=16000:
        raise ValueError('XEUS teacher/feature policy changed; cannot reuse histograms')
    current={(r['canonical_source'],r['id'],r['audio_sha256']) for r in originals(rows)}
    previous=originals(old_rows)
    keys=[(r['canonical_source'],r['id'],r['audio_sha256']) for r in previous]
    if not set(keys)<=current:raise ValueError('V4 expanded Train no longer contains parent original sources')
    arrays=[]
    for file in data['files']:
        if file['file'].startswith('hist_'):
            with np.load(manifest.parent/file['file'],allow_pickle=False) as archive:
                arrays.append(archive['histograms'].copy())
    h=np.concatenate(arrays)
    if h.shape!=(len(keys),128):raise ValueError('Parent XEUS source order/shape changed')
    with np.load(manifest.parent/'centers.npz',allow_pickle=False) as archive:
        centers=archive['centers'].copy()
    print(f'V4 XEUS reuse: {len(keys)} sources; frozen Train codebook; extract new sources only',flush=True)
    return dict(centers=centers,histograms=dict(zip(keys,h)),manifest_sha256=digest(manifest))
