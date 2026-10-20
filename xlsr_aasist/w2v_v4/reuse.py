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
