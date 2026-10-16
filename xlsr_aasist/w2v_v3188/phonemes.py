"""Reuse prepared CTC records; explicitly represent sub-receptive-field audio.

Normal V3.18.6 metadata remains byte-compatible. Originals below400 samples
cannot supply even one real CTC/Omni frontend frame. Their verified metadata
contains no phone intervals; classification can still use short reflection.
"""
from collections import OrderedDict
from pathlib import Path
from w2v_v3186.phonemes import *
from w2v_v3186.phonemes import (
    lineage as _legacy_lineage, validate_record as _legacy_validate, _safe_record,
)

SHORT_REASON='original_shorter_than_400_sample_receptive_field'


def lineage(row):
    samples=row.get('output_samples')
    if type(samples)is not int or samples<1:
        raise ValueError('Require a nonempty full original waveform')
    if samples>=400:return _legacy_lineage(row)
    identity=_legacy_lineage(dict(row,output_samples=400))
    return dict(identity,num_samples=samples)


def inventory(rows):
    result=[lineage(row) for row in rows]
    if not result or len({r['id'] for r in result})!=len(result):
        raise ValueError('Empty/duplicate phoneme source inventory')
    offline={r['id']:r for r in result if r['domain']=='offline'}
    for row in result:
        source=offline.get(row['source_id'])
        if source is None or any(source[k]!=row[k] for k in ('group_id','label','language')):
            raise ValueError('Phoneme inventory lacks verified same-source lineage: '+row['id'])
        if source['audio_sha256']!=row['group_id']:
            raise ValueError('Source group must be the original Offline SHA256')
    return sorted(result,key=lambda row:row['id'])


def record_name(row):
    return 'records/'+digest_json(lineage(row))+'.json'


def validate_record(record,row):
    if row['output_samples']>=400:return _legacy_validate(record,row)
    if (record.get('schema')!=SCHEMA or record.get('identity')!=lineage(row) or
            record.get('model_id')!=MODEL_ID or record.get('model_revision')!=MODEL_REVISION or
            record.get('algorithm')!=ALGORITHM or record.get('segments')!=[] or
            record.get('quality',{}).get('reason')!=SHORT_REASON):
        raise ValueError('Sub-frame originals require verified empty phoneme metadata: '+row['id'])
    return record


def bind_manifest(manifest_path,rows,*,verify_files=True):
    path=Path(manifest_path).expanduser().resolve();manifest=read_json(path)
    expected=inventory(rows)
    if (manifest.get('schema')!=SCHEMA or manifest.get('complete') is not True or
            manifest.get('model_id')!=MODEL_ID or manifest.get('model_revision')!=MODEL_REVISION or
            manifest.get('algorithm')!=ALGORITHM or manifest.get('inventory_sha256')!=digest_json(expected) or
            manifest.get('inventory')!=expected):
        raise ValueError('Phoneme manifest differs from the original Train inventory')
    entries=manifest.get('records',{})
    if set(entries)!={row['id'] for row in expected}:
        raise ValueError('Incomplete phoneme manifest')
    for row in rows:
        entry=entries[row['id']]
        if entry.get('path')!=record_name(row) or len(entry.get('sha256',''))!=64:
            raise ValueError('Phoneme record identity changed')
        if verify_files and not _safe_record(path.parent,entry['path']).is_file():
            raise FileNotFoundError('Missing phoneme metadata for '+row['id'])
    return dict(path=str(path),sha256=digest_file(path),inventory_sha256=manifest['inventory_sha256'],
        model_id=MODEL_ID,model_revision=MODEL_REVISION,records=len(entries),schema=SCHEMA,algorithm=ALGORITHM)


class PhonemeCache:
    def __init__(self,manifest_path,rows,max_records=256,*,binding=None):
        self.binding=bind_manifest(manifest_path,rows,verify_files=binding is None)
        if binding is not None and self.binding!=binding:
            raise ValueError('Saved phoneme provenance differs from worker manifest')
        self.root=Path(self.binding['path']).parent
        self.entries=read_json(self.binding['path'])['records']
        self.rows={r['id']:r for r in rows};self.cache=OrderedDict();self.max_records=max(1,int(max_records))

    def get(self,row):
        key=row['id']
        if key not in self.rows or lineage(row)!=lineage(self.rows[key]):
            raise ValueError('Unknown phoneme source')
        if key not in self.cache:
            entry=self.entries[key];path=_safe_record(self.root,entry['path'])
            if digest_file(path)!=entry['sha256']:
                raise ValueError('Phoneme metadata checksum changed: '+key)
            self.cache[key]=validate_record(read_json(path),row)
            if len(self.cache)>self.max_records:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]
