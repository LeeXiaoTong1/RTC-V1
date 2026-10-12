"""Source-isolated, category-aware noise bank with bounded shared LRU storage."""
from collections import OrderedDict, defaultdict
from pathlib import Path
import json
import hashlib
import numpy as np
import soundfile as sf

CATEGORIES=('office','coffee','rain','footsteps','keyboard','echo')
RECORDED=CATEGORIES[:-1]
SCHEMA='rtc_v3184_scene_noise_v1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_records(records):
    sources={s:set() for s in ('train','dev')};hashes={s:set() for s in sources}
    for split in sources:
        rows=records.get(split,[])
        if {r['category'] for r in rows}!=set(RECORDED):
            raise ValueError('Require all five recorded categories in '+split+'; echo is generated')
        for row in rows:
            if row.get('split')!=split or not row.get('original_sources') or not row.get('license'):
                raise ValueError('Noise split/provenance/license missing')
            sources[split].update(row['original_sources'])
            hashes[split].add(row['pcm_sha256'])
    if sources['train']&sources['dev'] or hashes['train']&hashes['dev']:
        raise ValueError('Noise original recording or PCM content overlaps Train/Dev')


def bind_manifest(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('Prepare noises first: bash prepare_noise_v3184.sh --out '+str(path.parent))
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema')!=SCHEMA or manifest.get('status')!='complete':
        raise ValueError('Scene noise preparation is incomplete')
    records={s:[] for s in ('train','dev')};files={str(path):sha256(path)}
    for raw in manifest['records']:
        p=(path.parent/raw['file']).resolve()
        if path.parent not in p.parents:raise ValueError('Noise file escapes its prepared directory')
        if sha256(p)!=raw['sha256']:raise ValueError('Prepared noise changed: '+str(p))
        stat=p.stat();row=dict(raw,path=str(p),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        records[raw['split']].append(row);files[str(p)]=raw['sha256']
    validate_records(records)
    return dict(scene_noise_manifest=str(path),scene_noise_records=records,scene_noise_files=files,
        scene_noise_provenance=manifest['provenance'],scene_noise_summary=manifest['summary'])


class SceneBank:
    def __init__(self,records,cap_bytes=32*1024**2):
        self.groups={c:defaultdict(list) for c in RECORDED}
        for row in records:self.groups[row['category']][row['source_group']].append(row)
        if any(not v for v in self.groups.values()):raise ValueError('Scene bank has a missing category')
        self.cap=cap_bytes;self.cache=OrderedDict();self.used=0

    def sample(self,category,rng):
        groups=self.groups[category];key=list(groups)[int(rng.integers(len(groups)))]
        rows=groups[key];row=rows[int(rng.integers(len(rows)))];path=Path(row['path']);stat=path.stat()
        if stat.st_size!=row['size'] or stat.st_mtime_ns!=row['mtime_ns']:raise ValueError('Scene audio changed during training')
        key=row['sha256']
        if key in self.cache:
            wave=self.cache[key];self.cache.move_to_end(key)
        else:
            wave,sr=sf.read(path,dtype='float32')
            if sr!=16000 or wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():
                raise ValueError('Prepared scene must be finite 16 kHz mono')
            if wave.nbytes<=self.cap:
                while self.cache and self.used+wave.nbytes>self.cap:
                    _,old=self.cache.popitem(last=False);self.used-=old.nbytes
                self.cache[key]=wave;self.used+=wave.nbytes
        return wave,row


def scene_wave(bank,category,length,rng):
    """Use natural whole event sequences, not arbitrary 50 ms noise fragments."""
    out=np.zeros(length,np.float32);cursor=0;hashes=set();sources=set()
    while cursor<length:
        wave,row=bank.sample(category,rng);hashes.add(row['sha256']);sources.update(row['original_sources'])
        fade=min(cursor,160);n=min(len(wave),length-cursor+fade)
        if n<=fade:raise ValueError('Noise recording is too short')
        start=int(rng.integers(len(wave)-n+1));piece=wave[start:start+n]
        if fade:
            weights=np.linspace(0,1,fade,dtype=np.float32)
            out[cursor-fade:cursor]=out[cursor-fade:cursor]*(1-weights)+piece[:fade]*weights
        take=n-fade;out[cursor:cursor+take]=piece[fade:];cursor+=take
    return out,sorted(hashes),sorted(sources)
