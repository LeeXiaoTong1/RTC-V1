"""Reuse full-wave manifests only; never import old model or optimizer states."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from .common import GROUPS, read_json, digest, seed_for


def find_manifest(source, split):
    source = Path(source)
    local = source/'features'/split
    if not (local/'complete.json').is_file():
        local = Path(read_json(source/'feature_reuse.json')[split]['path'])
    rows = read_json(local/'rows.json'); owner = read_json(local/'owner.json'); complete = read_json(local/'complete.json')
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (complete['files']['rows.json'] != digest(local/'rows.json') or
        owner['records_digest'] != signature or len(rows) != owner['shape'][0] or
        owner['identity']['split'] != split or complete['owner_sha256'] != digest(local/'owner.json')):
        raise ValueError('Canonical manifest is incomplete or altered: '+str(local))
    return rows, {str(local/n): digest(local/n) for n in ('rows.json','owner.json','complete.json')}


def inherited(cfg, key):
    for sub in (cfg, cfg.get('source_config', {}), cfg.get('dev_config', {})):
        if sub.get(key): return sub[key]
    raise ValueError('Recorded data configuration lacks '+key)


def data_inputs(source):
    source = Path(source).expanduser().resolve()
    original = read_json(source/'config.json')
    if original.get('version') not in ('3.15','3.15.1','3.16','3.16.1','3.17'):
        raise ValueError('--data-run must reference a V3.15–V3.17 data configuration')
    old = original.get('continuation_source_config', original)
    train, a = find_manifest(old['v37_run'], 'train')
    dev, b = find_manifest(old['v37_run'], 'dev')
    if (source/'dev_rows.json').is_file() and read_json(source/'dev_rows.json') != dev:
        raise ValueError('Historical fixed Dev manifest differs')
    if {r['group_id'] for r in train} & {r['group_id'] for r in dev}:
        raise ValueError('Train/Dev source overlap')
    train = [r for r in train if r['condition'] in ('offline','online')]
    for rows, split in ((train,'train'), (dev,'dev')):
        if any(r['split'] != split or r['label'] not in (0,1) or r['language'] not in ('en','zh') or not r.get('full_length') for r in rows):
            raise ValueError('Invalid full-wave split metadata')
    if {r['condition'] for r in dev} != {'online','seen','heldout'}:
        raise ValueError('Require historical Online/Seen/Heldout fixed Dev')
    # Compare against the actual official protocol; unpaired Online cannot vanish.
    from w2v_aasist.data import read_protocol
    official = read_protocol(inherited(old,'train_protocol'), inherited(old,'train_data_path'))
    expected = {(r['id'],r['language'],r['label']) for r in official}
    actual = {(r['id'],r['language'],r['label']) for r in train}
    if actual != expected:
        raise ValueError(f'Canonical Train does not cover official protocol: missing={len(expected-actual)} extra={len(actual-expected)}')
    files = {**a, **b, str(source/'config.json'): digest(source/'config.json')}
    for key in ('train_protocol','dev_protocol'):
        p = inherited(old,key); files[p] = digest(p)
    return old, train, dev, files


def dev_partition(rows, pair_path, seed, fraction=.2):
    """Keep duplicates, both noisy versions AND official Online partners together."""
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    by_stem = defaultdict(set)
    for r in rows:
        by_stem[Path(r['source_id']).name].add(r['group_id']); find(r['group_id'])
    for hashes in by_stem.values():
        if len(hashes) != 1: raise ValueError('Ambiguous Dev source basename')
    with Path(pair_path).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            a,b=Path(r['offline_id']).name,Path(r['online_id']).name
            if b not in by_stem: continue
            if a not in by_stem: raise ValueError('Dev Online pair has no original source')
            union(next(iter(by_stem[a])),next(iter(by_stem[b])))
    strata = defaultdict(set); labels = {}
    for r in rows:
        group = find(r['group_id']); key=(r['language'],r['label'])
        if group in labels and labels[group]!=key: raise ValueError('Paired Dev labels disagree')
        labels[group]=key; strata[key].add(group)
    calibration=set()
    for key, values in sorted(strata.items()):
        ordered=sorted(values,key=lambda g:seed_for(seed,'dev-calibration',key,g))
        count=max(1,round(len(ordered)*fraction))
        if count>=len(ordered): raise ValueError('Too few Dev sources to separate calibration')
        calibration.update(ordered[:count])
    result={'select':[], 'calibration':[]}
    for i,r in enumerate(rows): result['calibration' if find(r['group_id']) in calibration else 'select'].append(i)
    for part,indices in result.items():
        counts=Counter((rows[i]['condition'],rows[i]['language'],rows[i]['label']) for i in indices)
        if len(counts)!=12: raise ValueError('Dev partition lacks a condition/language/class cell: '+part)
    result.update(source_keys=[find(r['group_id']) for r in rows],pair_sha256=digest(pair_path),
                  note='Full Dev retained for historical reporting; select excludes calibration sources and their official Online partners')
    return result
