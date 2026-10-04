"""Read-only reuse of complete official and previously verified full-wave views."""
from collections import Counter
import json
from pathlib import Path

import soundfile as sf

from rtc_noisy_v2.plan import digest_json
from rtc_noisy.common import assert_noise_disjoint
from w2v_aasist.data import read_protocol, safe_audio_path
from w2v_aasist.runtime import sha256
from w2v_v33.cache import read_index
from w2v_v33.data import canonical_sources, resolve_pair_manifest
from w2v_v35.cache import FORMAT as DEV_FORMAT, assignments
from w2v_v35 import cache as dev_cache


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _audio(path, digest=None, samples=None):
    path = Path(path).resolve()
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1 or info.frames < 1:
        raise ValueError('Expected nonempty full 16 kHz mono audio: '+str(path))
    if samples is not None and info.frames != samples:
        raise ValueError('Full waveform duration changed: '+str(path))
    actual = sha256(path)
    if digest is not None and actual != digest:
        raise ValueError('Audio SHA256 changed: '+str(path))
    stat = path.stat()
    return dict(audio=str(path), audio_sha256=actual, audio_size=stat.st_size,
                audio_mtime_ns=stat.st_mtime_ns, output_samples=info.frames)


def _fixed_dev(cfg, official):
    """Do not call V3.5 prepare/generate, even when an existing file is missing."""
    root = Path(cfg['full_dev_cache_root']).expanduser().resolve()
    folder = root/'epoch_000'
    owner, recipe, inventory = (_json(root/name) for name in ('owner.json','recipe.json','sources.json'))
    generation, complete, manifest = (_json(folder/name) for name in ('generation.json','complete.json','manifest.json'))
    if owner != dict(format=DEV_FORMAT, role='dev', root=str(root)):
        raise ValueError('Fixed Dev cache owner differs')
    if (recipe.get('format') != DEV_FORMAT or recipe.get('role') != 'dev'
            or recipe.get('protocol_sha256') != sha256(cfg['dev_protocol'])
            or recipe.get('source_digest') != digest_json(inventory)
            or recipe.get('full_length') is not True or recipe.get('views_per_source') != 2
            or recipe.get('subtype') != 'FLOAT'
            or recipe.get('generator_sha256') != sha256(dev_cache.__file__)
            or recipe.get('seed') != cfg.get('dev_seed',935017)):
        raise ValueError('Fixed Dev cache recipe differs')
    recipe_hash = sha256(root/'recipe.json')
    if generation != dict(format=DEV_FORMAT, role='dev', epoch=0, recipe_sha256=recipe_hash, root=str(root)):
        raise ValueError('Fixed Dev generation differs')
    by_id = {r['id']:r for r in official}
    offline = {k:r for k,r in by_id.items() if r['domain']=='offline'}
    if (set(inventory) != set(by_id) or set(manifest) != {'seen','heldout'}
            or complete != dict(manifest_sha256=sha256(folder/'manifest.json'),
                                sources=len(offline),views=2*len(offline),full_length=True)):
        raise ValueError('Fixed Dev cache is incomplete or has different sources')
    source_meta = {}
    for key, row in by_id.items():
        saved = inventory[key]
        if any(saved.get(name) != row[name] for name in ('label','language','domain')):
            raise ValueError('Fixed Dev source labels differ')
        if Path(saved['audio']).resolve() != Path(row['audio']).resolve():
            raise ValueError('Fixed Dev source path differs')
        source_meta[key] = _audio(row['audio'],saved['sha256'],saved['samples'])
    choices = assignments(list(offline.values()),recipe['seed'],0,'dev')
    result=[]
    for condition in ('seen','heldout'):
        seen=set()
        for row in manifest[condition]:
            key=row['id']
            if key not in offline or key in seen or row.get('condition') != condition:
                raise ValueError('Duplicate or unknown fixed Dev view')
            expected=offline[key]
            path=Path(row['audio']).resolve()
            path.relative_to(folder)
            saved=_json(path.with_suffix('.json'))
            choice=choices[key][0 if condition=='seen' else 1]
            expected_identity=dict(format=DEV_FORMAT,epoch=0,role='dev',source=key,
                source_sha256=source_meta[key]['audio_sha256'],recipe_sha256=recipe_hash,assignment=choice)
            if (any(saved.get(k)!=v for k,v in expected_identity.items())
                    or safe_audio_path(folder,saved['audio']) != path
                    or any(row.get(k)!=expected[k] for k in ('label','language'))
                    or row.get('source_id') != key or row.get('full_length') is not True
                    or row.get('output_samples') != source_meta[key]['output_samples']
                    or row.get('band') != choice['band'] or row.get('processing_family') != choice['family']):
                raise ValueError('Fixed Dev view identity/processing differs')
            metadata=_audio(path,saved['audio_sha256'],source_meta[key]['output_samples'])
            if sf.info(path).subtype!='FLOAT': raise ValueError('Fixed Dev must preserve FLOAT full-wave cache')
            result.append(dict(row,**metadata,source_id=key,group_id=source_meta[key]['audio_sha256'],
                source_sha256=source_meta[key]['audio_sha256'],condition=condition,split='dev',view='full',noisy=True))
            seen.add(key)
        if seen != set(offline): raise ValueError('Fixed Dev condition does not cover every source')
    files=[root/name for name in ('owner.json','recipe.json','sources.json')]
    files += [folder/name for name in ('generation.json','manifest.json','complete.json')]
    return result, source_meta, files


def build_records(source_cfg, dev_cfg):
    """Reuse Train V3.3 full cache and Dev V3.5 fixed cache without mutations."""
    source_cfg=dict(source_cfg)
    # This is the field emitted by w2v_v33.config.configuration and saved in
    # its selected checkpoint. Never infer a different cache from a directory.
    cache_path=source_cfg.get('train_noisy_cache_v33')
    if not isinstance(cache_path,str) or not cache_path.strip():
        raise ValueError('Selected V3.3 source config is missing train_noisy_cache_v33; '
                         'use the completed V3.3 run configuration')
    cache=Path(cache_path).expanduser().resolve()
    train=read_protocol(source_cfg['train_protocol'],source_cfg['train_data_path'])
    official_dev=read_protocol(dev_cfg['dev_protocol'],dev_cfg['dev_data_path'])
    if Path(source_cfg['train_data_path']).resolve()==Path(dev_cfg['dev_data_path']).resolve():
        raise ValueError('Train and Dev roots must differ')
    pair_file=resolve_pair_manifest(source_cfg)
    sources=canonical_sources(train,pair_file)
    noisy,(raw,train_recipe)=read_index(cache,source_cfg,train,verify_audio_hash=True)
    assert_noise_disjoint({'noise':train_recipe['noise']},
                         _json(Path(dev_cfg['full_dev_cache_root'])/'recipe.json'))
    noisy_lookup={(r['source_id'],r['condition']):r for r in noisy}
    raw_lookup={(r['source'],r['condition']):r for r in raw}
    inventory=_json(cache/'sources.json')
    result=[]; actual_train={}; group_labels={}
    for source in sources:
        key=source['source_id']; group=inventory[key]['sha256']
        # Identical source content must be held out together, even across IDs.
        if group in group_labels and group_labels[group] != source['label']:
            raise ValueError('Identical Train source has conflicting real/fake labels')
        group_labels[group]=source['label']
        for condition in ('offline','online','noisy_a','noisy_b'):
            if condition=='online' and condition not in source['conditions']: continue
            if condition.startswith('noisy_'):
                row=noisy_lookup[(key,condition)]; saved=raw_lookup[(key,condition)]
                # read_index has already verified these bytes and full duration.
                path=Path(row['audio']).resolve(); stat=path.stat()
                metadata=dict(audio=str(path),audio_sha256=saved['audio_sha256'],audio_size=stat.st_size,
                    audio_mtime_ns=stat.st_mtime_ns,output_samples=saved['output_samples'])
            else:
                row=source['conditions'][condition]
                metadata=_audio(row['audio'],group if condition=='offline' else None)
                actual_train[row['id']]=metadata['audio_sha256']
            result.append(dict(row,**metadata,source_id=key,group_id=group,source_sha256=group,
                condition=condition,split='train',view='full',full_length=True,noisy=condition.startswith('noisy_')))
    noisy_dev,dev_meta,dev_files=_fixed_dev(dev_cfg,official_dev)
    if set(actual_train.values()) & {r['audio_sha256'] for r in dev_meta.values()}:
        raise ValueError('Identical official waveform occurs in Train and Dev')
    online=[]
    for row in official_dev:
        if row['domain']!='online': continue
        metadata=dev_meta[row['id']]
        online.append(dict(row,**metadata,source_id=row['id'],group_id=metadata['audio_sha256'],
            source_sha256=metadata['audio_sha256'],condition='online',split='dev',view='full',full_length=True,noisy=False))
    if not online: raise ValueError('Fixed Dev needs official Online sources')
    files=[source_cfg['train_protocol'],dev_cfg['dev_protocol'],pair_file]
    files += [cache/name for name in ('config.json','complete.json','manifest.jsonl','sources.json','assignments.json')]
    files += dev_files
    processor=Path(source_cfg['ssl_path'])/'preprocessor_config.json'
    files.append(processor)
    fingerprints={str(Path(p).resolve()):sha256(p) for p in files}
    coverage=dict(train_sources=len(sources),train_rows=len(result),dev_rows=len(online)+len(noisy_dev),
        missing_online=sum(s['missing_online'] for s in sources),train_unique_source_hashes=len(group_labels),
        train_conditions=dict(Counter(r['condition'] for r in result)),
        train_source_groups=dict(Counter(f'{s["language"]}/{s["label"]}' for s in sources)),
        dev_conditions=dict(Counter(r['condition'] for r in online+noisy_dev)),
        input_policy='full utterance; existing audio only; no crop, augmentation, or cache generation')
    return dict(train=result,dev=online+noisy_dev,fingerprints=fingerprints,coverage=coverage)
