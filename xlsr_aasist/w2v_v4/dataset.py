"""Transfer whole verified Dev EN-real sources; retain fixed select/audit sources."""
from collections import defaultdict, Counter
import copy
import math
from pathlib import Path
import soundfile as sf
from w2v_aasist.data import read_protocol
from w2v_v31810.identity import build_registry, registry_config, verify_registry, read_official_pairs
from w2v_v31810.validation import _AudioIdentity, _digest
from w2v_v31813.config import verify as verify_parent
from w2v_v31813.evaluate import selected
from .common import read_json, atomic_json, fingerprint, digest
from .source import resolve_parent, resolve_data_run


def choose_sources(dev, protocol, fraction, seed):
    if not 0 <= fraction < 1:
        raise ValueError('Dev EN-real transfer fraction must be >=0 and <1; retain validation')
    groups = defaultdict(list)
    for row in dev:
        groups[row['canonical_source']].append(row)
    moved, held = set(), set()
    for split in ('select','audit'):
        available = set(protocol[split]['speech_sources'])
        for paired in (True,False):
            eligible = [source for source,rows in groups.items() if source in available and
                all((r['language'],r['label'])==('en',1) for r in rows) and
                (any(r['condition']=='online' for r in rows)==paired)]
            order = sorted(eligible,key=lambda s:fingerprint((seed,'v4-fixed-transfer',split,paired,s)))
            # Keep at least one source in every existing availability stratum.
            retain = max(1,math.ceil(len(order)*(1-fraction))) if order else 0
            held.update(order[:retain]);moved.update(order[retain:])
    if not held:
        raise ValueError('No held-out Dev EN-real sources')
    return moved, held


def filtered_protocol(parent, moved):
    out = copy.deepcopy(parent)
    for split in ('select','audit'):
        part = out[split]
        keep = [i for i,r in enumerate(part['rows']) if r['canonical_source'] not in moved]
        index = {old:new for new,old in enumerate(keep)}
        part['rows'] = [part['rows'][i] for i in keep]
        part['tickets'] = [dict(part['tickets'][i],index=j) for j,i in enumerate(keep)]
        part['panels'] = {p:[index[i] for i in ids if i in index] for p,ids in part['panels'].items()}
        part['speech_sources'] = sorted(set(part['speech_sources'])-moved)
        for panel,ids in part['panels'].items():
            if {(part['rows'][i]['language'],part['rows'][i]['label']) for i in ids} != {('en',0),('en',1),('zh',0),('zh',1)}:
                raise ValueError(f'Transfer empties a validation group: {split}/{panel}')
    return out


def transfer(train, dev, protocol, registry, cfg, fraction=.75, seed=4001):
    verify_registry(train,registry,cfg,protocol)
    if len({r['id'] for r in dev}) != len(dev):raise ValueError('Duplicate official Dev ID')
    known = {r['id']:r for r in registry['dev']}
    if set(known) != {r['id'] for r in dev}:raise ValueError('Official Dev inventory changed')
    for r in dev:
        if any(r.get(k)!=known[r['id']].get(k) for k in ('audio_sha256','decoded_pcm_sha256','canonical_source','label','language')):
            raise ValueError('Official Dev identity changed')
    moved,held = choose_sources(dev,protocol,fraction,seed)
    rows = copy.deepcopy(train) + [dict(r,split='train',v4_origin_split='dev') for r in dev if r['canonical_source'] in moved]
    remaining = [dict(r) for r in dev if r['canonical_source'] not in moved]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Transferred IDs collide with Train')
    # The parent registry already unified official pairs and exact PCM. Check
    # its complete content relation again before binding the new training side.
    content = {}
    for r in rows+remaining:
        for field in ('audio_sha256','decoded_pcm_sha256'):
            key=field,r[field]
            if key in content and content[key]!=r['canonical_source']:
                raise ValueError('Identical audio assigned to different canonical sources')
            content[key]=r['canonical_source']
    if {r['canonical_source'] for r in rows} & {r['canonical_source'] for r in remaining}:
        raise ValueError('Train/Dev source overlap after transfer')
    counts = lambda items: dict(Counter(r['language']+'/'+str(r['label'])+'/'+r['condition'] for r in items))
    report = dict(schema='v4_source_transfer_v1',seed=seed,requested_train_fraction=fraction,
        moved_sources=sorted(moved),retained_en_real_sources=sorted(held),moved_rows=counts([r for r in rows if r.get('v4_origin_split')=='dev']),
        train_rows=counts(rows),remaining_dev_rows=counts(remaining),
        parent_registry_hash=registry['registry_hash'],selection_used_to_choose_sources=False,
        policy='Whole canonical sources, including official pair and exact-PCM relations; source-hash order, no model scores',
        parent_dev_is_historical=True)
    new_registry = build_registry(rows,remaining,dict(train=sum(r['condition']=='online' for r in rows),
        dev=sum(r['condition']=='online' for r in remaining)),dict(parent=registry.get('pairing_audit'),transfer=report))
    out = filtered_protocol(protocol,moved)
    out.update(schema='rtc_v4_locked_validation_v1',source_registry_hash=new_registry['registry_hash'],
        synthesis_seed=cfg['seed'],
        source_transfer_hash=fingerprint(report),selection_aggregation='equal EN/ZH macro-F1 within each panel; 0.30/0.35/0.35 panels',
        old_select_scores_comparable=False)
    out.pop('protocol_hash',None);out['protocol_hash']=_digest(out)
    bound=dict(cfg,source_registry=registry_config(new_registry))
    verify_registry(rows,new_registry,bound,out)
    return rows,remaining,out,new_registry,report


def official_rows(cfg, registry):
    raw = read_protocol(cfg['official_dev_protocol'],cfg['official_dev_root'])
    known = {r['id']:r for r in registry['dev']}
    if set(known)!={r['id'] for r in raw}:raise ValueError('Official Dev protocol membership changed')
    identity=_AudioIdentity();rows=[]
    for r in raw:
        old=known[r['id']];info=identity.audio(r['audio'],old['audio_sha256'])
        if info['decoded_pcm_sha256']!=old['decoded_pcm_sha256']:raise ValueError('Official Dev PCM changed')
        audio=sf.info(r['audio'])
        if audio.samplerate!=16000 or audio.channels!=1:raise ValueError('Original PCL/XEUS requires official 16 kHz mono')
        rows.append(dict(r,canonical_source=old['canonical_source'],condition=r['domain'],split='dev',
            audio_sha256=info['sha256'],decoded_pcm_sha256=info['decoded_pcm_sha256'],
            audio_size=info['size'],audio_mtime_ns=info['mtime_ns'],output_samples=int(audio.frames),full_length=True))
    by_id={r['id']:r for r in rows};by_name=defaultdict(list)
    for r in rows:by_name[Path(r['id']).name].append(r['id'])
    def resolve(value):
        if value in by_id:return value
        matches=by_name[Path(value).name]
        if len(matches)!=1:raise ValueError('Ambiguous Dev official pair ID: '+value)
        return matches[0]
    off_for={r['id']:r['id'] for r in rows if r['condition']=='offline'}
    for off,on in read_official_pairs([],cfg['dev_pairs']):
        if not on:continue
        a,b=resolve(off),resolve(on)
        if by_id[a]['condition']!='offline' or by_id[b]['condition']!='online' or by_id[a]['canonical_source']!=by_id[b]['canonical_source']:
            raise ValueError('Official Dev pair disagrees with registered content')
        if b in off_for and off_for[b]!=a:raise ValueError('Conflicting official pair')
        off_for[b]=a
    for r in rows:
        if r['id'] not in off_for:raise ValueError('Official Online lacks its recorded Offline pair')
        off=by_id[off_for[r['id']]]
        r.update(source_id=off['id'],group_id=off['audio_sha256'],source_sha256=off['audio_sha256'])
    return rows


def prepare(args):
    parent=resolve_parent(args.parent_run)
    old=verify_parent(parent);_,_,done=selected(parent)
    owner=dict(parent=str(parent),parent_config=digest(parent/'config.json'),parent_checkpoint=done['checkpoint_sha256'],
        fraction=args.dev_real_train_fraction,split_seed=args.split_seed)
    root=Path(args.assets_root).resolve()/fingerprint(owner)[:16]
    if (root/'config.json').is_file():
        resolve_data_run(root)
        if read_json(root/'config.json')['data_owner']!=owner:raise ValueError('Prepared data owner changed')
        return root,old
    registry=read_json(parent/'source_registry.json');train=read_json(parent/'train_rows.json')
    result=transfer(train,official_rows(old,registry),read_json(parent/'validation_protocol.json'),registry,old,
        args.dev_real_train_fraction,args.split_seed)
    rows,dev,protocol,registry,report=result
    root.mkdir(parents=True,exist_ok=True)
    files=dict(zip(('train_rows.json','dev_rows.json','validation_protocol.json','source_registry.json','source_transfer.json'),result))
    for name,value in files.items():atomic_json(root/name,value)
    cfg=dict(old,data_recipe='v4_source_transfer_v1',data_owner=owner,source_registry=registry_config(registry),
        validation_protocol_hash=fingerprint(protocol),validation_input_files=protocol['input_files'],
        prepared_files={name:digest(root/name) for name in files})
    atomic_json(root/'config.json',cfg)
    print(f'V4 sources: moved {len(report["moved_sources"])} Dev EN-real; retained {len(report["retained_en_real_sources"])}',flush=True)
    return root,old
