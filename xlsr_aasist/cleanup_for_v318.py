"""One-pass retirement of reviewed Train payloads and four old resume states.

Best weights, metadata, original audio, noise recordings and fixed Dev survive.
Only explicit files below two owned Train caches and four exact last.pt paths
can be deleted. No recursive directory deletion and no training-code changes.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

from audit_w2v_storage import current_inputs, load
import cleanup_early_checkpoints as early
from w2v_v316.cleanup import idle_lock, safe, sha

TRAIN_CACHE = 'data/rtc_noisy_v33/train'
ROLLING_CACHE = 'exp/w2v_v315_20261008_075905_5663/rolling_pairs'
RESUMES = {**early.TARGETS,
    'w2v_aasist_full_20260930_094306_c614': 'rtc_w2v_aasist_full_v1',
    'w2v_v313_20261006_192941_3608': 'rtc_v313_partial_state_v1'}
DATA_RUN = 'exp/w2v_v316_tfcl_20261009_020301_4d95'


def validate_selected(path, schema):
    if schema in early.TARGETS.values(): return early.validate_best(path, schema)
    if schema == 'rtc_v313_partial_state_v1':
        from w2v_v313.state import load_selected
        state, _ = load_selected(path.parent)
        # The reviewed failed run selected its preserved V3.12 starting model.
        # A different/non-baseline result needs its own review; never guess.
        if state.get('model') is not None:
            raise ValueError('V3.13 selected a trained candidate; retain last for separate review')
        cfg = state['config']
        deps = {cfg[k]: cfg[k+'_sha256'] for k in ('base_checkpoint','starting_checkpoint')}
        for name, digest in deps.items():
            if sha(name) != digest: raise ValueError('V3.13 best dependency changed: '+name)
        return dict(sha256=sha(path),dependencies=deps,selected=state['selected'])
    import torch
    state = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if state.get('schema') != schema or state.get('kind') != 'weights':
        raise ValueError('Not a standalone AASIST weights checkpoint')
    from w2v_aasist.model import Detector
    with torch.device('meta'):
        expected = Detector.load(None,config_dict=state['model_config'],checkpointing=False).state_dict()
    weights = state['model']
    if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in expected):
        raise ValueError('Incomplete AASIST best')
    return dict(sha256=sha(path),schema=schema)


def checkpoint_plan(root, validator=validate_selected):
    eligible, kept, bests = [], [], {}
    for run, schema in RESUMES.items():
        target = safe(root/'exp'/run/'last.pt',root)
        if not target.exists(): continue
        best = safe(target.parent/('best.pt' if schema=='rtc_v313_partial_state_v1' else 'best_model.pt'),root)
        try:
            if target.stat().st_nlink != 1 or list(target.parent.glob('*.previous')):
                raise ValueError('Linked checkpoint or uncommitted promotion')
            print('Checking preserved best: '+str(best),flush=True)
            before = early.stamp(best)
            info = validator(best,schema)
            if early.stamp(best) != before: raise ValueError('Best changed while checking')
            bests[str(best)] = info
            eligible.append(target)
        except Exception as exc:
            kept.append(dict(path=str(target),reason=str(exc)))
    checked = {}
    try:
        # Repeat only if an initially disposable checkpoint is pinned: its own
        # dependencies must then be protected as well (transitive retention).
        while eligible:
            refs, observed = early.references(root,eligible)
            checked.update(observed)
            pinned = [p for p in eligible if refs[p]]
            if not pinned: break
            for p in pinned: kept.append(dict(path=str(p),reason='referenced',references=sorted(refs[p])))
            eligible = [p for p in eligible if p not in pinned]
    except Exception as exc:
        kept.extend(dict(path=str(p),reason='Dependency check incomplete: '+str(exc)) for p in eligible)
        eligible = []
    return [dict(path=str(p),identity=early.stamp(p),kind='checkpoint') for p in eligible], kept, checked, bests


def cache_plan(root, protected):
    candidates, metadata = {}, {}
    def metadata_json(path):
        path = safe(path,root)
        before = sha(path); value = load(path)
        if sha(path) != before: raise ValueError('Cache metadata changed: '+str(path))
        metadata[str(path)] = before
        return value
    def add(path, boundary):
        path = safe(path,root)
        if boundary not in path.parents: raise ValueError('Payload escaped its cache')
        if not path.exists(): return  # already retired; metadata is intentionally retained
        if not path.is_file(): raise ValueError('Payload is not a regular file: '+str(path))
        if any(a in protected for a in (path,*path.parents)):
            raise ValueError('Current V3.18 input cannot be removed: '+str(path))
        candidates[str(path)] = dict(path=str(path),identity=early.stamp(path),kind='cache')
    cache = safe(root/TRAIN_CACHE,root)
    if cache.exists():
        cfg = metadata_json(cache/'config.json')
        complete = metadata_json(cache/'complete.json')
        manifest = safe(cache/'manifest.jsonl',root)
        before = sha(manifest)
        if (cfg.get('format') != 'rtc_v33_full_condition_cache_v1' or cfg.get('role') != 'train'
            or complete.get('config_sha256') != metadata[str(cache/'config.json')]
            or complete.get('manifest_sha256') != before):
            raise ValueError('V3.3 Train cache ownership or manifest changed')
        metadata[str(manifest)] = before
        with manifest.open(encoding='utf-8') as stream:
            for number,line in enumerate(stream,1):
                row = json.loads(line)
                rel = Path(row['audio'])
                if rel.is_absolute() or '..' in rel.parts or len(rel.parts)<2 or rel.parts[0]!='audio' or rel.suffix!='.wav':
                    raise ValueError('Unexpected generated Train path: '+str(rel))
                if row.get('condition') not in ('noisy_a','noisy_b'):
                    raise ValueError('Not a generated noisy Train record')
                add(cache/rel,cache/'audio')
                if number%20000==0: print(f'Checked {number:,} generated Train records',flush=True)
        if sha(manifest) != before: raise ValueError('Train manifest changed during check')
    cache = safe(root/ROLLING_CACHE,root)
    if cache.exists():
        owner = metadata_json(cache/'owner.json')
        cfg = metadata_json(cache.parent/'config.json')
        identity = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if (owner.get('format') != 'rtc_v315_bounded_pairs_v1' or owner.get('identity') != identity
            or owner.get('cap_bytes') != cfg['rolling_cache_bytes']):
            raise ValueError('V3.15 rolling cache ownership changed')
        for path in cache.iterdir():
            if path.name in ('owner.json','.cache.lock'): continue
            if not re.fullmatch(r'[0-9a-f]{64}\.(npz|tmp)',path.name):
                raise ValueError('Unexpected rolling cache content: '+str(path))
            add(path,cache)
    return list(candidates.values()), metadata


def allowed(root,path):
    if path in {root/'exp'/run/'last.pt' for run in RESUMES}: return True
    if root/TRAIN_CACHE/'audio' in path.parents and path.suffix=='.wav': return True
    return path.parent==root/ROLLING_CACHE and bool(re.fullmatch(r'[0-9a-f]{64}\.(npz|tmp)',path.name))


def apply(plan, journal):
    root = Path(plan['root']).resolve()
    for name,h in plan['metadata'].items():
        if sha(name)!=h: raise ValueError('Cache metadata changed; no deletion: '+name)
    for name,identity in plan['checked_files'].items():
        if early.stamp(Path(name))!=tuple(identity): raise ValueError('Checkpoint dependency changed: '+name)
    for name,info in plan['verified_best'].items():
        for dependency,h in {name:info['sha256'],**info.get('dependencies',{})}.items():
            if sha(dependency)!=h: raise ValueError('Best dependency changed: '+dependency)
    for item in plan['candidates']:
        path = safe(item['path'],root)
        if not allowed(root,path) or early.stamp(path)!=tuple(item['identity']):
            raise ValueError('Cleanup target changed or outside allowlist: '+str(path))
    count = total = 0
    # Preserve an incremental receipt even if interrupted halfway through unlink.
    with Path(journal).open('x',encoding='utf-8') as stream:
        for item in plan['candidates']:
            path = safe(item['path'],root)
            if early.stamp(path)!=tuple(item['identity']): raise ValueError('Target changed: '+str(path))
            path.unlink(); count+=1; total+=item['identity'][0]
            stream.write(json.dumps(dict(removed=str(path),bytes=item['identity'][0]))+'\n')
            if count%5000==0:
                stream.flush(); print(f'Removed {count:,} files, {total/1024**3:.2f} GiB logical',flush=True)
    return dict(files_deleted=count,removed_logical_bytes=total,free_bytes=shutil.disk_usage(root).free)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parent))
    p.add_argument('--data-run',default=DATA_RUN)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args(); root=Path(args.root).resolve()
    def work():
        print('Checking V3.18 original audio, noise and fixed Dev dependencies...',flush=True)
        protected,errors=current_inputs(root,args.data_run,lambda t:print(t,flush=True))
        if errors: raise ValueError('Current inputs could not be verified: '+repr(errors))
        caches, metadata = cache_plan(root,protected)
        checkpoints,kept,checked,bests=checkpoint_plan(root)
        result=dict(root=str(root),data_run=args.data_run,candidates=caches+checkpoints,metadata=metadata,
                    checked_files=checked,verified_best=bests,retained_checkpoints=kept,
                    policy='Keep all bests and metadata; retire only owned Train payloads and four reviewed resume paths')
        name=root/'exp'/('cleanup_v318_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        report=name.with_suffix('.json')
        with report.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        for label,items in (('CACHE',caches),('CHECKPOINT',checkpoints)):
            print(f'{label}_SELECTED_GiB={sum(i["identity"][0] for i in items)/1024**3:.3f}',flush=True)
        for item in kept:print('KEEP_CHECKPOINT='+item['path']+'; '+item['reason'],flush=True)
        print('CLEANUP_REPORT='+str(report),flush=True)
        if args.apply:
            result['result']=apply(result,name.with_suffix('.removed.jsonl'))
            report.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(f'FILES_DELETED={result["result"]["files_deleted"]}\n'
                  f'REMOVED_LOGICAL_GiB={result["result"]["removed_logical_bytes"]/1024**3:.3f}\n'
                  f'FREE_GiB={result["result"]["free_bytes"]/1024**3:.3f}',flush=True)
    if args.apply:
        with idle_lock(root):work()
    else:work()


if __name__=='__main__':main()
