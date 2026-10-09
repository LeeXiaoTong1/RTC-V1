"""Read-only storage inventory. No deletion, tensor loading, or audio hashing.

Outputs are recommendations for manual review, NEVER an executable cleanup plan.
Historical JSON path references are evidence of a possible dependency, not proof
that a checkpoint or cache is currently used. No reference is not deletion proof.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

LIMIT = 64*1024**2
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors'}
AUDIT_PREFIXES = ('cache_retirement_', 'cleanup_', 'storage_audit_')


def load(path):
    path = Path(path)
    if path.stat().st_size > LIMIT: raise ValueError('metadata exceeds 64 MiB: '+str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def strings(obj):
    if isinstance(obj, str): yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj: yield from strings(value)


def path_values(obj, root, parent):
    for value in strings(obj):
        if not value or len(value)>4096 or '\x00' in value: continue
        if not ('/' in value or '\\' in value or Path(value).suffix in WEIGHTS | {'.npy','.wav','.flac','.json','.csv'}): continue
        if '://' in value: continue
        p = Path(value).expanduser()
        for q in ([p] if p.is_absolute() else [root/p, parent/p]):
            yield q.resolve()


def existing_children(path):
    if not path.is_dir() or path.is_symlink(): return []
    return sorted(path.iterdir(), key=lambda p:p.name)


def cache_units(path, depth=2):
    """Split container directories, but never enumerate a feature hash hierarchy."""
    if path.is_symlink() or not path.is_dir():
        yield path
        return
    if path.name == 'w2v_feature_cache' or depth == 0 or any((path/n).is_file() for n in ('owner.json','config.json','generation.json')):
        yield path
        return
    children = existing_children(path)
    if not children or len(children)>64 or any(not p.is_dir() or p.is_symlink() for p in children):
        yield path
        return
    for child in children: yield from cache_units(child, depth-1)


def discover(root, references):
    items = {}
    for p in existing_children(root/'pretrained'): items[p] = 'pretrained'
    for p in existing_children(root/'data'):
        for unit in cache_units(p): items[unit] = 'data'
    for run in existing_children(root/'exp'):
        if run.is_symlink() or not run.is_dir():
            if run.is_file() and (run.name.startswith(AUDIT_PREFIXES) or run.suffix in WEIGHTS):
                items[run] = 'checkpoint' if run.suffix in WEIGHTS else 'audit_record'
            continue
        for current, dirs, files in os.walk(run, followlinks=False):
            parent = Path(current)
            keep_dirs = []
            for name in dirs:
                child = parent/name
                if child.is_symlink(): items[child] = 'linked_path'
                elif name == 'features':
                    for unit in cache_units(child, 1): items[unit] = 'feature_cache'
                elif name == 'rolling_pairs': items[child] = 'rolling_cache'
                elif name not in ('diagnostics','__pycache__','.git'): keep_dirs.append(name)
            dirs[:] = keep_dirs
            for name in files:
                p = parent/name
                if p.suffix in WEIGHTS: items[p] = 'checkpoint'
    # Some old Train caches live outside the checkout and only appear in config.
    families = {'rtc_noisy_cache_v2','rtc_noisy_improved_v1','rtc_noisy_full2_v1','rtc_v35','w2v_feature_cache'}
    found_roots = set()
    for p in references:
        for ancestor in (p, *p.parents):
            if ancestor.name in families:
                if ancestor not in found_roots and ancestor.is_dir():
                    for unit in cache_units(ancestor): items.setdefault(unit, 'historical_cache')
                found_roots.add(ancestor)
                break
    # Remove overlapping units; parents already include their children.
    result = {}
    for p, kind in sorted(items.items(), key=lambda item:len(item[0].parts)):
        if not any(a in result for a in p.parents): result[p] = kind
    return result


def scan_references(root):
    refs = defaultdict(set)
    skipped = []
    docs = list((root/'exp').glob('*.json'))
    for run in existing_children(root/'exp'):
        if run.is_dir() and not run.is_symlink(): docs.extend(run.glob('*.json'))
    for path in sorted(set(docs)):
        if path.name.startswith(AUDIT_PREFIXES) or path.name in ('train_rows.json','dev_rows.json'):
            skipped.append(dict(path=str(path), reason='historical log or waveform list; not scanned for checkpoint dependencies'))
            continue
        try:
            obj = load(path)
            for p in path_values(obj, root, path.parent): refs[p].add(str(path))
        except (OSError, ValueError) as exc:
            skipped.append(dict(path=str(path), reason=str(exc)))
    return refs, skipped


def current_inputs(root, source):
    """Mirror V3.18's manifest-only inputs without importing torch/fairseq2."""
    protected = defaultdict(set)
    errors = []
    def keep(p, reason): protected[Path(p).resolve()].add(reason)
    source = (root/source).resolve() if not Path(source).is_absolute() else Path(source).resolve()
    try:
        cfg = load(source/'config.json')
        if cfg.get('version') not in ('3.15','3.15.1','3.16','3.16.1','3.17'):
            raise ValueError('Unsupported V3.18 data-run version')
        keep(source/'config.json', 'V3.18 data-run configuration')
        old = cfg.get('continuation_source_config', cfg)
        producer = Path(old['v37_run'])
        if not producer.is_absolute(): producer = root/producer
        for split in ('train','dev'):
            folder = producer/'features'/split
            if not (folder/'complete.json').is_file():
                reuse = producer/'feature_reuse.json'
                keep(reuse, 'canonical manifest location')
                folder = Path(load(reuse)[split]['path'])
                if not folder.is_absolute(): folder = root/folder
            for name in ('rows.json','owner.json','complete.json'):
                keep(folder/name, 'V3.18 canonical manifest metadata')
            rows, owner, complete = [load(folder/name) for name in ('rows.json','owner.json','complete.json')]
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            signature = hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if (complete['files']['rows.json'] != digest(folder/'rows.json') or
                complete['owner_sha256'] != digest(folder/'owner.json') or
                owner['records_digest'] != signature or len(rows) != owner['shape'][0] or owner['identity']['split'] != split):
                raise ValueError('Canonical manifest identity mismatch: '+str(folder))
            for row in rows:
                if split=='dev' or row['condition'] in ('offline','online'):
                    keep(row['audio'], 'fixed Dev audio' if split=='dev' else 'original Train audio')
        for mapping in (old, old.get('source_config',{}), old.get('dev_config',{})):
            for key in ('train_data_path','dev_data_path','train_protocol','dev_protocol','train_noise_manifest','dev_noise_manifest'):
                if mapping.get(key): keep(mapping[key], 'V3.18 '+key)
        for p in old.get('augmentation_files',{}): keep(p, 'noise/augmentation input')
        for split, rows in old.get('noise_records',{}).items():
            for row in rows: keep(row['path'], 'noise recording: '+split)
        # Dev pair discovery also searches parents of the original audio tree.
        for p in old.get('data_fingerprints',{}):
            if Path(p).suffix=='.csv': keep(p, 'historical pair/protocol table')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        errors.append(str(exc))
    return protected, errors


def measure(path, progress=print):
    total = allocated = files = links = hardlinks = 0
    errors = []
    seen = set()
    tick = time.monotonic()
    def account(p):
        nonlocal total,allocated,files,links,hardlinks,tick
        try:
            s = p.lstat()
            if stat.S_ISLNK(s.st_mode): links += 1; return
            if not stat.S_ISREG(s.st_mode): return
            files += 1; total += s.st_size
            if s.st_nlink>1: hardlinks += 1
            key=(s.st_dev,s.st_ino)
            if key not in seen:
                allocated += getattr(s,'st_blocks',(s.st_size+511)//512)*512
                seen.add(key)
            if time.monotonic()-tick>=3:
                progress(f'  scanned {files:,} files, logical={total/1024**3:.2f} GiB')
                tick=time.monotonic()
        except OSError as exc: errors.append(str(exc))
    if path.is_symlink() or not path.is_dir(): account(path)
    else:
        for current, dirs, names in os.walk(path,followlinks=False,onerror=lambda e:errors.append(str(e))):
            parent=Path(current)
            for name in list(dirs):
                if (parent/name).is_symlink(): links+=1; dirs.remove(name)
            for name in names: account(parent/name)
    return dict(logical_bytes=total,allocated_bytes=allocated,files=files,symlinks_skipped=links,
                hardlinked_files=hardlinks,errors=errors)


def collect(root, source, progress=print):
    root=Path(root).resolve()
    if not (root/'exp').is_dir(): raise ValueError('Expected project containing exp/')
    progress('Reading current manifests and historical configurations; no tensor/audio loads...')
    refs, skipped=scan_references(root)
    current, errors=current_inputs(root,source)
    items=discover(root,refs)
    # Assign references once, by ancestors, rather than comparing every audio to every item.
    matches=defaultdict(set); history=defaultdict(set)
    for inventory,index in ((current,matches),(refs,history)):
        for path, reasons in inventory.items():
            for ancestor in (path,*path.parents):
                if ancestor in items: index[ancestor].update(reasons)
        for item in items:
            for ancestor in item.parents:
                if ancestor in inventory: index[item].update(inventory[ancestor])
    rows=[]
    for index,(path,kind) in enumerate(sorted(items.items(),key=lambda v:str(v[0])),1):
        progress(f'[{index}/{len(items)}] Measuring {path}')
        size=measure(path,progress)
        status='REVIEW_UNKNOWN'
        if kind=='checkpoint':
            status='REVIEW_CHECKPOINT'
            if 'best' in path.name.lower() or path.name=='inference.pt' or history[path]: status='KEEP_CHECKPOINT'
            if re.match(r'w2v_v3(?:1[2-9]|[2-9]\d)', path.relative_to(root/'exp').parts[0]): status='KEEP_CHECKPOINT'
        elif kind=='audit_record': status='KEEP_AUDIT_RECORD'
        elif kind=='pretrained': status='KEEP_PRETRAINED'
        elif kind=='linked_path' or size['symlinks_skipped'] or size['errors']: status='REVIEW_INCOMPLETE'
        elif kind in ('feature_cache','rolling_cache','historical_cache') or any(name in path.parts for name in ('rtc_v35','rtc_noisy_cache_v2','rtc_noisy_improved_v1','rtc_noisy_full2_v1','w2v_feature_cache')):
            status='REVIEW_OLD_CACHE'
        if matches[path]: status='KEEP_V318_INPUT_OR_METADATA'
        rows.append(dict(path=str(path),kind=kind,status=status,**size,
                         v318_reasons=sorted(matches[path]),historical_references=sorted(history[path])))
        progress(f'  {size["logical_bytes"]/1024**3:.3f} GiB; {status}; historical_refs={len(history[path])}')
    rows.sort(key=lambda r:r['logical_bytes'],reverse=True)
    return dict(schema='rtc_storage_readonly_v1',read_only=True,deletion_authorized=False,root=str(root),
                data_run=str(source),current_input_errors=errors,skipped_metadata=skipped,entries=rows,
                notes=['REVIEW does not mean safe to delete. No checkpoint score is inferred from its filename.',
                       'Directory references and embedded best/last require review before any deletion.',
                       'A features folder marked KEEP includes small required manifests; its arrays may be separately reviewable.',
                       'Reference search is limited to top-level experiment metadata, not tensor contents or all historical logs.',
                       'Audio/weight contents are not read or hashed; this is not a training preflight or full integrity check.',
                       'Sizes are not guaranteed reclaimed space: links, mount points and shared blocks may differ.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(Path(__file__).resolve().parent))
    p.add_argument('--data-run',required=True)
    args=p.parse_args()
    root=Path(args.root).resolve()
    result=collect(root,args.data_run,lambda text:print(text,flush=True))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path=root/'exp'/('storage_audit_'+stamp+'.json')
    text_path=path.with_suffix('.txt')
    # Only NEW report files are written. There is deliberately no --apply mode.
    with path.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    with text_path.open('x',encoding='utf-8') as f:
        f.write('READ ONLY: no files deleted. REVIEW is not deletion approval.\n')
        for row in result['entries']:
            f.write(f'{row["logical_bytes"]/1024**3:.3f} GiB\t{row["status"]}\t{row["path"]}\n')
            for reason in row['v318_reasons']: f.write('  V318: '+reason+'\n')
            for ref in row['historical_references']: f.write('  Historical: '+ref+'\n')
        if result['current_input_errors']: f.write('INPUT CHECK INCOMPLETE: '+repr(result['current_input_errors'])+'\n')
    print('\nLargest entries (not deletion candidates):',flush=True)
    for row in result['entries'][:20]: print(f'{row["logical_bytes"]/1024**3:8.3f} GiB  {row["status"]}  {row["path"]}')
    print(f'STORAGE_AUDIT_JSON={path}\nSTORAGE_AUDIT_TEXT={text_path}\nFILES_DELETED=0',flush=True)
    if result['current_input_errors']: print('CURRENT_INPUT_CHECK_INCOMPLETE='+repr(result['current_input_errors']),flush=True)


if __name__=='__main__': main()
