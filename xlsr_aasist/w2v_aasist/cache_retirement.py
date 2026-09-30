"""Retire only owned legacy Train WAVs/fixed filterbanks after replacement validation."""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import zipfile
import numpy as np
import soundfile as sf
from .runtime import atomic_json, sha256


def signature(path):
    s = path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def require_inside(path, root, protected):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink():
        raise ValueError('Refusing cache symlink: ' + str(path))
    resolved = path.resolve()
    resolved.relative_to(root)
    for item in protected:
        item = Path(item).resolve()
        if resolved == item or item in resolved.parents or resolved in item.parents:
            raise ValueError('Cleanup overlaps protected input: ' + str(resolved))
    return resolved


def legacy_files(folder, protocol_hash, protected):
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError('Refusing linked cache directory')
    folder = folder.resolve()
    if not folder.exists(): return [], []
    cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if (cfg.get('format') != 'rtc_noisy_pair_cache_v1' or cfg.get('role') != 'train'
            or cfg.get('split') != 'train' or cfg.get('protocol_sha256') != protocol_hash
            or cfg.get('cut') != 64600 or cfg.get('sr') != 16000 or cfg.get('limit')):
        raise ValueError('Not an obsolete full Train cache: ' + str(folder))
    files = []
    with (folder/'manifest.jsonl').open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            if number == 1 or number % 20000 == 0:
                print(f'Checking obsolete Train cache {folder.name}: {number} records',flush=True)
            row = json.loads(line)
            rel = Path(row['audio'])
            if (rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'audio'
                    or not re.fullmatch('[a-f0-9]{24}_b[0-3]\\.wav',rel.name)
                    or row.get('role') != 'train' or row.get('generation') != cfg['generation']):
                raise ValueError('Unrecognized cache ownership record')
            audio = require_inside(folder/rel, folder, protected)
            if audio.exists():
                header = sf.info(audio)
                if (header.frames,header.samplerate,header.channels) != (64600,16000,1):
                    raise ValueError('Unexpected legacy cache header')
                files.append(audio)
            sidecar = audio.with_suffix('.json')
            if sidecar.exists():
                require_inside(sidecar,folder,protected)
                saved = json.loads(sidecar.read_text(encoding='utf-8'))
                if saved.get('source') != row['source'] or saved.get('audio') != row['audio']:
                    raise ValueError('Sidecar ownership differs from manifest')
                files.append(sidecar)
    # Retain small metadata in place as well as an archive; resume old training requires regeneration.
    return files, [folder/'config.json',folder/'manifest.jsonl']


def feature_files(folder, protected):
    folder = Path(folder)
    if folder.is_symlink(): raise ValueError('Refusing linked feature cache')
    if not folder.exists(): return []
    result = []
    for number,p in enumerate(folder.rglob('*.npy'),1):
        if number == 1 or number % 20000 == 0:
            print(f'Checking obsolete fixed features: {number} entries',flush=True)
        rel = p.relative_to(folder).parts
        if (len(rel) != 3 or not re.fullmatch('[a-f0-9]{64}',rel[0])
                or not re.fullmatch('[a-f0-9]{2}',rel[1])
                or not re.fullmatch('[a-f0-9]{64}\\.npy',rel[2]) or rel[1] != rel[2][:2]):
            continue
        path = require_inside(p,folder,protected)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        valid = array.dtype == np.float32 and array.ndim == 2 and array.shape[1] == 160 and array.shape[0] >= 12
        if getattr(array,'_mmap',None) is not None: array._mmap.close()
        del array
        if not valid: raise ValueError('Not a recognized fixed filterbank file: ' + str(path))
        result.append(path)
    return result


def retire(source, cfg, project_root, ensure_idle):
    if not cfg.get('full_noisy') or len(cfg.get('train_caches',[])) != 1:
        raise ValueError('Only the validated full two-view replacement can retire legacy Train caches')
    replacement = Path(cfg['train_caches'][0])
    rcfg = json.loads((replacement/'config.json').read_text(encoding='utf-8'))
    done = json.loads((replacement/'complete.json').read_text(encoding='utf-8'))
    if (rcfg.get('format') != 'rtc_noisy_full2_cache_v1' or rcfg.get('role') != 'train'
            or done.get('config_sha256') != sha256(replacement/'config.json')
            or done.get('manifest_sha256') != sha256(replacement/'manifest.jsonl')
            or done.get('rows') != 2*rcfg.get('offline_count',0) or done.get('rows',0) < 2):
        raise ValueError('Replacement cache is incomplete/changed; no old cache removed')
    root = Path(project_root).resolve()
    train_parent = Path(source['train_data_path']).resolve().parents[1]/'rtc_noisy_cache_v2'
    improved_parent = root/'data'/'rtc_noisy_improved_v1'
    candidates = [Path(source['train_noisy_cache']), improved_parent/'train_g1']
    extra = source.get('extra_train_noisy_cache') or []
    candidates += [Path(extra)] if isinstance(extra,str) else list(map(Path,extra))
    protected = [cfg['baseline'],cfg['ssl_path'],cfg['train_data_path'],cfg['dev_data_path'],
                 cfg['dev_noisy_cache'],cfg['dev_heldout_cache'],*cfg['train_caches']]
    files, metadata, roots = {}, [], []
    for folder in dict.fromkeys(candidates):
        if not folder.exists(): continue
        if folder.is_symlink(): raise ValueError('Linked retirement target')
        resolved = folder.resolve()
        if (resolved.parent not in {train_parent.resolve(),improved_parent.resolve()}
                or not re.fullmatch(r'train_g\d+',resolved.name)):
            print('Keeping cache outside retirement allowlist: ' + str(folder),flush=True)
            continue
        require_inside(folder, resolved.parent, protected)
        owned, meta = legacy_files(folder,sha256(cfg['train_protocol']),protected)
        roots.append(resolved)
        metadata += meta
        for p in owned: files[p] = {'path':str(p),'root':str(resolved),'identity':signature(p)}
    feature = root/'data'/'w2v_feature_cache'
    if feature.exists():
        require_inside(feature,root/'data',protected)
        roots.append(feature.resolve())
        for p in feature_files(feature,protected):
            files[p] = {'path':str(p),'root':str(feature.resolve()),'identity':signature(p)}
    ensure_idle()
    for p,row in files.items():
        require_inside(p,row['root'],protected)
        if signature(p) != row['identity']: raise RuntimeError('Cache changed before cleanup')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log = root/'exp'/('cache_retirement_'+stamp+'.json')
    log.parent.mkdir(parents=True,exist_ok=True)
    backup = log.with_suffix('.zip')
    with zipfile.ZipFile(backup,'x',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(metadata): z.write(p,f'{i}_{p.parent.name}_{p.name}')
    with zipfile.ZipFile(backup) as z:
        for i,p in enumerate(metadata):
            if z.read(f'{i}_{p.parent.name}_{p.name}') != p.read_bytes():
                raise RuntimeError('Metadata archive verification failed')
    report = dict(status='planned',files=list(files.values()),protected=list(map(str,protected)),
                  metadata_archive=str(backup),logical_gib=sum(v['identity'][2] for v in files.values())/1024**3)
    atomic_json(log,report)
    deleted = 0
    try:
        for p,row in files.items():
            if deleted % 1000 == 0: ensure_idle()
            require_inside(p,row['root'],protected)
            if signature(p) != row['identity']: raise RuntimeError('Cache changed during cleanup')
            p.unlink()
            deleted += 1
            if deleted % 10000 == 0:
                print(f'Obsolete cache cleanup: {deleted}/{len(files)} files',flush=True)
    finally:
        atomic_json(log,dict(report,status='complete' if deleted==len(files) else 'partial',deleted_files=deleted))
    print(f'OBSOLETE_CACHE_FILES_DELETED={deleted} LOGICAL_GiB={report["logical_gib"]:.2f} REPORT={log}',flush=True)
    return report
