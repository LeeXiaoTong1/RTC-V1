"""Delete only manifest-owned replaced Train WAVs after reader validation."""
import json
from pathlib import Path
import zipfile
from w2v_aasist.runtime import atomic_json, sha256
from w2v_aasist.cache_retirement import require_inside, signature


def retire_replaced_cache(cfg, root, ensure_idle, report_dir):
    root = Path(root).resolve()
    old = Path(cfg['legacy_full_cache'])
    allowed = root/'data'/'rtc_noisy_full2_v1'/'train'
    if old.is_symlink() or old.resolve() != allowed.resolve():
        print('V33_KEEP_CACHE_OUTSIDE_ALLOWLIST='+str(old), flush=True)
        return {'status':'kept_outside_allowlist'}
    old = old.resolve()
    try:
        old.relative_to(root)
    except ValueError:
        # An allowlisted spelling may still traverse a linked data ancestor.
        print('V33_KEEP_CACHE_OUTSIDE_PROJECT='+str(old), flush=True)
        return {'status':'kept_outside_project'}
    if not old.is_dir():
        return {'status':'already_absent'}
    ensure_idle()
    new = Path(cfg['train_noisy_cache_v33']).resolve()
    if new == old or old in new.parents or new in old.parents:
        raise ValueError('Replacement overlaps obsolete cache')
    fingerprints = {str(Path(p).resolve()):digest for p,digest in cfg.get('preparation_fingerprints', {}).items()}
    # This map is recorded only after the actual V3.3 reader accepted the cache.
    for name in ('config.json','manifest.jsonl','complete.json'):
        path = new/name
        if str(path) not in fingerprints or sha256(path) != fingerprints[str(path)]:
            raise ValueError('Replacement not validated or metadata changed; no deletion: '+str(path))
    oldcfg = json.loads((old/'config.json').read_text(encoding='utf-8'))
    if (oldcfg.get('format') != 'rtc_noisy_full2_cache_v1' or oldcfg.get('role') != 'train'
            or oldcfg.get('protocol_sha256') != sha256(cfg['train_protocol'])):
        raise ValueError('Unexpected old cache ownership; no deletion')
    protected = [cfg[k] for k in ('baseline','warm_checkpoint','ssl_path','train_data_path',
        'dev_data_path','dev_noisy_cache','dev_heldout_cache')]+[str(new)]
    files = {}
    with (old/'manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            rel = Path(row['audio'])
            if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0]!='audio' or rel.suffix!='.wav':
                raise ValueError('Unexpected manifest path; no deletion')
            path = require_inside(old/rel, old, protected)
            if path.exists():
                files[str(path)] = signature(path)
            # Only the sidecar explicitly tied to this owned WAV may be retired.
            sidecar = path.with_suffix('.json')
            if sidecar.exists():
                require_inside(sidecar, old, protected)
                side = json.loads(sidecar.read_text(encoding='utf-8'))
                if side.get('source') != row.get('source') or side.get('audio') != row.get('audio'):
                    raise ValueError('Sidecar ownership mismatch; no deletion')
                files[str(sidecar)] = signature(sidecar)
    report_dir = Path(report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    archive = report_dir/'retired_cache_metadata.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('config.json','manifest.jsonl','complete.json'):
            path = old/name
            if path.is_file():
                z.write(path, name)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if z.read(name) != (old/name).read_bytes():
                raise RuntimeError('Metadata backup verification failed; no deletion')
    ensure_idle()
    for filename, expected in files.items():
        path = require_inside(filename, old, protected)
        if signature(path) != expected:
            raise RuntimeError('Old cache changed during preparation; no deletion')
    report = {'status':'planned','old_root':str(old),'replacement':str(new),
        'files':files,'logical_bytes':sum(v[2] for v in files.values()),
        'note':'Unlinking hardlinked A WAVs does not free their shared data; no recursive directory deletion.',
        'old_run_resume':'Retired cache WAVs must be regenerated to resume old V3/V3.1/V3.2 training.'}
    atomic_json(report_dir/'cache_retirement.json', report)
    deleted = 0
    try:
        for filename, expected in files.items():
            if deleted % 1000 == 0:
                ensure_idle()
            path = require_inside(filename, old, protected)
            if signature(path) != expected:
                raise RuntimeError('Old cache changed during retirement')
            path.unlink(); deleted += 1
    finally:
        report.update(status='complete' if deleted==len(files) else 'partial', deleted_files=deleted)
        atomic_json(report_dir/'cache_retirement.json', report)
        atomic_json(old/'retired_by_v33.json', {k:v for k,v in report.items() if k!='files'})
    print(f'V33_OBSOLETE_CACHE_FILES_DELETED={deleted}; checkpoints and Dev retained.', flush=True)
    return report
