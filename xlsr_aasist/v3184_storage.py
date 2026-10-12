"""Preview storage; optionally retire ONLY verified, completed noise-preparation raw inputs.

Never deletes checkpoints, training/Dev audio, prepared noise, environments or
global package/model caches. No recursive deletion. Default is a preview.
"""
import argparse
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import stat
import time
from w2v_v3184.noise import bind_manifest,sha256
from w2v_v3184.prepare_noise import preparation_lock,atomic_json

ROOT=Path(__file__).resolve().parent
SOURCES=ROOT/'noise_v3184_rnnoise'/'sources.json'
GIB=1024**3


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_path(bank,relative,missing=False):
    parts=PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in ('..','.') or '\\' in p or ':' in p for p in parts.parts):
        raise ValueError('Unsafe bank-relative path: '+relative)
    path=bank.joinpath(*parts.parts)
    current=bank
    for name in parts.parts:
        current=current/name
        if current.is_symlink():raise ValueError('Refuse symlink: '+str(current))
    if path.resolve().is_relative_to(bank) is False:raise ValueError('Path escapes noise bank')
    if not path.exists():
        if missing:return path
        raise FileNotFoundError(path)
    if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Expected regular file: '+str(path))
    return path


def stamp(path):
    st=path.lstat()
    return [st.st_size,st.st_mtime_ns,st.st_dev,st.st_ino]


def noise_plan(bank,sources_path=SOURCES):
    bank=Path(bank).expanduser().resolve()
    manifest_path=checked_path(bank,'manifest.json')
    manifest=read(manifest_path);sources=read(sources_path)
    identity=manifest.get('preparation_identity',{})
    if (manifest.get('variant')!='v3184-published-noise-sources-v1'
            or not re.fullmatch(r'[0-9a-f]{64}',identity.get('sources_sha256',''))):
        raise ValueError('Not a completed published-source noise bank')
    report=checked_path(bank,'preparation_report.json')
    if read(report).get('identity')!=identity:raise ValueError('Noise completion report mismatch')
    # Only the canonical prepared audio directory is eligible for this utility.
    for row in manifest.get('records',[]):
        if not row['file'].startswith('audio/'):raise ValueError('Noncanonical prepared audio path')
        checked_path(bank,row['file'])
    bound=bind_manifest(manifest_path)  # Hash all retained audio before any deletion.
    index_path=checked_path(bank,'raw/rnnoise_selected/index.json')
    index=read(index_path)
    if index.get('identity',{}).get('archive_sha256')!=sources['archive']['sha256']:
        raise ValueError('Selected raw index belongs to a different archive')
    candidates={}
    def add(relative,size,digest):
        if relative in candidates:raise ValueError('Duplicate cleanup candidate')
        if not isinstance(size,int) or size<=0 or not re.fullmatch(r'[0-9a-f]{64}',digest):
            raise ValueError('Invalid expected raw fingerprint')
        path=checked_path(bank,relative,missing=True)
        if not path.exists():return
        if path.stat().st_size!=size:raise ValueError('Raw size changed; keep file: '+str(path))
        if str(path) in bound['scene_noise_files']:raise ValueError('Cannot retire a training dependency')
        candidates[relative]=dict(file=relative,bytes=size,sha256=digest,stat=stamp(path))
    add('raw/rnnoise_contributions.tar.gz',sources['archive']['bytes'],sources['archive']['sha256'])
    add('raw/silero_vad.jit',sources['vad']['bytes'],sources['vad']['sha256'])
    for row in sources['esc50']:
        if not re.fullmatch(r'esc50-[A-Za-z0-9-]+',row['key']):raise ValueError('Unsafe ESC key')
        add('raw/'+row['key']+'.wav',row['bytes'],row['sha256'])
    seen=set()
    for row in index['records']:
        name=row['file']
        if not re.fullmatch(r'\d{13}-(office|coffee)\.raw',name) or name in seen:
            raise ValueError('Unexpected selected raw name')
        seen.add(name);add('raw/rnnoise_selected/'+name,row['bytes'],row['source_sha256'])
    pins={str(p):sha256(p) for p in (manifest_path,report,index_path,Path(sources_path))}
    return dict(bank=str(bank),candidates=list(candidates.values()),
        reclaimable_bytes=sum(r['bytes'] for r in candidates.values()),
        retained_prepared_audio_bytes=sum(r['size'] for rows in bound['scene_noise_records'].values() for r in rows),
        completion_pins=pins,
        recorded_source_catalog_sha256=identity['sources_sha256'],
        current_source_catalog_sha256=sha256(sources_path),
        note='Keep prepared audio, manifest, raw index, screening reports, provenance and all checkpoints. '
             'Completed preparation and training need no retired raw input; rebuilding a changed recipe requires re-download.')


def apply_plan(plan,progress=print):
    """Caller holds the same OS lock as the noise preparation process."""
    bank=Path(plan['bank'])
    if not plan['candidates']:
        return dict(status='already_clean',removed_bytes=0,removed_files=0)
    for path,digest in plan['completion_pins'].items():
        if sha256(path)!=digest:raise ValueError('Completion metadata changed; no files removed')
    # Authenticate every payload before deleting the first one. This is a one-off
    # maintenance cost, not an extra hash/scan in the training loop.
    progress('[Verify] checking raw hashes before cleanup; large archives can take a few minutes')
    for row in plan['candidates']:
        path=checked_path(bank,row['file'])
        if stamp(path)!=row['stat'] or sha256(path)!=row['sha256'] or stamp(path)!=row['stat']:
            raise ValueError('Raw content changed; no files removed: '+str(path))
    receipt=bank/'raw_cleanup_receipt.json'
    checked_path(bank,receipt.name,missing=True)
    checked_path(bank,receipt.name+'.tmp',missing=True)
    # A durable journal describes the validated plan even if cleanup is interrupted.
    result=dict(status='deleting',started_at_unix=time.time(),plan=plan)
    atomic_json(receipt,result)
    for row in plan['candidates']:
        path=checked_path(bank,row['file'])
        if stamp(path)!=row['stat']:raise ValueError('Raw file changed during cleanup: '+str(path))
        path.unlink()
    result.update(status='complete',removed_bytes=plan['reclaimable_bytes'],removed_files=len(plan['candidates']))
    atomic_json(receipt,result)
    return result


def checkpoint_inventory(root):
    """Metadata only: never deserialize old checkpoint tensors or infer unusedness."""
    exp=Path(root)/'exp';rows=[]
    if not exp.is_dir():return rows
    for run in sorted(exp.iterdir()):
        if not run.is_dir() or run.is_symlink():continue
        for path in sorted(run.iterdir()):
            if not path.is_symlink() and path.is_file() and path.suffix in ('.pt','.pth','.ckpt','.safetensors'):
                rows.append(dict(path=str(path),bytes=path.stat().st_size,action='KEEP'))
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(ROOT))
    parser.add_argument('--noise-bank',help='Default: ROOT/assets/noise_v3184_rnnoise')
    parser.add_argument('--apply',action='store_true',help='Remove only the verified raw noise preparation payloads')
    args=parser.parse_args(argv);root=Path(args.root).expanduser().resolve()
    bank=Path(args.noise_bank).expanduser().resolve() if args.noise_bank else root/'assets'/'noise_v3184_rnnoise'
    bank=bank.resolve()
    print('MODE='+('APPLY_NOISE_RAW_ONLY' if args.apply else 'PREVIEW_NO_DELETION'),flush=True)
    print(f'PROJECT_FREE_GIB={shutil.disk_usage(root).free/GIB:.3f}',flush=True)
    models=checkpoint_inventory(root)
    print(f'CHECKPOINTS_KEEP={len(models)}; GIB={sum(r["bytes"] for r in models)/GIB:.3f}',flush=True)
    for row in models:print(f'KEEP {row["bytes"]/GIB:.3f} GiB {row["path"]}',flush=True)
    print('Keep last.pt: R2/AUG5 store best_model, latest model and optimizer together. No automatic checkpoint deletion.',flush=True)
    if not (bank/'manifest.json').is_file():
        if args.apply:raise ValueError('Noise preparation has no completed manifest; do not clean raw inputs')
        print('NOISE_RAW_CLEANUP=not_ready; finish preparation first',flush=True);return
    # Never contend with an active downloader/screener; a stale lock FILE is OK.
    checked_path(bank,'.prepare.lock',missing=True)
    with preparation_lock(bank/'.prepare.lock'):
        print('[Verify] checking completed noise manifest and retained prepared audio',flush=True)
        plan=noise_plan(bank)
        print('NOISE_BANK='+str(bank),flush=True)
        print(f'NOISE_RAW_CANDIDATES={len(plan["candidates"])}; RECLAIMABLE_GIB={plan["reclaimable_bytes"]/GIB:.3f}',flush=True)
        print(f'KEEP_PREPARED_AUDIO_GIB={plan["retained_prepared_audio_bytes"]/GIB:.3f}',flush=True)
        if args.apply:
            result=apply_plan(plan,lambda text:print(text,flush=True))
            print(f'REMOVED_FILES={result["removed_files"]}; REMOVED_GIB={result["removed_bytes"]/GIB:.3f}',flush=True)
            print('RECEIPT='+str(bank/'raw_cleanup_receipt.json'),flush=True)
        else:print('PREVIEW_COMPLETE; optional cleanup: repeat with --apply',flush=True)


if __name__=='__main__':main()
