"""Transfer only the pinned public LID dependency, never user model/audio data."""
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from w2v_aasist.runtime import atomic_json,sha256
from w2v_aasist.launch import run_lock
from .language import ASSET_FILES,REPO_ID,REVISION,WEIGHT_BYTES,WEIGHT_SHA256

FORMAT='rtc_v37_public_lid_bundle_v1'
EXPECTED={
    'embedding_model.ckpt':dict(size=WEIGHT_BYTES,sha256=WEIGHT_SHA256),
    'hyperparams.yaml':dict(size=1519,sha256='88fec9791a8416a152fb10834327e18d38e5bf7a351e9b714e08cdc4af05de6f'),
    'README.md':dict(size=9538,sha256='e2a24d3c912f3c02bc104abb86904271bacc245e55b9f856220e71c7d96eaf4c'),
}


def manifest():
    return dict(format=FORMAT,repo_id=REPO_ID,revision=REVISION,license='apache-2.0',files=EXPECTED)


def _verify_files(folder):
    paths={}
    for name,expected in EXPECTED.items():
        path=Path(folder)/name
        if (not path.is_file() or path.is_symlink() or path.stat().st_size!=expected['size']
                or sha256(path)!=expected['sha256']):
            raise ValueError('Offline teacher file differs from the pinned official bytes: '+name)
        paths[name]=path.resolve()
    return paths


def imported_paths(cache_dir):
    if not cache_dir:return None
    folder=Path(cache_dir).expanduser().resolve()/'v37_offline'/REVISION
    if not folder.exists():return None
    if folder.is_symlink() or json.loads((folder/'bundle.json').read_text(encoding='utf-8'))!=manifest():
        raise ValueError('Offline teacher manifest/revision differs')
    return _verify_files(folder)


def import_bundle(archive,cache_dir):
    """Validate bounded flat entries, stage under cache root, then rename atomically."""
    archive=Path(archive).expanduser().resolve()
    root=Path(cache_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    target=root/'v37_offline'/REVISION
    target.parent.mkdir(parents=True,exist_ok=True)
    # Verify the final target stays inside this explicit cache even with symlinks.
    target.resolve().relative_to(root)
    with run_lock(root/'.v37-import.lock'):
        with zipfile.ZipFile(archive) as z:
            infos=z.infolist();names=[entry.filename for entry in infos]
            if len(names)!=len(set(names)) or set(names)!=set(ASSET_FILES)|{'bundle.json','LICENSE'}:
                raise ValueError('Teacher ZIP must contain exactly the pinned flat files, manifest and license')
            for entry in infos:
                size=EXPECTED[entry.filename]['size'] if entry.filename in EXPECTED else 128*1024
                if (entry.is_dir() or ((entry.external_attr>>16)&0o170000)==0o120000
                        or entry.file_size>size or entry.flag_bits&1):
                    raise ValueError('Unexpected teacher ZIP entry/type/size: '+entry.filename)
                if entry.filename in EXPECTED and entry.file_size!=size:
                    raise ValueError('Teacher ZIP file size differs: '+entry.filename)
            if json.loads(z.read('bundle.json'))!=manifest():
                raise ValueError('Teacher ZIP manifest/revision differs from the pinned official version')
            if imported_paths(root) is not None:
                print('V37_OFFLINE_TEACHER_REUSED='+str(target),flush=True)
                return target
            required=sum(v['size'] for v in EXPECTED.values())+16*1024**2
            if shutil.disk_usage(root).free<required:
                raise OSError('Need %.3f GiB free to import the public teacher; no existing files deleted'%(required/1024**3))
            with tempfile.TemporaryDirectory(prefix='.v37-import-',dir=root) as directory:
                stage=Path(directory).resolve();stage.relative_to(root)
                payload=stage/'payload';payload.mkdir()
                for name in names:
                    # Explicit known names only; never extractall on a supplied ZIP.
                    with z.open(name) as source,(payload/name).open('wb') as destination:
                        shutil.copyfileobj(source,destination,1024*1024)
                        destination.flush();os.fsync(destination.fileno())
                _verify_files(payload)
                os.replace(payload,target)
    print('V37_OFFLINE_TEACHER_IMPORTED='+str(target),flush=True)
    return target


def export_bundle(cache_dir,out):
    """An allowlist excludes all user-trained best checkpoints, audio and vectors."""
    from .language import ensure_language_assets
    assets=ensure_language_assets(dict(language_cache_dir=str(cache_dir),language_offline=True))
    paths={name:Path(assets['files'][name]['path']) for name in ASSET_FILES}
    for name,path in paths.items():
        if path.stat().st_size!=EXPECTED[name]['size'] or sha256(path)!=EXPECTED[name]['sha256']:
            raise ValueError('Cannot export an unpinned public teacher file: '+name)
    dist=distribution('speechbrain')
    licenses=[p for p in dist.files or [] if str(p).endswith('.dist-info/LICENSE')]
    if len(licenses)!=1:raise ValueError('SpeechBrain Apache-2.0 LICENSE missing from installation')
    license_text=Path(dist.locate_file(licenses[0])).read_text(encoding='utf-8')
    if 'Apache License' not in license_text or 'Version 2.0' not in license_text:
        raise ValueError('Expected the distributed Apache-2.0 license')
    out=Path(out).expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Refusing to replace an existing offline bundle: '+str(out))
    temporary=out.with_name(out.name+'.tmp')
    if temporary.exists():raise FileExistsError('Unfinished bundle exists: '+str(temporary))
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name,path in paths.items():z.write(path,arcname=name)
            z.writestr('bundle.json',json.dumps(manifest(),indent=2))
            z.writestr('LICENSE',license_text)
        os.replace(temporary,out)
    finally:
        temporary.unlink(missing_ok=True)
    record=dict(archive=str(out),size=out.stat().st_size,sha256=sha256(out),**manifest())
    atomic_json(out.with_suffix(out.suffix+'.json'),record)
    print('V37_PUBLIC_TEACHER_BUNDLE='+str(out),flush=True)
    print('V37_PUBLIC_TEACHER_BUNDLE_SHA256='+record['sha256'],flush=True)
    return out
