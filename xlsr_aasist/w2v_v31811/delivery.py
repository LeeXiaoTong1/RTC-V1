"""Package frozen V3.18.11 diagnostics and publish a temp.sh download link.

Only allowlisted diagnostic files are included. Audio, model checkpoints,
optimizer state, and Progress/Eval outputs are NEVER gathered or uploaded.
The diagnostic run is not modified except for upload receipt/link metadata.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse
import zipfile


FILES = {
    '.': ('completed.json', 'failure.json'),
    'scores': ('score_summary.json', 'epoch_metrics.csv', 'epoch_flips.json',
               'paired_summary.json', 'paired_sources.csv', 'en_online_margins.png'),
    'scores/audit': ('paired_summary.json', 'paired_sources.csv'),
    'features': ('feature_manifest.json', 'feature_probe_summary.json', 'train_features.npz',
                 'select_features.npz', 'audit_features.npz'),
}


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def input_files(directory):
    root=Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Expected a V3.18.11 diagnostic directory')
    feature_manifest=root/'features'/'feature_manifest.json'
    verified_features=set()
    if feature_manifest.is_file():
        if feature_manifest.is_symlink():
            raise ValueError('Refusing feature manifest symlink')
        manifest=json.loads(feature_manifest.read_text(encoding='utf8'))
        if manifest.get('schema')!='v31811_source_bound_features_v2':
            raise ValueError('Feature manifest is not from the corrected source-bound V3.18.11')
        verified_features=set(manifest.get('files',{}))
        required={'train_features.npz','select_features.npz','feature_probe_summary.json'}
        if not required.issubset(verified_features):
            raise ValueError('Feature manifest is incomplete')
        for name,checksum in manifest['files'].items():
            if name not in FILES['features'] or name=='feature_manifest.json':
                raise ValueError('Unknown file in feature manifest: '+name)
            entry=root/'features'/name
            if entry.is_symlink() or not entry.is_file() or sha256_file(entry)!=checksum:
                raise ValueError('Feature artifact differs from verified manifest: '+name)
    found=[]
    for folder, names in FILES.items():
        parent=root/folder
        if parent.is_symlink():
            raise ValueError('Diagnostic directory contains a symbolic link: '+str(parent))
        if not parent.exists():
            continue
        for name in names:
            if folder=='features' and (not feature_manifest.is_file() or
                                      (name!='feature_manifest.json' and name not in verified_features)):
                continue
            path=parent/name
            if path.is_symlink():
                raise ValueError('Refusing diagnostic symlink: '+str(path))
            if path.exists():
                if not path.is_file():
                    raise ValueError('Expected a regular diagnostic file: '+str(path))
                found.append((path,path.relative_to(root).as_posix()))
    if not found or not any(name.startswith(('scores/', 'features/')) for _,name in found):
        raise ValueError('No saved diagnostic results to package; run diagnostics first')
    return root,sorted(found,key=lambda x:x[1])


def pack(directory, download_dir='/home/ubuntu/LXT/temp'):
    root, files=input_files(directory)
    dest=Path(download_dir).expanduser().resolve()
    if dest==root or root in dest.parents:
        raise ValueError('Diagnostic ZIP destination must be outside diagnostics')
    dest.mkdir(parents=True,exist_ok=True)
    run_id=root.parent.name if root.name=='v31811_diagnostics' else root.name
    archive=dest/(run_id+'_v31811_diagnostics.zip')
    entries=[dict(file=name,size_bytes=path.stat().st_size,sha256=sha256_file(path)) for path,name in files]
    manifest=dict(version='3.18.11',purpose='Read-only frozen model diagnostic handoff',
                  scope='Official Train and Dev only; no Progress/Eval',
                  included_content='Aggregated metrics, optional Train/Dev derived features, source IDs and logits',
                  excludes='Original waveforms, checkpoint weights, optimizer state and evaluation submission',
                  files=entries)
    data=json.dumps(manifest,sort_keys=True,ensure_ascii=True,indent=2).encode('utf-8')
    with tempfile.NamedTemporaryFile(prefix=archive.name+'.',suffix='.tmp',dir=dest,delete=False) as f:
        temporary=Path(f.name)
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for path,name in files:
                before=entries[[e['file'] for e in entries].index(name)]['sha256']
                if sha256_file(path)!=before:
                    raise ValueError('Diagnostic file changed before packaging: '+name)
                z.write(path,arcname=name)
                if sha256_file(path)!=before:
                    raise ValueError('Diagnostic file changed during packaging: '+name)
            z.writestr('manifest.json',data)
        os.replace(temporary,archive)
    finally:
        if temporary.exists():temporary.unlink()
    print('V31811_ARCHIVE='+str(archive),flush=True)
    return archive, sha256_file(archive), manifest


def valid_url(url):
    p=urlparse(url)
    return p.scheme=='https' and p.netloc=='temp.sh' and p.path not in ('','/','/upload')


def deliver(directory,download_dir='/home/ubuntu/LXT/temp',force=False,upload=True,uploader=None):
    root=Path(directory).expanduser().resolve(strict=True)
    archive,sha,manifest=pack(root,download_dir)
    if not upload:
        return dict(archive=str(archive),sha256=sha,files=len(manifest['files']))
    receipt=root/'upload_receipt.json'
    if receipt.is_file() and not force:
        old=json.loads(receipt.read_text(encoding='utf-8'))
        if old.get('archive_sha256')==sha and valid_url(old.get('url','')):
            url=old['url']
            (root/'temp_download_url.txt').write_text(url+'\n',encoding='utf-8')
            print('V31811_TEMP_DOWNLOAD_URL='+url,flush=True)
            return dict(archive=str(archive),sha256=sha,url=url,cached=True)
    if uploader is None:
        from w2v_aasist.launch import upload_archive
        uploader=upload_archive
    url=uploader(archive)
    if not valid_url(url):
        raise ValueError('Unexpected temp.sh upload response: '+repr(url))
    if sha256_file(archive)!=sha:
        raise ValueError('Diagnostic ZIP changed during upload')
    record=dict(url=url,archive_sha256=sha,archive=str(archive),diagnostic_dir=str(root),
                file_count=len(manifest['files']))
    receipt.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    (root/'temp_download_url.txt').write_text(url+'\n',encoding='utf-8')
    (root/'download_links.txt').write_text('V31811_DIAGNOSTICS='+url+'\n',encoding='utf-8')
    print('V31811_TEMP_DOWNLOAD_URL='+url,flush=True)
    print('V31811_LINK_FILE='+str(root/'download_links.txt'),flush=True)
    return dict(record,cached=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True,help='Existing V3.18.11 diagnostics output directory')
    p.add_argument('--download-dir',default='/home/ubuntu/LXT/temp')
    p.add_argument('--force',action='store_true',help='Re-upload, e.g. after temp.sh link expiry')
    p.add_argument('--pack-only',action='store_true',help='Produce ZIP without network upload')
    a=p.parse_args()
    deliver(a.out,a.download_dir,force=a.force,upload=not a.pack_only)


if __name__=='__main__':main()
