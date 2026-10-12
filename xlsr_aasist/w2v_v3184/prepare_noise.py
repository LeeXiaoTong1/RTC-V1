"""Download only the pinned small non-speech catalog; no login, GPU or bulk archive."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from urllib.parse import urlparse
import numpy as np
import soundfile as sf
from .noise import SCHEMA,RECORDED,sha256,validate_records

CATALOG=Path(__file__).with_name('noise_catalog.json')
ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def preparation_lock(path):
    """OS lock releases on exit, kill, or reboot; a leftover file is harmless."""
    with Path(path).open('a+b') as stream:
        stream.seek(0,os.SEEK_END)
        if stream.tell()==0:stream.write(b'\0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Noise preparation is already running for '+str(path.parent)) from exc
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def download(row,path,opener=urllib.request.urlopen,attempts=3):
    """Bounded resume with fixed length/hash; an error never becomes valid data."""
    path=Path(path);expected=int(row['bytes']);url=row['url']
    if expected<=0 or expected>32*1024**2:raise ValueError('Unexpected noise asset size')
    if urlparse(url).scheme!='https':raise ValueError('Only HTTPS catalog downloads are accepted')
    if path.is_file():
        if path.stat().st_size==expected and sha256(path)==row['sha256']:return 'cached'
        raise ValueError('Cached source is corrupt; move it aside before retry: '+str(path))
    part=path.with_suffix(path.suffix+'.part');path.parent.mkdir(parents=True,exist_ok=True)
    error=None
    for attempt in range(attempts):
        try:
            offset=part.stat().st_size if part.is_file() else 0
            if offset==expected and sha256(part)==row['sha256']:
                part.replace(path);return 'resumed'
            if offset>=expected:part.unlink();offset=0
            headers={'User-Agent':'RTC-SDD-research/3.18.4'}
            if offset:headers['Range']=f'bytes={offset}-'
            with opener(urllib.request.Request(url,headers=headers),timeout=30) as response:
                status=getattr(response,'status',200)
                if offset and status==206:
                    if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('Invalid HTTP range response')
                    mode='ab'
                elif status==200:mode='wb';offset=0
                else:raise ValueError('Unexpected HTTP status: '+str(status))
                total=offset
                with part.open(mode) as f:
                    while True:
                        chunk=response.read(256*1024)
                        if not chunk:break
                        total+=len(chunk)
                        if total>expected:raise ValueError('Download exceeds pinned size')
                        f.write(chunk)
            if total!=expected:raise ValueError('Incomplete download')
            if sha256(part)!=row['sha256']:
                part.unlink();raise ValueError('Download checksum mismatch')
            part.replace(path);return 'downloaded'
        except Exception as exc:
            error=exc
            if attempt+1<attempts:time.sleep(min(attempt+1,2))
    raise RuntimeError(f'Download failed after {attempts} attempts: {url}; {error}') from error


def validate_catalog(catalog):
    if catalog.get('schema')!='rtc_v3184_download_catalog_v1':raise ValueError('Unknown download catalog')
    rows=catalog['records'];keys=set();sources={s:set() for s in ('train','dev')}
    for row in rows:
        key=row['key']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',key) or key in keys:raise ValueError('Invalid/duplicate asset key')
        keys.add(key)
        if row['category'] not in RECORDED or row['split'] not in sources:raise ValueError('Invalid scene/split')
        if not row['license'] or not row['original_sources'] or not row['source_group']:
            raise ValueError('Missing source attribution')
        if not re.fullmatch(r'[0-9a-f]{64}',row['sha256']):raise ValueError('Missing pinned SHA256')
        sources[row['split']].update(row['original_sources'])
    if sources['train']&sources['dev']:raise ValueError('Catalog source overlap')
    for split in sources:
        if {r['category'] for r in rows if r['split']==split}!=set(RECORDED):raise ValueError('Catalog lacks a scene')
    return rows


def convert(source,destination,ffmpeg):
    # Conversion happens ONCE. Training does no MP3 decode or resampling for noise.
    result=subprocess.run([ffmpeg,'-nostdin','-v','error','-i',str(source),'-ar','16000','-ac','1',
        '-f','f32le','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45,check=False)
    if result.returncode:raise RuntimeError('Noise decode failed: '+result.stderr.decode(errors='replace')[-500:])
    wave=np.frombuffer(result.stdout,dtype='<f4').copy()
    if len(wave)<3200 or len(wave)>16000*600 or not np.isfinite(wave).all() or np.max(np.abs(wave))<1e-6:
        raise ValueError('Invalid, silent or unexpectedly long scene audio')
    # Preserve amplitudes/dynamics; normalization is performed only during mixing.
    wave=np.clip(wave,-1,1);pcm=np.round(wave*32767).astype('<i2')
    tmp=destination.with_suffix('.tmp.wav');sf.write(tmp,pcm,16000,subtype='PCM_16');tmp.replace(destination)
    return dict(samples=len(wave),seconds=len(wave)/16000,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
        peak=float(np.max(np.abs(wave))),sha256=sha256(destination))


def convert_cached(source,destination,ffmpeg,source_hash,ffmpeg_hash):
    receipt=destination.with_suffix('.json')
    identity=dict(source_sha256=source_hash,ffmpeg_sha256=ffmpeg_hash,conversion='mono16k-pcm16-v1')
    if destination.is_file() and receipt.is_file():
        cached=json.loads(receipt.read_text(encoding='utf-8'))
        if cached.get('identity')==identity and sha256(destination)==cached['info']['sha256']:
            return cached['info']
    info=convert(source,destination,ffmpeg)
    atomic_json(receipt,dict(identity=identity,info=info))
    return info


def summary(records):
    result={}
    for split in ('train','dev'):
        result[split]={}
        for category in RECORDED:
            rows=[r for r in records if r['split']==split and r['category']==category]
            result[split][category]=dict(clips=len(rows),original_groups=len({r['source_group'] for r in rows}),
                seconds=round(sum(r['seconds'] for r in rows),3))
    return result


def prepare(out,catalog_path=CATALOG,jobs=4,ffmpeg=None):
    if not 1<=jobs<=8:raise ValueError('Use 1 to 8 download workers')
    out=Path(out).expanduser().resolve();catalog_path=Path(catalog_path).resolve()
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'));rows=validate_catalog(catalog);catalog_hash=sha256(catalog_path)
    out.mkdir(parents=True,exist_ok=True)
    with preparation_lock(out/'.prepare.lock'):
        manifest=out/'manifest.json'
        if manifest.is_file():
            old=json.loads(manifest.read_text())
            if old.get('catalog_sha256')!=catalog_hash:raise ValueError('Catalog changed; prepare into a new directory')
            from .noise import bind_manifest
            bind_manifest(manifest)
            print('NOISE_MANIFEST='+str(manifest)+'; verified existing complete bank',flush=True)
            return old
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_hash=sha256(ffmpeg)
        (out/'raw').mkdir(exist_ok=True);(out/'audio').mkdir(exist_ok=True)
        print(f'[Noise] {len(rows)} selected files, {sum(r["bytes"] for r in rows)/1024**2:.1f} MiB download; {jobs} workers; no speech/large archives',flush=True)
        def one(row):
            suffix=Path(urlparse(row['url']).path).suffix
            source=out/'raw'/(row['key']+suffix)
            status=download(row,source)
            destination=out/'audio'/(row['key']+'.wav')
            info=convert_cached(source,destination,ffmpeg,row['sha256'],ffmpeg_hash)
            return dict(row,**info,file=destination.relative_to(out).as_posix(),source_sha256=row['sha256'],download_status=status)
        completed=[]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures=[pool.submit(one,row) for row in rows]
            try:
                for future in as_completed(futures):
                    completed.append(future.result())
                    if len(completed)%10==0 or len(completed)==len(rows):print(f'[Noise] ready {len(completed)}/{len(rows)}',flush=True)
            except BaseException:
                for future in futures:future.cancel()
                raise
        completed.sort(key=lambda r:r['key']);records={s:[r for r in completed if r['split']==s] for s in ('train','dev')}
        validate_records(records)
        result=dict(schema=SCHEMA,status='complete',catalog_sha256=catalog_hash,records=completed,summary=summary(completed),
            provenance=catalog['provenance'],ffmpeg_sha256=ffmpeg_hash,
            note='Noise-source diagnostic split. No claim that these are the organizers exact files or that automatic semantic/speech screening is infallible.')
        atomic_json(out/'preparation_report.json',result)
        credits=['V3.18.4 small non-speech scene catalog','']
        for r in rows:
            credits.extend([r['key']+' | '+r['category']+' | '+r['license'],r.get('attribution',''),r['page'],''])
        (out/'ATTRIBUTION.txt').write_text('\n'.join(credits),encoding='utf-8')
        previews=['<!doctype html><meta charset="utf-8"><title>V3.18.4 noise catalog</title><h1>V3.18.4 noise catalog</h1><p>Public non-speech scene assets. Office/coffee are small scene approximations.</p>']
        for r in completed:
            previews.append('<p>'+html.escape(r['category']+' / '+r['split']+' / '+r['key'])+'</p><audio controls preload="none" src="'+html.escape(r['file'])+'"></audio>')
        (out/'preview.html').write_text('\n'.join(previews),encoding='utf-8')
        atomic_json(manifest,result)  # Completion marker is written LAST.
        print('NOISE_MANIFEST='+str(manifest),flush=True)
        print('NOISE_REPORT='+str(out/'preparation_report.json'),flush=True)
        print(json.dumps(result['summary']),flush=True)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(ROOT/'assets'/'noise_v3184'))
    p.add_argument('--catalog',default=str(CATALOG));p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--ffmpeg');p.add_argument('--list-only',action='store_true')
    args=p.parse_args()
    if args.list_only:
        catalog=json.loads(Path(args.catalog).read_text());rows=validate_catalog(catalog)
        print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),categories=dict(Counter(r['category'] for r in rows))),indent=2));return
    prepare(args.out,args.catalog,args.jobs,args.ffmpeg)


if __name__=='__main__':main()
