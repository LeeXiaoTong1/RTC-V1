"""Best-only post-training export and retryable submission-only temp.sh delivery."""
import argparse
from datetime import datetime,timezone
import math
import os
from pathlib import Path
import shlex
import tempfile
from urllib.parse import urlparse
import zipfile
from w2v_aasist.data import read_protocol
from w2v_aasist.launch import upload_archive,run_lock
from .common import ROOT,atomic_json,read_json,digest,fingerprint,verify_files


def check_archive(out):
    out=Path(out);meta=read_json(out/'submission_meta.json');archive=out/'submission.zip'
    if meta.get('version')!='3.18.3-R2' or meta.get('score')!='P(fake=0)' or meta.get('kind') not in ('best','last'):
        raise ValueError('Expected a V3.18.3-R2 submission export')
    if meta['zip_sha256']!=digest(archive):raise ValueError('Submission ZIP hash changed')
    with zipfile.ZipFile(archive) as z:
        if z.namelist()!=['scores.txt']:raise ValueError('Upload must contain only scores.txt')
        lines=z.read('scores.txt').decode('utf-8').splitlines()
    ids=[]
    for line in lines:
        fields=line.split()
        if len(fields)!=2:raise ValueError('Invalid score line')
        value=float(fields[1])
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid fake probability')
        ids.append(fields[0])
    if not ids or len(ids)!=meta['count'] or len(set(ids))!=len(ids) or fingerprint(ids)!=meta['id_order_sha256']:
        raise ValueError('Submission count/order differs from the exported protocol')
    return archive,meta


def valid_url(url):
    parsed=urlparse(url)
    return parsed.scheme=='https' and parsed.netloc=='temp.sh' and parsed.path not in ('','/','/upload')


def upload_submission(out,force=False):
    out=Path(out).resolve()
    if not out.is_dir():raise FileNotFoundError('Submission directory is missing: '+str(out))
    with run_lock(out/'.upload.lock'):
        return _upload_submission(out,force)


def _upload_submission(out,force):
    out=Path(out).resolve();archive,meta=check_archive(out)
    receipt=out/'upload_receipt.json';link=out/'temp_download_url.txt'
    if receipt.is_file() and not force:
        old=read_json(receipt)
        if (old.get('zip_sha256')==meta['zip_sha256'] and old.get('selected')==meta['selected'] and
            old.get('checkpoint_sha256')==meta['checkpoint_sha256'] and valid_url(old.get('url',''))):
            link.write_text(old['url']+'\n',encoding='utf-8')
            print('TEMP_DOWNLOAD_URL='+old['url'],flush=True)
            return old['url']
    try:
        # The established uploader prints TEMP_DOWNLOAD_URL once. Only the
        # verified scores archive is passed, never checkpoint/audio/report files.
        url=upload_archive(archive)
        if not valid_url(url):raise ValueError('Unexpected upload response URL')
        if digest(archive)!=meta['zip_sha256']:raise ValueError('Submission changed during upload')
        atomic_json(receipt,dict(url=url,zip_sha256=meta['zip_sha256'],selected=meta['selected'],
            checkpoint_sha256=meta['checkpoint_sha256'],count=meta['count'],uploaded_at=datetime.now(timezone.utc).isoformat()))
        link.write_text(url+'\n',encoding='utf-8')
        atomic_json(out/'upload_status.json',dict(status='complete',url=url))
        print('TEMP_LINK_FILE='+str(link),flush=True)
        return url
    except Exception as exc:
        atomic_json(out/'upload_status.json',dict(status='failed',error=str(exc),submission_retained=str(archive)))
        retry='bash upload_w2v_v3183r2.sh --out '+shlex.quote(str(out))
        raise RuntimeError(f'Upload failed; submission retained. Retry without GPU inference: {retry}; {exc}') from exc


def prepare_request(args,run,cfg):
    run=Path(run).resolve();path=run/'submission_request.json'
    existing=read_json(path) if path.is_file() else None
    rows=None
    if not args.submit_best:
        request=existing
    else:
        defaults=existing or {}
        base='/home/ubuntu/LXT/RTC/xlsr_aasist/dataset'
        protocol=str(Path(args.submission_protocol or defaults.get('protocol') or os.environ.get('EVAL_PROTOCOL',base+'/progress.txt')).expanduser().resolve())
        audio=str(Path(args.submission_audio_root or defaults.get('audio_root') or os.environ.get('EVAL_AUDIO_ROOT',base+'/wav/progress')).expanduser().resolve())
        out=Path(args.submission_out or defaults.get('out') or Path(args.download_dir)/(run.name+'_submission_best')).expanduser().resolve()
        if out==run or run in out.parents:raise ValueError('Submission must be outside the training run')
        rows=read_protocol(protocol,audio,labeled=False)
        request=dict(version='3.18.3-R2',kind='best',protocol=protocol,protocol_sha256=digest(protocol),
            audio_root=audio,out=str(out),count=len(rows),id_order_sha256=fingerprint([r['id'] for r in rows]),
            upload=bool(args.upload_submission or defaults.get('upload',False)))
        if existing and any(request[k]!=existing[k] for k in request if k!='upload'):
            raise ValueError('Saved submission request differs; use standalone export for another evaluation set')
        atomic_json(path,request)
    if request:
        if request.get('version')!='3.18.3-R2' or request.get('kind')!='best':raise ValueError('Invalid saved submission request')
        verify_files({request['protocol']:request['protocol_sha256']})
        if rows is None:rows=read_protocol(request['protocol'],request['audio_root'],labeled=False)
        if len(rows)!=request['count'] or fingerprint([r['id'] for r in rows])!=request['id_order_sha256']:
            raise ValueError('Planned submission inputs changed')
        print('SUBMISSION_AFTER_TRAINING='+str(path),flush=True)
    return request


def finish_request(run,cfg,request,exporter=None):
    if not request:return None
    from .evaluate import export,selected
    run=Path(run).resolve();out=Path(request['out']).resolve()
    if out==run or run in out.parents:raise ValueError('Submission must be outside the training run')
    verify_files({request['protocol']:request['protocol_sha256']})
    done=read_json(run/'completed.json')
    if done.get('status')!='complete':raise ValueError('Training must complete before best export')
    status=run/'submission_status.json'
    def match_protocol(meta):
        if (meta['protocol_sha256']!=request['protocol_sha256'] or meta['count']!=request['count'] or
            meta['id_order_sha256']!=request['id_order_sha256']):
            raise ValueError('Export differs from the planned evaluation protocol')
        if (meta['kind']!='best' or meta['selected']!=done['best_tag'] or
            meta['checkpoint_sha256']!=done['checkpoint_sha256'] or Path(meta['run']).resolve()!=run or
            meta['calibration'] is None):
            raise ValueError('Export is not the completed run best checkpoint with its recorded calibration')
    try:
        if out.exists():
            # Upload retry reuses validated scores and does not load the GPU model.
            _,weights,identity=selected(run,'best');del weights
            archive,meta=check_archive(out)
            if (meta['kind']!='best' or meta['selected']!=identity['selected'] or
                meta['checkpoint_sha256']!=identity['checkpoint_sha256'] or
                Path(meta['run']).resolve()!=run or meta['calibration'] is None):
                raise ValueError('Existing export belongs to a different best checkpoint or scoring mode')
        else:
            out.parent.mkdir(parents=True,exist_ok=True)
            # Stage on the output filesystem. Failed inference cannot leave a
            # half-written final export directory that blocks a later retry.
            with tempfile.TemporaryDirectory(prefix=out.name+'.pending-',dir=out.parent) as d:
                stage=Path(d)/'export'
                (exporter or export)(run,stage,kind='best',device=cfg['device'],workers=cfg['workers'],
                    protocol=request['protocol'],audio_root=request['audio_root'],announce=False)
                _,meta=check_archive(stage)
                match_protocol(meta)
                stage.replace(out)
            archive=out/'submission.zip'
        match_protocol(meta)
        atomic_json(status,dict(status='exported',out=str(out),selected=meta['selected'],zip_sha256=meta['zip_sha256']))
        print('SUBMISSION_ZIP='+str(archive),flush=True)
        url=upload_submission(out) if request['upload'] else None
        result=dict(status='complete',out=str(out),archive=str(archive),selected=meta['selected'],url=url,
                    zip_sha256=meta['zip_sha256'],uploaded=bool(url))
        atomic_json(status,result)
        (ROOT/'exp').mkdir(parents=True,exist_ok=True)
        (ROOT/'exp'/'.latest_v3183r2_submission').write_text(str(out)+'\n',encoding='utf-8')
        return result
    except Exception as exc:
        atomic_json(status,dict(status='failed',out=str(out),training_complete=True,error=str(exc)))
        raise


def main():
    p=argparse.ArgumentParser(description='Upload an existing verified submission without rerunning inference')
    p.add_argument('--out',required=True);p.add_argument('--force',action='store_true',help='Re-upload even if a link is saved, e.g. after expiry')
    args=p.parse_args();out=Path(args.out).expanduser().resolve()
    upload_submission(out,args.force)


if __name__=='__main__':main()
