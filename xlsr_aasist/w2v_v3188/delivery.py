"""Best-only export and retryable artifact delivery without repeating inference."""
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
from .common import ROOT,RECIPE,atomic_json,read_json,digest,fingerprint,verify_files


def check_archive(out):
    out=Path(out);meta=read_json(out/'submission_meta.json');archive=out/'submission.zip'
    if meta.get('version')!='3.18.8' or meta.get('recipe')!=RECIPE or meta.get('score')!='P(fake=0)' or meta.get('kind') not in ('best','last'):
        raise ValueError('Expected a V3.18.8 submission export')
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


def export_identity(run, meta):
    """Validate a completed or acknowledged paused checkpoint without torch.load."""
    run=Path(run).resolve()
    completed=run/'completed.json';paused=run/'paused.json'
    done=read_json(completed) if completed.is_file() else read_json(paused) if paused.is_file() else {}
    status=done.get('status')
    if status not in ('complete','paused') or (status=='paused' and meta.get('kind')!='best'):
        raise ValueError('Export requires a completed run or an acknowledged paused best')
    if (Path(meta['run']).resolve()!=run or meta.get('selected')!=done.get(meta['kind']+'_tag') or
        meta.get('checkpoint_sha256')!=done.get('checkpoint_sha256')):
        raise ValueError('Standalone export does not match this saved run checkpoint')
    # Upload retries verify the actual retained checkpoint identity without GPU
    # loading or rerunning inference. A resumed run cannot reuse old scores.
    if not (run/'last.pt').is_file() or digest(run/'last.pt')!=done['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since export; do not reuse stale scores')
    return done


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
        retry='bash upload_w2v_v3188.sh --out '+shlex.quote(str(out))
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
        request=dict(version='3.18.8',recipe=RECIPE,kind='best',protocol=protocol,protocol_sha256=digest(protocol),
            audio_root=audio,out=str(out),count=len(rows),id_order_sha256=fingerprint([r['id'] for r in rows]),
            upload=bool(args.upload_submission or defaults.get('upload',False)))
        if existing and any(request[k]!=existing[k] for k in request if k!='upload'):
            raise ValueError('Saved submission request differs; use standalone export for another evaluation set')
        atomic_json(path,request)
    if request:
        if request.get('version')!='3.18.8' or request.get('recipe')!=RECIPE or request.get('kind')!='best':raise ValueError('Invalid saved submission request')
        verify_files({request['protocol']:request['protocol_sha256']})
        if rows is None:rows=read_protocol(request['protocol'],request['audio_root'],labeled=False)
        if len(rows)!=request['count'] or fingerprint([r['id'] for r in rows])!=request['id_order_sha256']:
            raise ValueError('Planned submission inputs changed')
        from .resources import audit_inference
        atomic_json(run/'submission_resource_audit.json',audit_inference(rows,cfg))
        print('SUBMISSION_AFTER_TRAINING='+str(path),flush=True)
    return request


def finish_request(run,cfg,request,exporter=None):
    if not request:return None
    from .evaluate import export
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
            archive,meta=check_archive(out)
            export_identity(run,meta)
            if (meta['kind']!='best' or meta['selected']!=done['best_tag'] or
                meta['checkpoint_sha256']!=done['checkpoint_sha256'] or
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
        # A manual paused-best export may predate this now-completed run.
        # The completed workflow's verified export becomes the retry target.
        # Keep its actual files, but do not let the old manual pointer route
        # `--run` retries back to an obsolete checkpoint.
        (run/'manual_export_request.json').unlink(missing_ok=True)
        atomic_json(status,dict(status='exported',out=str(out),selected=meta['selected'],zip_sha256=meta['zip_sha256']))
        print('SUBMISSION_ZIP='+str(archive),flush=True)
        url=upload_submission(out) if request['upload'] else None
        result=dict(status='complete',out=str(out),archive=str(archive),selected=meta['selected'],url=url,
                    zip_sha256=meta['zip_sha256'],uploaded=bool(url))
        atomic_json(status,result)
        from .reports import update_links
        update_links(run)
        if url:print('SUBMISSION_TEMP_URL='+url,flush=True)
        (ROOT/'exp').mkdir(parents=True,exist_ok=True)
        (ROOT/'exp'/'.latest_v3188_submission').write_text(str(out)+'\n',encoding='utf-8')
        return result
    except Exception as exc:
        atomic_json(status,dict(status='failed',out=str(out),training_complete=True,error=str(exc)))
        raise


def deliver_export(run,out,report_destination=None,force=False):
    """Deliver BOTH artifacts after a standalone export, with CPU-only retries."""
    from .reports import report,update_links
    run=Path(run).expanduser().resolve();out=Path(out).expanduser().resolve()
    archive,meta=check_archive(out);export_identity(run,meta)
    saved=read_json(run/'report_request.json') if (run/'report_request.json').is_file() else {}
    destination=Path(report_destination or saved.get('destination') or out.parent).expanduser().resolve()
    if destination==run or run in destination.parents:
        raise ValueError('Report destination must be outside the run')
    request=dict(out=str(out),destination=str(destination),selected=meta['selected'],
                 checkpoint_sha256=meta['checkpoint_sha256'],zip_sha256=meta['zip_sha256'])
    request_path=run/'manual_export_request.json'
    if not request_path.is_file() or read_json(request_path)!=request:atomic_json(request_path,request)
    errors=[]
    with run_lock(run/'.delivery-retry.lock'):
        try:
            url=upload_submission(out,force)
            value=dict(status='complete',out=str(out),archive=str(archive),selected=meta['selected'],url=url,
                       zip_sha256=meta['zip_sha256'],uploaded=True)
            status=run/'submission_status.json'
            if not status.is_file() or read_json(status)!=value:atomic_json(status,value)
            print('SUBMISSION_TEMP_URL='+url,flush=True)
        except Exception as exc:
            errors.append('submission: '+str(exc))
            atomic_json(run/'submission_status.json',dict(status='failed',out=str(out),archive=str(archive),error=str(exc)))
        try:report(run,destination,upload=True,force=force)
        except Exception as exc:errors.append('report: '+str(exc))
        links=update_links(run)
    if errors:
        raise RuntimeError('Artifact delivery incomplete. Retry without inference: bash upload_w2v_v3188.sh --run '
                           +shlex.quote(str(run))+'; '+'; '.join(errors))
    return links


def main():
    p=argparse.ArgumentParser(description='Retry existing submission/report uploads without rerunning training or inference')
    target=p.add_mutually_exclusive_group(required=True)
    target.add_argument('--out',help='Existing submission directory; upload scores only')
    target.add_argument('--run',help='targeted augmentation training directory; independently retry submission and report')
    p.add_argument('--force',action='store_true',help='Re-upload even if a link is saved, e.g. after expiry')
    args=p.parse_args()
    if args.out:upload_submission(Path(args.out).expanduser().resolve(),args.force)
    else:
        run=Path(args.run).expanduser().resolve();manual=run/'manual_export_request.json'
        if manual.is_file():
            request=read_json(manual)
            archive,meta=check_archive(request['out'])
            if any(meta[k]!=request[k] for k in ('selected','checkpoint_sha256','zip_sha256')):
                raise ValueError('Standalone export changed since the delivery request')
            deliver_export(run,request['out'],request['destination'],args.force)
        else:
            from .reports import retry_outputs
            retry_outputs(run,args.force)


if __name__=='__main__':main()
