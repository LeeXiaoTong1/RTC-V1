"""Retryable report delivery and separately named artifact links; no GPU inference."""
from datetime import datetime,timezone
from pathlib import Path
import re
import zipfile
from w2v_aasist.launch import run_lock,upload_archive
from .common import RECIPE,atomic_json,read_json,digest,fingerprint

EXCLUDED={'report_status.json','report_upload_receipt.json','report_temp_download_url.txt',
          'artifact_links.json','download_links.txt','REPORT_MANIFEST.json'}
SCORE_FILE=re.compile(r'(?:dev_scores_epoch_\d+_step_\d+|scene_scores_epoch_\d+_step_\d+|scene_baseline_scores|parent_baseline_scores|parent_reference_scores|robust_scores_[A-Za-z0-9_.-]+)\.npz')


def prepare_request(args,run,argv):
    run=Path(run).resolve();path=run/'report_request.json'
    old=read_json(path) if path.is_file() else None
    explicit=any(arg.split('=')[0]=='--download-dir' for arg in argv)
    destination=str(Path(args.download_dir).expanduser().resolve()) if not old or explicit else old['destination']
    if Path(destination)==run or run in Path(destination).parents:
        raise ValueError('Keep the report destination outside the training run')
    request=dict(destination=destination,upload=bool(args.upload_temp or (old or {}).get('upload',False)))
    if request!=old:atomic_json(path,request)
    return request


def update_links(run):
    from .delivery import valid_url
    run=Path(run)
    with run_lock(run/'.artifact-links.lock'):
        links={}
        for kind in ('submission','report'):
            path=run/(kind+'_status.json');value=read_json(path) if path.is_file() else {}
            url=value.get('url','')
            links[kind+'_url']=url if valid_url(url or '') else None
            links[kind+'_status']=value.get('status','pending')
            links[kind+'_archive']=value.get('archive')
        atomic_json(run/'artifact_links.json',links)
        text=''.join(kind.upper()+'_URL='+(links[kind+'_url'] or 'PENDING')+'\n' for kind in ('submission','report'))
        temporary=run/'download_links.txt.tmp';temporary.write_text(text,encoding='ascii')
        temporary.replace(run/'download_links.txt')
    return links


def report_files(run):
    rows=[]
    for path in sorted(run.rglob('*')):
        relative=path.relative_to(run)
        if path.name in EXCLUDED or path.name.startswith('.') or not path.is_file():continue
        if path.suffix not in ('.json','.jsonl','.txt','.log','.md','.csv','.html','.npz'):continue
        if path.suffix=='.npz' and (len(relative.parts)!=1 or not SCORE_FILE.fullmatch(path.name)):continue
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p!=run and run in p.parents):
            raise ValueError('Report files must not be symlinks: '+str(relative))
        if not path.resolve().is_relative_to(run):raise ValueError('Report file escapes its run')
        rows.append(dict(file=relative.as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    return rows


def report(run,destination,upload=False,force=False):
    from .delivery import valid_url
    run=Path(run).expanduser().resolve();destination=Path(destination).expanduser().resolve()
    cfg=read_json(run/'config.json')
    if cfg.get('version')!='3.19.1' or cfg.get('recipe')!=RECIPE:raise ValueError('Expected this V3.19.1 condition run')
    if destination==run or run in destination.parents:raise ValueError('Report destination must be outside the run')
    destination.mkdir(parents=True,exist_ok=True)
    archive=destination/(run.name+'_report.zip');status=run/'report_status.json'
    with run_lock(run/'.report.lock'):
        try:
            print('[Report] packaging training metrics and diagnostics',flush=True)
            rows=report_files(run);identity=fingerprint(rows)
            old=read_json(status) if status.is_file() else {}
            reusable=(old.get('archive')==str(archive) and old.get('content_identity')==identity and archive.is_file()
                      and old.get('zip_sha256')==digest(archive))
            if not reusable:
                temporary=archive.with_suffix('.zip.tmp')
                with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
                    for row in rows:
                        path=run/row['file'];before=(path.stat().st_size,path.stat().st_mtime_ns)
                        z.write(path,row['file'])
                        if before!=(path.stat().st_size,path.stat().st_mtime_ns) or digest(path)!=row['sha256']:
                            raise ValueError('Report input changed while packaging: '+row['file'])
                    import json
                    z.writestr('REPORT_MANIFEST.json',json.dumps(dict(version='3.19.1',recipe=RECIPE,
                        content_identity=identity,files=rows),sort_keys=True,indent=2))
                temporary.replace(archive)
            archive_hash=digest(archive)
            result=dict(status='exported',archive=str(archive),zip_sha256=archive_hash,content_identity=identity,
                        file_count=len(rows),uploaded=False,url=None)
            atomic_json(status,result);print('REPORT_ZIP='+str(archive),flush=True)
            if upload:
                print('[Report] uploading report ZIP',flush=True)
                receipt=run/'report_upload_receipt.json';previous=read_json(receipt) if receipt.is_file() else {}
                if not force and previous.get('zip_sha256')==archive_hash and valid_url(previous.get('url','')):
                    url=previous['url']
                else:
                    url=upload_archive(archive)
                    if not valid_url(url):raise ValueError('Unexpected report upload URL')
                    if digest(archive)!=archive_hash:raise ValueError('Report changed during upload')
                    atomic_json(receipt,dict(url=url,zip_sha256=archive_hash,archive=str(archive),
                        uploaded_at=datetime.now(timezone.utc).isoformat()))
                (run/'report_temp_download_url.txt').write_text(url+'\n',encoding='ascii')
                result.update(status='complete',uploaded=True,url=url);atomic_json(status,result)
                print('REPORT_TEMP_URL='+url,flush=True)
            update_links(run)
            return archive
        except Exception as exc:
            previous=read_json(status) if status.is_file() else {}
            atomic_json(status,dict(previous,status='failed',url=None,error=str(exc),archive=str(archive)))
            update_links(run)
            import shlex
            raise RuntimeError('Report delivery failed; saved training retained. Retry without training or inference: bash upload_w2v_v3191.sh --run '
                               +shlex.quote(str(run))+'; '+str(exc)) from exc


def retry_outputs(run,force=False):
    """Try both artifacts independently; do not load checkpoints or audio."""
    from .delivery import upload_submission,check_archive,export_identity
    run=Path(run).expanduser().resolve();errors=[]
    with run_lock(run/'.delivery-retry.lock'):
        try:
            request=read_json(run/'submission_request.json');out=Path(request['out'])
            archive,meta=check_archive(out);done=export_identity(run,meta)
            if (Path(meta['run']).resolve()!=run or meta['kind']!='best'
                or meta['selected']!=done['best_tag'] or meta['checkpoint_sha256']!=done['checkpoint_sha256']
                or meta['protocol_sha256']!=request['protocol_sha256'] or meta['count']!=request['count']
                or meta['id_order_sha256']!=request['id_order_sha256'] or meta.get('calibration') is None):
                raise ValueError('Existing submission does not match the completed run best')
            url=upload_submission(out,force)
            value=dict(status='complete',out=str(out),archive=str(archive),selected=meta['selected'],url=url,
                       zip_sha256=meta['zip_sha256'],uploaded=True)
            if not (run/'submission_status.json').is_file() or read_json(run/'submission_status.json')!=value:
                atomic_json(run/'submission_status.json',value)
            print('SUBMISSION_TEMP_URL='+url,flush=True)
        except Exception as exc:
            errors.append('submission: '+str(exc));atomic_json(run/'submission_status.json',dict(status='failed',error=str(exc)))
        try:
            request=read_json(run/'report_request.json') if (run/'report_request.json').is_file() else dict(destination='/home/ubuntu/LXT/temp')
            report(run,request['destination'],upload=True,force=force)
        except Exception as exc:errors.append('report: '+str(exc))
        update_links(run)
    if errors:raise RuntimeError('; '.join(errors))
    return read_json(run/'artifact_links.json')
