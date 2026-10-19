"""V4 lifecycle and small diagnostic delivery."""
from datetime import datetime
from pathlib import Path
import secrets
import sys
import traceback
import zipfile
from .common import ROOT,read_json,atomic_json,digest
from w2v_v31810.processes import ensure_idle
from w2v_aasist.launch import run_lock,upload_archive
from .config import parser,initialize,verify,prepare_missing
from .train import run_experiment
from .evaluate import export


def report(run,upload=False):
    archive=run/'report.zip';tmp=run/'report.zip.partial'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(run.iterdir()):
            if file.is_file() and file.suffix in ('.json','.jsonl','.npz','.log'):
                z.write(file,file.name)
    tmp.replace(archive);print('REPORT='+str(archive),flush=True)
    if upload:
        url=upload_archive(archive);(run/'report_url.txt').write_text(url+'\n',encoding='utf-8');print('REPORT_URL='+url,flush=True)


def main():
    args=parser().parse_args();ensure_idle()
    if args.upload_submission:args.submit_best=True
    with run_lock(ROOT/'exp/.aasist-launch.lock'):
        ensure_idle()
        if args.resume:
            run=Path(args.resume).resolve();cfg=verify(run);rows=read_json(run/'train_rows.json');protocol=read_json(run/'validation_protocol.json')
            # Avoid apparently accepted but silently ignored training overrides.
            allowed={'--resume','--submit-best','--upload-submission','--upload-temp','--submission-protocol','--submission-audio-root'}
            if any(x.split('=')[0] not in allowed for x in sys.argv[1:] if x.startswith('--')):
                raise ValueError('Resume uses the recorded configuration; pass only resume and delivery options')
        else:
            prepared=prepare_missing(args)
            if args.prepare_only:
                print('V4_PREPARED='+str(prepared),flush=True)
                return
            run=ROOT/'exp'/('w2v_v4_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(3))
            cfg,rows,protocol=initialize(args,run,prepared)
        (ROOT/'exp/.latest_v4_run').write_text(str(run)+'\n',encoding='utf-8');print('V4_RUN='+str(run),flush=True)
        request=run/'submission_request.json'
        if args.submit_best:
            planned=dict(protocol=str(Path(args.submission_protocol).resolve()),protocol_sha256=digest(args.submission_protocol),
                audio_root=str(Path(args.submission_audio_root).resolve()),upload=args.upload_submission)
            if request.exists() and any(read_json(request)[k]!=planned[k] for k in ('protocol','protocol_sha256','audio_root')):
                raise ValueError('Submission request changed; use standalone export for another protocol')
            atomic_json(request,planned)
        outcome=None
        try:
            if (run/'completed.json').exists():outcome=read_json(run/'completed.json')
            else:outcome=run_experiment(cfg,run,rows,protocol)
            if outcome['status']=='complete' and request.exists():
                req=read_json(request)
                if digest(req['protocol'])!=req['protocol_sha256']:raise ValueError('Submission protocol changed')
                export(run,req['protocol'],req['audio_root'],run.parent/(run.name+'_submission'),cfg['device'],cfg['eval_workers'],req['upload'])
            (run/'failure.log').unlink(missing_ok=True)
        except BaseException:
            (run/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
        finally:
            if not outcome or outcome['status']!='paused':
                failed=sys.exc_info()[0] is not None
                try:report(run,args.upload_temp)
                except Exception as exc:
                    print('REPORT_FAILED='+str(exc),flush=True)
                    if not failed:raise

if __name__=='__main__':main()
