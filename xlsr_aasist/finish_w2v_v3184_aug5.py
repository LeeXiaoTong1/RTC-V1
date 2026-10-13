"""Seal the last atomic AUG5 epoch without modifying the resumable original run."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from types import SimpleNamespace
import torch
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.launch import run_lock
from w2v_v3184_aug5.common import RECIPE,ROOT,read_json,atomic_json,digest,fingerprint,schema_for,verify_files
from w2v_v3184_aug5.config import verify_inputs
from w2v_v3184_aug5.sampling import Plan
from w2v_v3184.records import dev_partition
from w2v_v3184.monitor import measure
from w2v_v3184.calibration import choose
from w2v_v3183r2.parent import PARENT_ARTIFACTS
from w2v_v3184_aug5.reports import report_files


def authenticate_supervisor(root,run,pid,proc=Path('/proc')):
    root=Path(root).resolve();run=Path(run).resolve();entry=Path(proc)/str(pid)
    if pid<=1:raise ValueError('Invalid supervisor PID')
    log=Path((root/'exp'/'.latest_v3184_aug5_log').read_text().strip()).resolve()
    latest=Path((root/'exp'/'.latest_v3184_aug5_run').read_text().strip()).resolve()
    args=(entry/'cmdline').read_bytes().decode().strip('\0').split('\0')
    if latest!=run or len(args)<3 or Path(args[0]).name!='bash' or args[1:3]!=[str(root/'v3184_aug5_supervise.sh'),str(log)]:
        raise ValueError('Refusing to signal a PID that is not the recorded AUG5 supervisor for this run')
    if (entry/'cwd').resolve()!=root or entry.stat().st_uid!=os.getuid() or os.getpgid(pid)!=pid:
        raise ValueError('Supervisor owner, directory or session identity changed')
    if not any(s.rstrip()==('V3184_RUN='+str(run)) for s in log.read_text(errors='replace').splitlines()):
        raise ValueError('Supervisor log is not bound to the requested run')
    return args


def group_live(pgid):
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            stat=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0]!='Z' and os.getpgid(int(entry.name))==pgid:return True
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return False


def stop_latest(root,run):
    cfg=read_json(Path(run)/'config.json')
    if cfg.get('version')!='3.18.4' or cfg.get('recipe')!=RECIPE:
        raise ValueError('Refusing to stop a non-AUG5 run')
    pointer=Path(root)/'exp'/'.latest_v3184_aug5_pid'
    if not pointer.is_file():
        ensure_idle();return
    pid=int(pointer.read_text().strip())
    if not Path('/proc',str(pid)).exists():
        ensure_idle();return
    authenticate_supervisor(root,run,pid)
    print(f'[Stop] verified AUG5 supervisor PID={pid}; sending TERM to its process group',flush=True)
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:pass
    deadline=time.monotonic()+30
    while group_live(pid) and time.monotonic()<deadline:time.sleep(.25)
    if group_live(pid):raise RuntimeError('AUG5 has not exited; no checkpoint was sealed. Check the recorded process before retrying.')
    ensure_idle()


def diagnostic_files(run,state):
    """Copy committed-epoch diagnostics only; never audio, weights or delivery receipts."""
    fixed={'input_audit.json','parameter_inventory.json','initialization_audit.json',
           'parent_baseline.json','scene_baseline.json','scene_baseline_scores.npz','scene_panel.json'}
    for entry in state['history']:
        tag=entry['tag'];epoch=entry['epoch']
        fixed.update({f'dev_scores_{tag}.npz',f'scene_scores_{tag}.npz',
                      f'sampling_and_loss_{tag}.json',f'performance_{tag}.json',f'coverage_epoch_{epoch}.json'})
    # Reuse the report path/link guards, but only include this explicit inventory.
    return {r['file']:r['sha256'] for r in report_files(run) if r['file'] in fixed}


def validate_state(run,cfg,state):
    run=Path(run)
    if (cfg.get('version')!='3.18.4' or cfg.get('recipe')!=RECIPE) or state.get('schema')!=schema_for(cfg) or state.get('identity')!=fingerprint(cfg):
        raise ValueError('AUG5 checkpoint/config identity mismatch')
    verify_files({str(run/n):h for n,h in cfg['saved_manifests'].items()})
    train=read_json(run/'train_rows.json');dev=read_json(run/'dev_rows.json')
    steps=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas'],cfg['view_factor']).steps
    epoch=state.get('epoch',0);history=state.get('history',[])
    if not 1<=epoch<=cfg['epochs'] or state.get('cursor')!=epoch*steps or len(history)!=epoch:
        raise ValueError('No complete atomic epoch, or checkpoint cursor/history is inconsistent')
    tags=[f'epoch_{i}_step_{steps}' for i in range(1,epoch+1)]
    if [h.get('tag') for h in history]!=tags or [h.get('epoch') for h in history]!=list(range(1,epoch+1)) or state['last_tag']!=tags[-1]:
        raise ValueError('Committed epoch sequence changed')
    def key(m):
        values=tuple(float(m[n]) for n in ('weighted_f1','noisy_f1','clean_f1'))
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite selection metrics')
        return values
    best=max(history,key=lambda h:key(h['selection_metrics']))
    if state['best_tag']!=best['tag'] or key(state['best_metrics'])!=key(best['selection_metrics']):
        raise ValueError('Saved best is not the best committed selection epoch')
    partition=dev_partition(dev,cfg['dev_pairs'],cfg['seed'],cfg['calibration_fraction'])
    if partition!=read_json(run/'dev_partition.json'):raise ValueError('Dev partition changed')
    ix=partition['select'];rows=[dev[i] for i in ix]
    for kind,entry in (('best',best),('last',history[-1])):
        logits=state[kind+'_logits']
        if not isinstance(logits,torch.Tensor) or logits.shape!=(len(dev),2) or not torch.isfinite(logits).all():
            raise ValueError('Missing/invalid committed scores: '+kind)
        actual=key(measure(rows,logits.numpy()[ix]));recorded=key(entry['selection_metrics'])
        if any(abs(a-b)>1e-9 for a,b in zip(actual,recorded)):raise ValueError('Cached scores disagree with selection: '+kind)
        values=state['best_model' if kind=='best' else 'model']
        if not values or any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in values.values()):
            raise ValueError('Invalid committed model tensors: '+kind)
    return dev,partition


def seal(run,verify=True,pointer=None):
    run=Path(run).expanduser().resolve();cfg=read_json(run/'config.json')
    if (cfg.get('version')!='3.18.4' or cfg.get('recipe')!=RECIPE):raise ValueError('Only V3.18.4 AUG5 runs can be sealed')
    print('[2/4] Verifying inputs and the last committed AUG5 checkpoint',flush=True)
    if verify:verify_inputs(cfg)
    names=set(PARENT_ARTIFACTS)-{'completed.json'}
    names.update(cfg['saved_manifests'])
    for n in names:
        if Path(n).name!=n or (run/n).is_symlink():
            raise ValueError('Expected regular run-local checkpoint artifacts: '+n)
    before={n:digest(run/n) for n in sorted(names)}
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    dev,partition=validate_state(run,cfg,state)
    if (run/'completed.json').is_file():
        from w2v_v3184_aug5.evaluate import selected
        selected(run,'best');out=run
    else:
        out=run.with_name(run.name+'_sealed_e'+str(state['epoch'])+'_'+before['last.pt'][:10])
        if out.exists():
            receipt=read_json(out/'seal_receipt.json')
            if receipt['source_run']!=str(run) or receipt['source_files']!=before:
                raise ValueError('Existing sealed snapshot has a different source')
            verify_files({str(out/n):h for n,h in receipt['sealed_files'].items()})
        else:
            diagnostics=diagnostic_files(run,state)
            files=dict(before,**diagnostics)
            if shutil.disk_usage(run).free<sum((run/n).stat().st_size for n in files)+1024**3:
                raise OSError('Need checkpoint snapshot space plus 1 GiB reserve')
            with tempfile.TemporaryDirectory(prefix=out.name+'.pending-',dir=out.parent) as temporary:
                stage=Path(temporary)/'snapshot';stage.mkdir()
                for n in files:shutil.copy2(run/n,stage/n)
                fitted={}
                for kind in ('best','last'):
                    tag=state[kind+'_tag']
                    if tag not in fitted:fitted[tag]=choose(dev,state[kind+'_logits'].numpy(),partition)
                    atomic_json(stage/f'calibration_{kind}.json',dict(fitted[tag],checkpoint_sha256=before['last.pt'],selected=tag))
                done=dict(version='3.18.4',recipe=RECIPE,pooling=cfg['pooling'],status='complete',best_tag=state['best_tag'],last_tag=state['last_tag'],
                    checkpoint_sha256=before['last.pt'],completed_epochs=state['epoch'],planned_epochs=cfg['epochs'],
                    stopped_early=state['epoch']<cfg['epochs'],completion_reason='user_sealed_last_committed_epoch',
                    source_run=str(run),baseline_fallback=False,selection=cfg['selection'],initialization=cfg['initialization'],
                    parent_baseline=state['baseline'],
                    best_selection_delta=state['best_metrics']['weighted_f1']-state['baseline']['selection_metrics']['weighted_f1'])
                atomic_json(stage/'completed.json',done);atomic_json(stage/'training_history.json',state['history'])
                with (stage/'steps.jsonl').open('w',encoding='utf-8') as stream:
                    for entry in state.get('steps',[]):stream.write(json.dumps(entry,sort_keys=True)+'\n')
                verify_files({str(stage/n):h for n,h in files.items()})
                verify_files({str(run/n):h for n,h in files.items()})
                sealed={p.name:digest(p) for p in stage.iterdir() if p.is_file()}
                atomic_json(stage/'seal_receipt.json',dict(source_run=str(run),source_files=before,sealed_files=sealed,
                    note='Original run unchanged and resumable; snapshot contains only the last committed epoch and its saved best.'))
                stage.replace(out)
    verify_files({str(run/n):h for n,h in before.items()})
    if pointer:
        pointer=Path(pointer);pointer.parent.mkdir(parents=True,exist_ok=True)
        temp=pointer.with_suffix('.pending');temp.write_text(str(out)+'\n',encoding='utf-8');temp.replace(pointer)
    print('V3184_EXPORT_RUN='+str(out),flush=True)
    print('EXPORT_BEST='+state['best_tag']+'; last_committed='+state['last_tag']+'; original_run_preserved=True',flush=True)
    return out


def deliver(run,source_run,args,exporter=None):
    """Generate saved BEST once. Retry uploads independently without repeating inference."""
    from w2v_v3184_aug5.delivery import prepare_request,finish_request
    from w2v_v3184_aug5.reports import report,update_links
    run=Path(run).resolve();source_run=Path(source_run).resolve()
    cfg=read_json(run/'config.json');errors=[]
    # Keep the original protocol choice, while binding a sealed snapshot to its own output.
    source_request=source_run/'submission_request.json'
    original=read_json(source_request) if source_request.is_file() else {}
    original_report=source_run/'report_request.json'
    saved_report=read_json(original_report) if original_report.is_file() else {}
    existing_report=read_json(run/'report_request.json') if (run/'report_request.json').is_file() else {}
    destination=str(Path(args.download_dir or existing_report.get('destination') or
                         saved_report.get('destination') or '/home/ubuntu/LXT/temp').expanduser().resolve())
    if Path(destination)==run or run in Path(destination).parents:
        raise ValueError('Report destination must be outside the run')
    params=SimpleNamespace(submit_best=True,upload_submission=args.upload_temp,submission_out=None,
        download_dir=destination,
        submission_protocol=args.submission_protocol or original.get('protocol'),
        submission_audio_root=args.submission_audio_root or original.get('audio_root'))
    atomic_json(run/'report_request.json',dict(destination=destination,
        upload=bool(args.upload_temp or existing_report.get('upload',False))))
    try:
        print('[3/4] Exporting saved BEST and delivering submission',flush=True)
        if original:verify_files({original['protocol']:original['protocol_sha256']})
        request=prepare_request(params,run,cfg)
        if original and any(request[k]!=original[k] for k in
                            ('protocol','protocol_sha256','audio_root','count','id_order_sha256')):
            raise ValueError('Submission protocol differs from the original recorded request')
        finish_request(run,cfg,request,exporter)
    except Exception as exc:
        errors.append('submission: '+str(exc));print('SUBMISSION_FAILED='+str(exc),flush=True)
    try:
        print('[4/4] Packaging and delivering report',flush=True)
        report(run,destination,upload=read_json(run/'report_request.json')['upload'])
    except Exception as exc:
        errors.append('report: '+str(exc));print('REPORT_FAILED='+str(exc),flush=True)
    links=update_links(run)
    print((run/'download_links.txt').read_text().strip(),flush=True)
    print('LINKS_FILE='+str(run/'download_links.txt'),flush=True)
    if errors:raise RuntimeError('; '.join(errors))
    return links


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',help='Defaults to the recorded latest AUG5 run')
    p.add_argument('--stop-running',action='store_true',help='TERM only the authenticated AUG5 supervisor group')
    p.add_argument('--seal-only',action='store_true')
    p.add_argument('--upload-temp',action='store_true',help='Upload submission and report to temp.sh')
    p.add_argument('--submission-protocol');p.add_argument('--submission-audio-root');p.add_argument('--download-dir')
    args=p.parse_args()
    if os.name!='posix':raise RuntimeError('Run this command on the Linux training server')
    source=Path(args.run or (ROOT/'exp'/'.latest_v3184_aug5_run').read_text().strip()).expanduser().resolve()
    print('[1/4] Checking the recorded AUG5 training process',flush=True)
    if args.stop_running:stop_latest(ROOT,source)
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle()
        snapshot=seal(source,pointer=ROOT/'exp'/'.latest_v3184_aug5_export_run')
        if not args.seal_only:
            # seal already verified pinned inputs under this same exclusive lock.
            # Avoid a second hash pass over the 3B checkpoint/noise inventory.
            from functools import partial
            from w2v_v3184_aug5.evaluate import export
            deliver(snapshot,source,args,exporter=partial(export,verify=False))


if __name__=='__main__':main()

