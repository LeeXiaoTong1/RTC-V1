"""Seal the last atomic R2 epoch without modifying the resumable original run."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import torch
from w2v_aasist.full_workflow import ensure_idle
from w2v_aasist.launch import run_lock
from w2v_v3183r2.common import ROOT,read_json,atomic_json,digest,fingerprint,schema_for,verify_files
from w2v_v3183r2.config import verify_inputs
from w2v_v3183r2.sampling import Plan
from w2v_v3183r2.records import dev_partition
from w2v_v3183r2.monitor import measure
from w2v_v3183r2.calibration import choose
from w2v_v3183r2.parent import PARENT_ARTIFACTS


def authenticate_supervisor(root,run,pid,proc=Path('/proc')):
    root=Path(root).resolve();run=Path(run).resolve();entry=Path(proc)/str(pid)
    if pid<=1:raise ValueError('Invalid supervisor PID')
    log=Path((root/'exp'/'.latest_v3183r2_log').read_text().strip()).resolve()
    latest=Path((root/'exp'/'.latest_v3183r2_run').read_text().strip()).resolve()
    args=(entry/'cmdline').read_bytes().decode().strip('\0').split('\0')
    if latest!=run or len(args)<3 or Path(args[0]).name!='bash' or args[1:3]!=[str(root/'v3183r2_supervise.sh'),str(log)]:
        raise ValueError('Refusing to signal a PID that is not the recorded R2 supervisor for this run')
    if (entry/'cwd').resolve()!=root or entry.stat().st_uid!=os.getuid() or os.getpgid(pid)!=pid:
        raise ValueError('Supervisor owner, directory or session identity changed')
    if not any(s.rstrip()==('V3183R2_RUN='+str(run)) for s in log.read_text(errors='replace').splitlines()):
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
    pointer=Path(root)/'exp'/'.latest_v3183r2_pid'
    if not pointer.is_file():
        ensure_idle();return
    pid=int(pointer.read_text().strip())
    if not Path('/proc',str(pid)).exists():
        ensure_idle();return
    authenticate_supervisor(root,run,pid)
    print(f'[Stop] verified R2 supervisor PID={pid}; sending TERM to its process group',flush=True)
    os.killpg(pid,signal.SIGTERM)
    deadline=time.monotonic()+30
    while group_live(pid) and time.monotonic()<deadline:time.sleep(.25)
    if group_live(pid):raise RuntimeError('R2 has not exited; no checkpoint was sealed. Check the recorded process before retrying.')
    ensure_idle()


def validate_state(run,cfg,state):
    run=Path(run)
    if cfg.get('version')!='3.18.3-R2' or state.get('schema')!=schema_for(cfg) or state.get('identity')!=fingerprint(cfg):
        raise ValueError('R2 checkpoint/config identity mismatch')
    verify_files({str(run/n):h for n,h in cfg['saved_manifests'].items()})
    train=read_json(run/'train_rows.json');dev=read_json(run/'dev_rows.json')
    steps=Plan(train,cfg['stream_sources'],cfg['seed'],cfg['group_quotas']).steps
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
    if cfg.get('version')!='3.18.3-R2':raise ValueError('Only full-layer R2 parents can be sealed')
    print('[Seal] verifying R2 inputs and the last committed checkpoint',flush=True)
    if verify:verify_inputs(cfg)
    before={n:digest(run/n) for n in PARENT_ARTIFACTS if n!='completed.json'}
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    dev,partition=validate_state(run,cfg,state)
    if (run/'completed.json').is_file():
        from w2v_v3183r2.evaluate import selected
        selected(run,'best');out=run
    else:
        out=run.with_name(run.name+'_sealed_e'+str(state['epoch'])+'_'+before['last.pt'][:10])
        if out.exists():
            receipt=read_json(out/'seal_receipt.json')
            if receipt['source_run']!=str(run) or receipt['source_files']!=before:
                raise ValueError('Existing sealed snapshot has a different source')
            verify_files({str(out/n):h for n,h in receipt['sealed_files'].items()})
        else:
            if shutil.disk_usage(run).free<(run/'last.pt').stat().st_size+1024**3:
                raise OSError('Need checkpoint snapshot space plus 1 GiB reserve')
            with tempfile.TemporaryDirectory(prefix=out.name+'.pending-',dir=out.parent) as temporary:
                stage=Path(temporary)/'snapshot';stage.mkdir()
                for n in before:shutil.copy2(run/n,stage/n)
                fitted={}
                for kind in ('best','last'):
                    tag=state[kind+'_tag']
                    if tag not in fitted:fitted[tag]=choose(dev,state[kind+'_logits'].numpy(),partition)
                    atomic_json(stage/f'calibration_{kind}.json',dict(fitted[tag],checkpoint_sha256=before['last.pt'],selected=tag))
                done=dict(version='3.18.3-R2',status='complete',best_tag=state['best_tag'],last_tag=state['last_tag'],
                    checkpoint_sha256=before['last.pt'],completed_epochs=state['epoch'],planned_epochs=cfg['epochs'],
                    stopped_early=state['epoch']<cfg['epochs'],completion_reason='user_sealed_last_committed_epoch',
                    source_run=str(run),baseline_fallback=False,selection=cfg['selection'],initialization=cfg['initialization'])
                atomic_json(stage/'completed.json',done);atomic_json(stage/'training_history.json',state['history'])
                verify_files({str(stage/n):h for n,h in before.items()})
                verify_files({str(run/n):h for n,h in before.items()})
                sealed={p.name:digest(p) for p in stage.iterdir() if p.is_file()}
                atomic_json(stage/'seal_receipt.json',dict(source_run=str(run),source_files=before,sealed_files=sealed,
                    note='Original run unchanged and resumable; snapshot contains only the last committed epoch and its saved best.'))
                stage.replace(out)
    verify_files({str(run/n):h for n,h in before.items()})
    if pointer:
        pointer=Path(pointer);pointer.parent.mkdir(parents=True,exist_ok=True)
        temp=pointer.with_suffix('.pending');temp.write_text(str(out)+'\n',encoding='utf-8');temp.replace(pointer)
    print('V3184_PARENT_RUN='+str(out),flush=True)
    print('PARENT_BEST='+state['best_tag']+'; last_committed='+state['last_tag']+'; original_run_preserved=True',flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run');p.add_argument('--stop-running',action='store_true')
    args=p.parse_args();run=Path(args.run or (ROOT/'exp'/'.latest_v3183r2_run').read_text().strip()).resolve()
    if os.name!='posix':raise RuntimeError('Run this command on the Linux training server')
    if args.stop_running:stop_latest(ROOT,run)
    ensure_idle()
    with run_lock(ROOT/'exp'/'.aasist-launch.lock'):
        ensure_idle();seal(run,pointer=ROOT/'exp'/'.latest_v3184_parent_run')


if __name__=='__main__':main()
