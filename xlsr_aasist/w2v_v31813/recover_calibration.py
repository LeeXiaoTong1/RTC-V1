"""Copy the failed epoch-1 boundary and preserve optimizer/RNG for safe resume."""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import secrets
import shutil
import torch
from w2v_v31810.common import ROOT,atomic_json,atomic_save,digest,fingerprint,read_json,verify_files
from w2v_v31810.identity import verify_registry
from w2v_v31810.processes import ensure_idle
from w2v_aasist.launch import run_lock
from .config import verify_parent,verify
from . import SCHEMA,VERSION

OLD_STEP='a8907306ed231fa8ca7a12c8541d0e6800072f55c2fb3fe4e62c5a714a894188'
ADDED={'calibration.py','recover_calibration.py','test_calibration.py','test_recover_calibration.py'}


def current_code():
    return {str(p.resolve()):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def validate_code(saved,current):
    def keyed(items):
        result={}
        for name,sha in items.items():
            path=Path(name)
            if path.parent.name!='w2v_v31813' or path.name in result:raise ValueError('Unexpected saved code path')
            result[path.name]=(name,sha)
        return result
    old,new=keyed(saved),keyed(current)
    if set(new)-set(old)!=ADDED or set(old)-set(new):raise ValueError('Unexpected code inventory for this specific repair')
    for name,(_,sha) in old.items():
        expected=OLD_STEP if name=='step.py' else new[name][1]
        if sha!=expected:raise ValueError('Unknown or unrelated code change: '+name)


def validate_boundary(cfg,state,failure):
    if cfg.get('version')!=VERSION or cfg.get('recipe')!=SCHEMA or cfg.get('warm_epochs')!=1:
        raise ValueError('Expected the original V3.18.13 one-epoch warmup recipe')
    if state.get('schema')!=SCHEMA or state.get('identity')!=fingerprint(cfg):raise ValueError('Checkpoint/config mismatch')
    history=state.get('history',[])
    if (state.get('epoch')!=1 or state.get('next_step')!=0 or state.get('weights') is not None or state.get('totals') or
        len(history)!=1 or history[0].get('tag')!='epoch_1' or state.get('last_tag')!='epoch_1' or state.get('cursor',0)<=0):
        raise ValueError('Only the completed epoch-1 boundary before any auxiliary update can be recovered')
    if 'No usable prior calibration gradient' not in failure or 'calibrate' not in failure:
        raise ValueError('This recovery is only for the reported prior-gradient calibration failure')
    for key in ('model','best_model','optimizer','grad_scaler','rng'):
        if key not in state:raise ValueError('Incomplete checkpoint: '+key)


def recover(run):
    run=Path(run).expanduser().resolve()
    if (run/'completed.json').exists():raise ValueError('Cannot recover a completed experiment')
    cfg=read_json(run/'config.json');code=current_code();validate_code(cfg['version_code'],code)
    verify_parent(read_json(run/'parent_config.json'))
    verify_files({str(run/name):sha for name,sha in cfg['saved_manifests'].items()})
    verify_registry(read_json(run/'train_rows.json'),read_json(run/'source_registry.json'),cfg,read_json(run/'validation_protocol.json'))
    names={p.name for p in run.iterdir() if p.suffix in ('.json','.jsonl','.npz','.pt')}
    names-= {'completed.json','paused.json','stop_requested.json'}
    names|={'config.json','last.pt','failure.log','training_history.json'}
    for name in names:
        p=run/name
        if p.is_symlink() or not p.is_file() or p.resolve()!=p.absolute():raise ValueError('Expected regular in-run files')
    before={name:digest(run/name) for name in names}
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    validate_boundary(cfg,state,(run/'failure.log').read_text(encoding='utf-8'))
    if read_json(run/'training_history.json')!=state['history']:raise ValueError('Saved history disagrees with checkpoint')
    new=copy.deepcopy(cfg);new.update(version_code=code,calibration_revision='fp32_zero_safe_v1')
    state['identity']=fingerprint(new)
    reserve=cfg.get('free_reserve_bytes',0)+cfg.get('disk_margin_bytes',0)
    if shutil.disk_usage(run.parent).free<2*sum((run/n).stat().st_size for n in names)+reserve:
        raise OSError('Need disk space for a recovery copy plus the configured reserve')
    suffix=datetime.now().strftime('%Y%m%d_%H%M%S')+'_calfix_'+secrets.token_hex(3)
    destination=run.parent/('w2v_v31813_'+suffix);staging=run.parent/('.w2v_v31813_'+suffix+'.partial')
    staging.mkdir(exist_ok=False)
    for name in names-{'config.json','last.pt','failure.log'}:shutil.copyfile(run/name,staging/name)
    atomic_json(staging/'config.json',new);atomic_save(staging/'last.pt',state,reserve)
    (staging/'recovery').mkdir();shutil.copyfile(run/'failure.log',staging/'recovery/original_failure.log')
    verify_files({str(run/name):sha for name,sha in before.items()})
    verify_files({str(staging/name):before[name] for name in names-{'config.json','last.pt','failure.log'}})
    verify(staging)
    atomic_json(staging/'calibration_recovery.json',dict(source_run=str(run),source_files=before,
        old_identity=fingerprint(cfg),new_identity=fingerprint(new),changed_config_fields=['version_code','calibration_revision'],
        completed_epochs=1,next_step=0,total_epochs=cfg['epochs'],
        policy='Keep model, best, optimizer moments, scaler, RNG, cursor and history; remeasure Train auxiliaries in FP32'))
    staging.rename(destination)
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args()
    ensure_idle()
    with run_lock(ROOT/'exp/.aasist-launch.lock'):
        ensure_idle();result=recover(a.run)
    print(result)


if __name__=='__main__':main()
