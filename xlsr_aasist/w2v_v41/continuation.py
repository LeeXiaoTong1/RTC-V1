"""Explicit migration, never a best-weight restart or an implicit config bypass."""
import copy
import shutil
from pathlib import Path
import torch
from w2v_v4.config import verify as verify_v4
from w2v_v4 import SCHEMA as OLD_SCHEMA
from . import VERSION, SCHEMA
from .common import ROOT, read_json, atomic_json, atomic_save, fingerprint, digest
from .config import execution_settings


def migrate(args, run):
    parent=Path(args.continue_from).resolve();run=Path(run).resolve()
    old=verify_v4(parent)
    status=read_json(parent/'paused.json')
    sha=digest(parent/'last.pt')
    if status.get('status')!='paused' or status.get('version')!='4.0' or status.get('checkpoint_sha256')!=sha:
        raise ValueError('Stop V4 safely and wait for PAUSED_RUN before migration')
    state=torch.load(parent/'last.pt',map_location='cpu',weights_only=True)
    if state.get('schema')!=OLD_SCHEMA or state.get('identity')!=fingerprint(old):
        raise ValueError('V4 checkpoint/config identity mismatch')
    required=('model','best_model','optimizer','grad_scaler','rng','weights','history','totals','cursor','epoch','next_step')
    if any(k not in state for k in required):raise ValueError('V4 checkpoint lacks complete continuation state')
    if state['epoch']>=old['epochs']:raise ValueError('The original epoch budget is already complete')
    if run.exists():raise ValueError('Migration destination already exists')
    cfg=copy.deepcopy(old);cfg.update(version=VERSION,recipe=SCHEMA,**execution_settings(args))
    for key in ('workers','eval_workers'):
        if getattr(args,key) is not None:
            if getattr(args,key)<0:raise ValueError('Worker counts must be nonnegative')
            cfg[key]=getattr(args,key)
    cfg['continuation']=dict(parent_run=str(parent),parent_checkpoint_sha256=sha,
        parent_config_sha256=digest(parent/'config.json'),epoch=state['epoch'],next_step=state['next_step'],
        cursor=state['cursor'],optimizer_restored=True,rng_restored=True,learning_rate_restart=False,
        data_and_objective_unchanged=True,total_epochs=old['epochs'])
    cfg['initialization']=dict(old.get('initialization',{}),mode='continue_v4_last',
        detector_weights_loaded=True,optimizer_loaded=True,parent_run=str(parent))
    # Original dependencies remain frozen; add the isolated V4.1 execution code.
    cfg['version_code'].update({str(p.resolve()):digest(p) for p in (ROOT/'w2v_v41').glob('*.py') if not p.name.startswith('test_')})
    run.mkdir(parents=True)
    for name in cfg['saved_manifests']:shutil.copy2(parent/name,run/name)
    for pattern in ('selection_*.json','select_*.npz','coverage_epoch_*.json','diagnostic_*.json'):
        for p in parent.glob(pattern):shutil.copy2(p,run/p.name)
    for name in ('training_history.json','steps.jsonl','auxiliary_calibration.json','model_inventory.json','submission_request.json'):
        if (parent/name).is_file():shutil.copy2(parent/name,run/name)
    if digest(parent/'last.pt')!=sha:raise ValueError('V4 checkpoint changed during migration')
    state.update(schema=SCHEMA,identity=fingerprint(cfg))
    atomic_json(run/'config.json',cfg)
    atomic_save(run/'last.pt',state,cfg.get('free_reserve_bytes',0))
    atomic_json(run/'migration.json',cfg['continuation'])
    print(f'V4.1 continues V4: completed epochs={state["epoch"]}, next step={state["next_step"]}, total epochs={cfg["epochs"]}; optimizer/LR/PCL/sampling restored.',flush=True)
    return cfg,read_json(run/'train_rows.json'),read_json(run/'validation_protocol.json')
