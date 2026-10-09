"""Atomic adapter/head checkpoints referencing one immutable public encoder."""
from pathlib import Path
import torch
from w2v_v313.state import (identity,partial_state,apply_partial,to_cpu,atomic_save,capture_rng,restore_rng)
from w2v_v3161.state import metric_key,storage_budget
from w2v_v39.common import read_json,digest

SCHEMA='rtc_v317_lora_forensic_v1'
KINDS=('best','best_weighted','last')


def promote(state,tag,current,metrics,origin):
    key=metric_key(metrics)
    if state['best_metrics'] is None or key>metric_key(state['best_metrics']):
        state.update(best_tag=tag,best_model=current,best_metrics=metrics,best_origin=origin)
        return True
    return False


def load_resume(run,cfg):
    run=Path(run); state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    if state.get('schema')!=SCHEMA or state.get('identity')!=identity(cfg):
        raise ValueError('V3.17 checkpoint/configuration differs')
    if state['execution_plan_sha256']!=digest(run/'execution_plan.json'):
        raise ValueError('Recorded numerical execution changed')
    return state


def load_selected(run,kind='best'):
    if kind not in KINDS: raise ValueError('Choose best, best_weighted or last')
    run=Path(run).expanduser().resolve();cfg=read_json(run/'config.json');done=read_json(run/'completed.json')
    if done.get('version')!='3.17' or done.get('status')!='complete':raise ValueError('Completed V3.17 run required')
    if digest(run/'last.pt')!=done['checkpoint_sha256']:raise ValueError('Checkpoint content changed')
    state=load_resume(run,cfg)
    if not state['history'] or state['history'][-1]['cursor']!=state['cursor']:
        raise ValueError('No committed V3.17 validation')
    if (state['cursor']!=done['committed_updates'] or state['last_tag']!=done['last_tag'] or state['best_tag']!=done['best_tag']):
        raise ValueError('Completion selection/cursor differs')
    selected=state['last_tag'] if kind=='last' else state['best_tag']
    matches=[e for e in state['history'] if e['tag']==selected and e['committed']]
    if len(matches)!=1 or not selected.startswith('epoch_'):raise ValueError('Only this run\'s trained checkpoints are eligible')
    weights=state['model'] if kind=='last' else state['best_model']
    meta=dict(done,selected=selected,origin=dict(run=str(run),version='3.17',tag=selected),
        checkpoint_kind=kind,checkpoint_path=str(run/'last.pt'),baseline_fallback=False)
    return dict(config=cfg,candidate=dict(kind='partial',state=weights)),meta
