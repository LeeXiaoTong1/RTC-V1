"""One atomic adapter/head transaction; the 7B base is referenced exactly once."""
from pathlib import Path
import torch
from w2v_v313.state import (identity,partial_state,apply_partial,to_cpu,atomic_save,capture_rng,restore_rng)
from w2v_v39.common import read_json,digest

SCHEMA='rtc_v316_omni_lora_v1'
KINDS=('best_guarded','best_weighted','last')


def budget(model,auxiliary,cfg):
    p=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    a=sum(p.numel()*p.element_size() for p in auxiliary.parameters())
    one=int((5*p+3*a)*1.15)+cfg['disk_margin_bytes']
    peak=2*one+256*1024**2
    if peak>cfg['maximum_new_peak_bytes']: raise OSError('Partial checkpoint atomic peak exceeds 4 GiB budget')
    return dict(trainable_bytes=p,checkpoint_estimated_bytes=one,atomic_peak_bytes=peak,
        feature_cache_bytes=0,audio_cache_bytes=0,base_copies_per_run=0,
        required_free_bytes=peak+cfg['free_reserve_bytes'])


def load_resume(run,cfg):
    run=Path(run)
    state=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    if state.get('schema')!=SCHEMA or state.get('identity')!=identity(cfg):
        raise ValueError('Omni LoRA checkpoint identity mismatch')
    if state['execution_sha256']!=digest(run/'execution_plan.json'):
        raise ValueError('Execution plan changed after checkpoint commit')
    return state


def choose(state,tag,metrics,eligible,current):
    promoted=[]
    for kind,allowed in (('best_weighted',True),('best_guarded',eligible)):
        if allowed and metrics['weighted_f1']>state['scores'][kind]:
            state['selections'][kind]=tag; state['scores'][kind]=metrics['weighted_f1']; promoted.append(kind)
    if promoted: state['candidates'][tag]=current
    state['candidates']={k:v for k,v in state['candidates'].items() if k in state['selections'].values()}
    return promoted


def load_selected(run,kind='best_guarded'):
    if kind not in KINDS: raise ValueError('Unknown selector')
    run=Path(run).expanduser().resolve(); cfg=read_json(run/'config.json')
    done=read_json(run/'completed.json')
    if done.get('version')!='3.16' or done.get('status')!='complete' or digest(run/'last.pt')!=done['checkpoint_sha256']:
        raise ValueError('Completed checkpoint missing or changed')
    state=load_resume(run,cfg)
    if state['cursor']!=done['committed_updates'] or state['selections']!=done['selections']:
        raise ValueError('Completed selection/cursor mismatch')
    tag=state['last_tag'] if kind=='last' else state['selections'][kind]
    if tag is None or state['cursor']==0:
        raise ValueError('No Omni model passed V3.15 guards. Use the preserved V3.15 best, or explicitly export --checkpoint best_weighted for comparison.')
    weights=state['model'] if kind=='last' else state['candidates'][tag]
    return cfg,weights,dict(done,selected=tag,checkpoint_kind=kind,checkpoint_path=str(run/'last.pt'))
