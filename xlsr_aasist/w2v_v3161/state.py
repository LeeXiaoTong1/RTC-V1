"""One bounded transaction for LAST, Adam, TFCL and at most one selected best."""
from pathlib import Path
import math
import torch
from w2v_v313.state import (identity, partial_state, apply_partial, to_cpu, atomic_save,
                            capture_rng, restore_rng)
from w2v_v39.common import digest, read_json
from w2v_v316_tfcl.state import load_selected as source_selected

SCHEMA = 'rtc_v3161_ssl_tfcl_continuation_v1'
KINDS = ('best', 'best_weighted', 'last')


def metric_key(metrics):
    values = tuple(float(metrics[k]) for k in ('weighted_f1', 'noisy_f1', 'clean_f1'))
    if not metrics.get('complete') or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
        raise ValueError('Best selection requires complete finite fixed Dev metrics')
    return values


def promote(state, tag, current, metrics, origin):
    if metric_key(metrics) > metric_key(state['best_metrics']):
        state.update(best_tag=tag, best_model=current, best_metrics=metrics, best_origin=origin)
        return True
    return False


def source_state(cfg):
    # The old strict loader proves that LAST is an actually committed V3.16 model.
    checkpoint, meta = source_selected(cfg['source_run'], 'last')
    if meta['checkpoint_sha256'] != cfg['source_checkpoint_sha256'] or meta['selected'] != cfg['source_last_tag']:
        raise ValueError('V3.16 LAST changed after continuation was configured')
    del checkpoint
    saved = torch.load(cfg['source_checkpoint'], map_location='cpu', weights_only=True)
    if saved.get('inference_only') or not saved.get('optimizer'):
        raise ValueError('An un-compacted V3.16 LAST with optimizer is required for continuation')
    history = {x['tag']:x for x in saved['history'] if x.get('committed')}
    tag = saved['last_tag']
    if tag not in history or tag in ('initialization', 'starting_parent'):
        raise ValueError('No committed V3.16 LAST validation')
    # Only trained states from THIS V3.16 run. Never import its V3.15 fallback.
    choices = [(tag, saved['model'], history[tag]['metrics'])]
    for key in ('best_weighted', 'best_guarded'):
        name = saved['selections'][key]
        candidate = saved['candidates'].get(name)
        if name in history and candidate and candidate['kind']=='partial':
            choices.append((name, candidate['state'], history[name]['metrics']))
    best_tag, best_model, best_metrics = max(choices, key=lambda x:metric_key(x[2]))
    return saved, dict(tag='source:'+best_tag, model=best_model, metrics=best_metrics,
                      origin=dict(run=cfg['source_run'], version='3.16', tag=best_tag,
                                  checkpoint_sha256=cfg['source_checkpoint_sha256']))


def storage_budget(model, auxiliary, cfg):
    p = sum(v.numel()*v.element_size() for v in model.parameters() if v.requires_grad)
    a = sum(v.numel()*v.element_size() for v in auxiliary.parameters())
    one = int(1.1*(4*p+3*a))+cfg['disk_margin_bytes']
    peak = 2*one+256*1024**2
    if peak > cfg['maximum_new_peak_bytes']:
        raise OSError('Continuation atomic disk peak exceeds configured cap')
    return dict(estimated_checkpoint_bytes=one, new_peak_bytes=peak,
                required_start_free_bytes=peak+cfg['free_reserve_bytes'],
                generated_audio_bytes=0, frame_cache_bytes=0, frozen_prefix_copies=0)


def load_resume(run, cfg):
    run = Path(run)
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    if state.get('schema') != SCHEMA or state.get('identity') != identity(cfg):
        raise ValueError('Continuation checkpoint configuration differs')
    if state['execution_plan_sha256'] != digest(run/'execution_plan.json'):
        raise ValueError('Continuation numerical execution differs')
    return state


def load_selected(run, kind='best'):
    if kind not in KINDS: raise ValueError('Choose best, best_weighted or last')
    run = Path(run).expanduser().resolve()
    cfg, done = read_json(run/'config.json'), read_json(run/'completed.json')
    if done.get('version') != '3.16.1' or done.get('status') != 'complete':
        raise ValueError('Completed V3.16.1 run required')
    if digest(run/'last.pt') != done['checkpoint_sha256']:
        raise ValueError('Completed continuation checkpoint changed')
    state = load_resume(run, cfg)
    if (state['cursor'] != done['committed_updates'] or state['last_tag'] != done['last_tag']
            or state['best_tag'] != done['best_tag']):
        raise ValueError('Completed continuation cursor/selection differs')
    if not state['history'] or state['history'][-1]['cursor'] != state['cursor']:
        raise ValueError('No committed continuation validation')
    last = kind=='last'
    selected = state['last_tag'] if last else state['best_tag']
    origin = dict(run=str(run), version='3.16.1', tag=selected) if last else state['best_origin']
    if origin['version'] not in ('3.16', '3.16.1') or 'starting_parent' in selected:
        raise ValueError('Refusing historical fallback')
    weights = state['model'] if last else state['best_model']
    meta = dict(done, selected=selected, origin=origin, checkpoint_kind=kind,
                checkpoint_path=str(run/'last.pt'), baseline_fallback=False)
    return dict(config=cfg, candidate=dict(kind='partial', state=weights)), meta
