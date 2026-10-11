"""Immutable shared warmup with architecture/data/optimizer/RNG compatibility."""
from copy import deepcopy
from pathlib import Path
import torch
from .common import fingerprint, digest, read_json, atomic_json, atomic_save, to_cpu, schema_for


def signature(cfg):
    return fingerprint({k:v for k,v in cfg.items() if k not in
                        ('contrastive_weight','initialization','warmup_parent')})


def save_warmup(run, cfg, state):
    run = Path(run)
    atomic_save(run/'warmup.pt', to_cpu(state), cfg['free_reserve_bytes']+cfg['disk_margin_bytes'])
    atomic_json(run/'warmup.json', dict(version='3.18.2', epoch=state['epoch'],
        checkpoint_sha256=digest(run/'warmup.pt'), signature=signature(cfg),
        note='Classifier, LoRA initialization, optimizer, BN buffers and RNG are shared.'))


def initial_state(parent, cfg):
    parent = Path(parent)
    meta = read_json(parent/'warmup.json')
    if meta['version'] != '3.18.2' or meta['signature'] != signature(cfg):
        raise ValueError('Shared warmup architecture/data/schedule/code differs')
    if digest(parent/'warmup.pt') != meta['checkpoint_sha256']:
        raise ValueError('Shared warmup checkpoint changed')
    state = torch.load(parent/'warmup.pt', map_location='cpu', weights_only=True)
    if state.get('schema')!=schema_for(cfg) or state.get('identity')!=fingerprint(read_json(parent/'config.json')):
        raise ValueError('Warmup state/configuration identity differs')
    if state['epoch'] != cfg['warm_epochs'] or meta['epoch'] != state['epoch']:
        raise ValueError('Shared warmup must end exactly at the phase boundary')
    if any(e['phase'] != 'head_warmup' for e in state['history']):
        raise ValueError('Warmup includes joint optimization')
    state['identity'] = fingerprint(cfg)
    return state


def fork_configuration(parent, args):
    from .config import verify_inputs, parser
    from .assets import spec
    defaults = parser().parse_args([])
    allowed = {'init_warmup','ce_only','contrastive_weight','download_dir','upload_temp'}
    # validate() resolves these device-profile defaults before this function.
    resolved = {'microbatch','frame_budget','eval_batch'}
    for key in resolved:
        if getattr(args,key) not in (None,spec(args.omni_size)[key]):
            raise ValueError('Shared warmup does not accept a physical batch override: '+key)
    for key, value in vars(args).items():
        if key not in allowed | resolved and value != getattr(defaults,key):
            raise ValueError('Shared warmup keeps the parent configuration; unsupported override: '+key)
    cfg = deepcopy(read_json(Path(parent)/'config.json'))
    verify_inputs(cfg)
    cfg['contrastive_weight'] = 0. if args.ce_only else args.contrastive_weight
    cfg['warmup_parent'] = str(Path(parent).resolve())
    cfg['initialization'] = dict(mode='shared_v3182_classification_warmup',
        old_detector_loaded=False, shared_optimizer_loaded=True, source_run=cfg['warmup_parent'])
    return cfg
