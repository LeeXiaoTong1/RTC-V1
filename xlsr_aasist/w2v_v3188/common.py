"""Version-isolated checkpoint and resume contract."""
from w2v_v3184.common import *

RECIPE = 'v3188_single_noisy_memory_bounded_v1'


def schema_for(cfg):
    if cfg.get('version') != '3.18.8' or cfg.get('recipe') != RECIPE:
        raise ValueError('Expected V3.18.8 configuration; older runs cannot be resumed')
    if cfg.get('pooling') != 'asp' or not cfg.get('local_evidence'):
        raise ValueError('V3.18.8 requires overlapping windows and statistics pooling')
    for name in ('window', 'hop', 'window_batch'):
        if type(cfg.get(name)) is not int or cfg[name] < 1:
            raise ValueError('Invalid window geometry: ' + name)
    if cfg['window'] < 12 or cfg['hop'] > cfg['window']:
        raise ValueError('Windows must cover all frames')
    if cfg.get('consistency') not in ('dual', 'none'):
        raise ValueError('Unknown consistency objective')
    if cfg.get('noisy_views_per_source',1)!=1:
        raise ValueError('V3.18.8 requires exactly one Noisy view per Offline source visit')
    arch = cfg.get('omni_arch', '3b')
    if arch not in ('1b', '3b'):
        raise ValueError('Unknown Omni architecture')
    return 'rtc_v3188_omni' + arch + '_gn_temporal_single_noisy_v1'


def partial_state(model):
    """Save every detection tensor plus the Train-only PCL scale lifecycle."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(('head.', 'fusion.', 'pcl_scale.')) or k.endswith(('.lora_a', '.lora_b'))}


def apply_partial(model, values):
    target = model.state_dict()
    # Validate metadata against live tensors; do not copy all 60-layer LoRA
    # weights back to CPU merely to check shapes before loading a checkpoint.
    expected = {k:v for k,v in target.items()
                if k.startswith(('head.', 'fusion.', 'pcl_scale.')) or k.endswith(('.lora_a', '.lora_b'))}
    if set(expected) != set(values):
        raise ValueError('V3.18.8 partial keys differ; full detector and PCL scale required')
    # Validate the entire snapshot before changing any live tensor.
    for name, value in values.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid V3.18.8 partial tensor: ' + name)
    with torch.no_grad():
        for name, value in values.items():
            target[name].copy_(value)
    if hasattr(model,'pcl_scale'):
        model.pcl_scale.sync_state()
