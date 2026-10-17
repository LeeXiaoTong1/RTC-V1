"""Version-isolated checkpoint and resume contract."""
from w2v_v3184.common import *

RECIPE = 'v31810_source_isolated_first_order_metadg_v1'


def schema_for(cfg):
    if cfg.get('version') != '3.18.10' or cfg.get('recipe') != RECIPE:
        raise ValueError('Expected V3.18.10 configuration; older runs cannot be resumed')
    if cfg.get('pooling') != 'asp' or not cfg.get('local_evidence'):
        raise ValueError('V3.18.10 requires overlapping windows and statistics pooling')
    for name in ('window', 'hop', 'window_batch'):
        if type(cfg.get(name)) is not int or cfg[name] < 1:
            raise ValueError('Invalid window geometry: ' + name)
    if cfg['window'] < 12 or cfg['hop'] > cfg['window']:
        raise ValueError('Windows must cover all frames')
    if cfg.get('consistency') != 'none':
        raise ValueError('Unknown consistency objective')
    if any(cfg.get(name,0.)!=0. for name in ('pcl_noise_weight','pcl_platform_weight','pcl_variance_weight','en_real_confidence_weight')):
        raise ValueError('V3.18.10 classification-only recipe rejects auxiliary losses')
    if cfg.get('noisy_views_per_source',1)!=1:
        raise ValueError('V3.18.10 requires exactly one Noisy view per Offline source visit')
    if cfg.get('rtc_activity_policy','mixed') not in ('mixed','off'):
        raise ValueError('Unknown RTC activity policy')
    arch = cfg.get('omni_arch', '3b')
    if arch not in ('1b', '3b'):
        raise ValueError('Unknown Omni architecture')
    return 'rtc_v31810_omni' + arch + '_gn_temporal_metadg_v1'


def partial_state(model):
    """Save detection tensors only; the official frozen backbone is immutable."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(('head.', 'fusion.')) or k.endswith(('.lora_a', '.lora_b'))}


def apply_partial(model, values):
    target = model.state_dict()
    # Validate metadata against live tensors; do not copy all 60-layer LoRA
    # weights back to CPU merely to check shapes before loading a checkpoint.
    expected = {k:v for k,v in target.items()
                if k.startswith(('head.', 'fusion.')) or k.endswith(('.lora_a', '.lora_b'))}
    if set(expected) != set(values):
        raise ValueError('V3.18.10 partial keys differ; complete fresh detector required')
    # Validate the entire snapshot before changing any live tensor.
    for name, value in values.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid V3.18.10 partial tensor: ' + name)
    with torch.no_grad():
        for name, value in values.items():
            target[name].copy_(value)
