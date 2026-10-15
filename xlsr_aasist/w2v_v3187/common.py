"""Version-isolated checkpoint and resume contract."""
from w2v_v3184.common import *

RECIPE = 'v3187_fresh_window_asp_dynamic_dual_pcl_v1'


def schema_for(cfg):
    if cfg.get('version') != '3.18.7' or cfg.get('recipe') != RECIPE:
        raise ValueError('Expected V3.18.7 configuration; older runs cannot be resumed')
    if cfg.get('pooling') != 'asp' or not cfg.get('local_evidence'):
        raise ValueError('V3.18.7 requires overlapping windows and statistics pooling')
    for name in ('window', 'hop', 'window_batch'):
        if type(cfg.get(name)) is not int or cfg[name] < 1:
            raise ValueError('Invalid window geometry: ' + name)
    if cfg['window'] < 12 or cfg['hop'] > cfg['window']:
        raise ValueError('Windows must cover all frames')
    if cfg.get('consistency') not in ('dual', 'none'):
        raise ValueError('Unknown consistency objective')
    arch = cfg.get('omni_arch', '3b')
    if arch not in ('1b', '3b'):
        raise ValueError('Unknown Omni architecture')
    return 'rtc_v3187_omni' + arch + '_window_asp_dynamic_v1'
