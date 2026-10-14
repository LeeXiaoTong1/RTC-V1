"""Version-isolated checkpoint contract; prior releases remain immutable."""
from w2v_v3184.common import *

RECIPE = 'v3186_window_asp_dual_pcl_v2'


def schema_for(cfg):
    if cfg.get('version') != '3.18.6' or cfg.get('recipe') != RECIPE:
        raise ValueError('Expected restored-window V3.18.6 configuration; start a new run from the ASP parent, not --resume of a whole-utterance run')
    if cfg.get('pooling') != 'asp' or not cfg.get('local_evidence'):
        raise ValueError('V3.18.6 requires overlapping windows and attention statistics pooling')
    for name in ('window','hop','window_batch'):
        if type(cfg.get(name)) is not int or cfg[name] < 1:
            raise ValueError('Invalid inherited window geometry: '+name)
    if cfg['window'] < 12 or cfg['hop'] > cfg['window']:
        raise ValueError('Window geometry must cover all frames with valid AASIST windows')
    if cfg.get('consistency') not in ('dual', 'none'):
        raise ValueError('Unknown V3.18.6 objective')
    arch = cfg.get('omni_arch', '3b')
    if arch not in ('1b', '3b'):
        raise ValueError('Unsupported Omni architecture')
    return 'rtc_v3186_omni'+arch+'_window_asp_dual_pcl_v2'
