"""Version-isolated checkpoint contract; prior releases remain immutable."""
from w2v_v3184.common import *

RECIPE = 'v3186_whole_dual_pcl_v1'


def schema_for(cfg):
    if cfg.get('version') != '3.18.6' or cfg.get('recipe') != RECIPE:
        raise ValueError('Expected V3.18.6 configuration')
    if cfg.get('pooling') != 'whole' or cfg.get('local_evidence'):
        raise ValueError('V3.18.6 requires whole-utterance AASIST without external window pooling')
    if cfg.get('consistency') not in ('dual', 'none'):
        raise ValueError('Unknown V3.18.6 objective')
    arch = cfg.get('omni_arch', '3b')
    if arch not in ('1b', '3b'):
        raise ValueError('Unsupported Omni architecture')
    return 'rtc_v3186_omni'+arch+'_whole_dual_pcl_v1'
