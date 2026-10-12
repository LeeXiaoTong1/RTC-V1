"""Reuse state utilities with an incompatible, explicit AUG5 checkpoint schema."""
from w2v_v3184.common import *

RECIPE = 'v3184_aug5_asp_v1'


def schema_for(cfg):
    if cfg.get('recipe') != RECIPE or cfg.get('pooling') not in ('asp','mean'):
        raise ValueError('Not a V3.18.4 AUG5 recipe')
    arch=cfg.get('omni_arch','1b')
    if arch not in ('1b','3b'):raise ValueError('Unsupported encoder')
    return 'rtc_v3184_aug5_omni'+arch+'_'+cfg['pooling']+'_v1'
