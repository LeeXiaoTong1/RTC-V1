"""Reuse state utilities with an incompatible, explicit targeted augmentation checkpoint schema."""
from w2v_v3184.common import *

RECIPE = 'v3185_asp_v1'


def schema_for(cfg):
    if cfg.get('recipe') != RECIPE or cfg.get('pooling') not in ('asp','mean'):
        raise ValueError('Not a V3.18.5 targeted augmentation recipe')
    arch=cfg.get('omni_arch','1b')
    if arch not in ('1b','3b'):raise ValueError('Unsupported encoder')
    return 'rtc_v3185_omni'+arch+'_'+cfg['pooling']+'_v1'
