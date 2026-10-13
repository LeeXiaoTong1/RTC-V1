from w2v_v3184.test_support import tiny_cfg as prior_cfg,scene_fixture
from w2v_v318.test_core import TinyFrontend,TinyEncoder,TinyLayout
from .model import Detector
from .common import RECIPE

def tiny_cfg():
    cfg=prior_cfg();cfg.update(version='3.18.5',recipe=RECIPE,pooling='asp',view_factors=[5,5,1,3],
        refresh_base_variants=False,base_original_fraction=.2,pool_lr=.001,classifier_lr=.001)
    return cfg

def tiny_model(cfg):return Detector(TinyFrontend(),TinyEncoder(),cfg,TinyLayout)
