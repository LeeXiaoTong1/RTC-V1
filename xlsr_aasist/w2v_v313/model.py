"""Continue trained V3.12 weights with the identical inference architecture."""
import torch

from w2v_v312.model import (FeatureClassifier, LanguageAdversary, CONDITIONS,
                           optimizer_for, load_model as original_model)
from w2v_v312.state import SCHEMA as V312_SCHEMA, apply_partial
from w2v_v39.common import verify_files


def load_model(cfg, device=None, training=True):
    verify_files({cfg['starting_checkpoint']:cfg['starting_checkpoint_sha256']})
    state = torch.load(cfg['starting_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    if (state.get('schema') != V312_SCHEMA or state.get('identity') != cfg['starting_config_identity']
            or state.get('cursor') != cfg['starting_cursor'] or not state.get('model')):
        raise ValueError('Starting checkpoint is not the pinned trained V3.12 LAST')
    model = original_model(cfg, device, training)
    apply_partial(model, state['model'])
    del state
    return model
