"""Reuse the V3.15 inference architecture and load the chosen parent exactly."""
from w2v_v315.model import load_model as original_model, optimizer_for
from w2v_v315.state import load_selected as load_parent, apply_candidate
from .config import verify_parent


def load_model(cfg,device=None,training=True):
    verify_parent(cfg)
    parent,meta=load_parent(cfg['parent_run'],cfg['parent_selector'])
    if (meta['selected']!=cfg['parent_selected_tag'] or
            meta['checkpoint_sha256']!=cfg['parent_checkpoint_sha256']):
        raise ValueError('V3.15 parent selection changed')
    model=original_model(cfg,device,training)
    apply_candidate(model,parent['candidate'])
    del parent
    return model
