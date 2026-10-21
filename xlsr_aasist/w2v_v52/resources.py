"""Bound the complete three-view source even when its PCL is disabled."""
from w2v_v5.resources import audit_training as parent_audit
from .contrastive import configuration


def audit_training(rows, dev, cfg):
    bound = dict(cfg)
    if configuration(cfg)[0]:
        bound.update(consistency='dual', pcl_platform_weight=max(.001, cfg.get('pcl_platform_weight', 0.)),
                     platform_sources_per_group=max(1, cfg.get('platform_sources_per_group', 0)))
    result = parent_audit(rows, dev, bound)
    result['v52_notes'] = 'Official Online is counted for contrast even when no valid phoneme pairs exist; reused by PCL.'
    return result
