"""Preserve every detector parameter and restore detector Adam moments by group."""
import torch
from w2v_v316_tfcl.model import load_model as construct_source
from w2v_v315.model import optimizer_for
from w2v_v313.state import apply_partial, to_cpu


def load_model(cfg, device=None, training=True):
    # This only constructs the correct frozen prefix and parameter inventory.
    # Every trainable weight is subsequently restored from LAST/selected state.
    source = dict(cfg['continuation_source_config'], device=device or cfg['device'], checkpointing=cfg['checkpointing'])
    model = construct_source(source, device or cfg['device'], training)
    model.head_only = False
    return model


def restore_detector_optimizer(optimizer, saved):
    if not saved or not saved.get('state'):
        raise ValueError('Source LAST has no Adam state; restore the un-compacted training checkpoint before continuing')
    current = optimizer.state_dict()
    old = {g['name']:g for g in saved['param_groups']}
    if len(old) != len(saved['param_groups']):
        raise ValueError('Duplicate optimizer group names')
    expected = {g['name'] for g in current['param_groups'] if g['name'] != 'training_only_tfcl'}
    if set(old) - {'training_only_tfcl'} != expected:
        raise ValueError('Source detector optimizer groups differ')
    restored = 0
    for group in current['param_groups']:
        if group['name'] == 'training_only_tfcl':
            continue  # A new SSL auxiliary branch requires new moments.
        previous = old[group['name']]
        if len(group['params']) != len(previous['params']):
            raise ValueError('Source optimizer parameter inventory differs')
        for target, origin in zip(group['params'], previous['params']):
            if origin in saved['state']:
                current['state'][target] = to_cpu(saved['state'][origin])
                restored += 1
    optimizer.load_state_dict(current)  # Preserves the NEW rates and scheduler bases.
    for group in optimizer.param_groups:
        for p in group['params']:
            for key, value in optimizer.state.get(p, {}).items():
                if key != 'step' and torch.is_tensor(value) and value.shape != p.shape:
                    raise ValueError('Source Adam tensor shape differs')
    return dict(detector_parameters_with_moments=restored, auxiliary_moments='new',
                learning_rate_schedule='restarted_for_additional_epochs')
