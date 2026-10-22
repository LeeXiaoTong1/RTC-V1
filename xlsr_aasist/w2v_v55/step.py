"""One V5.1 AdamW update plus the paper's two auxiliary objectives."""
import copy
import math
import torch
from w2v_v3191.meta import (Transaction, NonfiniteGradient, validate_units, module_norms,
                           _update_norms, _summary)
from w2v_v3191.precision import scaler_for
from w2v_v3191.ce import classification_weights
from .side import side_pass

def train_step(model, optimizer, units, cfg, alpha=0., beta=1., joint=True, audit=False):
    """One view-weighted, class-balanced AdamW update, no adaptation."""
    ce_weights = classification_weights(cfg)
    if float(alpha) != 0. or not joint:
        raise ValueError('V5.5 uses joint classification without virtual adaptation or head warmup')
    beta = float(beta)
    if not math.isfinite(beta) or beta != 1.:
        raise ValueError('All arms require equal A/B classification weights (beta=1)')
    validate_units(units)
    if hasattr(model, 'set_phase') and getattr(model, 'joint', None) is not True:
        model.set_phase(True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    runtime = dict(cfg, current_update=int(getattr(optimizer, '_v3189_joint_update',
                                                 cfg.get('current_update', 1))))
    scaler = scaler_for(optimizer, runtime)
    scaler_initial = copy.deepcopy(scaler.state_dict())
    scale = float(scaler.get_scale())
    transaction = Transaction(model)
    scaler.scale(torch.zeros((), device=transaction.parameters[0][1].device))
    encoded = 0
    try:
        selected = units
        selection = dict(arm='baseline', scoring_encoded_views=0, candidate_scoring=False)
        runtime['_hp_audit'] = audit
        sides = validate_units(selected)
        a = side_pass(model, sides['A'], runtime, scaler, backward=True, backward_factor=1/(1+beta))
        encoded += a['encoded_views']
        b = side_pass(model, sides['B'], runtime, scaler, backward=True, backward_factor=beta/(1+beta))
        encoded += b['encoded_views']
        scaler.unscale_(optimizer)
        norms = module_norms(optimizer, transaction, [p.grad for _, p in transaction.parameters]) if audit else {}
        parameters = [p for _, p in transaction.parameters if p.grad is not None]
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(cfg.get('max_grad_norm', 1.)),
                                                  error_if_nonfinite=False))
        if not math.isfinite(norm):
            raise NonfiniteGradient('Nonfinite paired-classification gradient')
    except (NonfiniteGradient, FloatingPointError) as exc:
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled():
            scaler.load_state_dict(scaler_initial)
            raise
        scaler.update(new_scale=scale*float(scaler.get_backoff_factor()))
        return dict(loss=None, classification_loss=None, encoded_views=encoded, views=len(units),
                    classification_ce_weights=ce_weights,
                    classification_source_views=len(units), unique_original_sources=len(units),
                    auxiliary_reference_views=0, cells={}, raw_ce_sum=None, max_example_ce=None,
                    predicted_fake_by_true_class={'fake': [0, 0], 'real': [0, 0]}, grad_norm=None,
                    phase='joint', lrs={g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)},
                    amp=dict(enabled=True, previous_scale=scale, scale=scaler.get_scale(),
                             skipped_update=True, reason=str(exc)),
                    meta=dict(alpha=0., beta=beta, objective='paired_CE_no_virtual_update'))
    except BaseException:
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.update(new_scale=scale)
        scaler.load_state_dict(scaler_initial)
        raise
    # Single commit point. Earlier failures cannot modify Adam moments.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    result = _summary(selected, a, b, beta)
    result.update(views=a['views']+b['views'], classification_source_views=len(units),
                  classification_ce_weights=ce_weights,
                  paired_classification_views=a['paired_classification_views']+b['paired_classification_views'],
                  auxiliary_reference_views=a['auxiliary_reference_views']+b['auxiliary_reference_views'],
                  encoded_views=encoded, training_encoded_views=a['encoded_views']+b['encoded_views'],
                  grad_norm=norm, selection=selection,
                  amp=dict(enabled=scaler.is_enabled(), previous_scale=scale, scale=scaler.get_scale(),
                           skipped_update=scaler.get_scale() < scale),
                  lrs={g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)},
                  meta=dict(alpha=0., beta=beta, A_loss=a['loss'], B_loss=b['loss'],
                            objective='(A_paired_CE + A_parent_PCL + beta*B_paired_CE)/(1+beta)',
                            virtual_update_norm=0., module_outer_gradient_norms=norms,
                            module_outer_update_norms=_update_norms(optimizer, transaction) if audit else {}))
    hp = dict(groups=a['hard_positive']['groups'] + b['hard_positive']['groups'],
        guidance_weighted=(a['hard_positive']['guidance_weighted'] + b['hard_positive']['guidance_weighted']) * .5,
        triplet_weighted=(a['hard_positive']['triplet_weighted'] + b['hard_positive']['triplet_weighted']) * .5,
        extra_encoded_views=a['extra_classification_views'] + b['extra_classification_views'],
        extra_scoring_passes=0, mode=cfg.get('hp_mode', 'hard_positive'), sides={})
    for name, side in (('A', a), ('B', b)):
        hp['sides'][name] = side['hard_positive']
    result['hard_positive'] = hp
    result['pcl_losses'] = {kind: dict(item, weighted_loss=item['weighted_loss']*.5)
                           for kind, item in a['pcl_losses'].items()}
    result['pcl_variance'] = dict(a['pcl_variance'], weighted_loss=a['pcl_variance']['weighted_loss']*.5)
    result['en_real_confidence'] = dict(a['en_real_confidence'],
        weighted_loss=(a['en_real_confidence']['weighted_loss']+b['en_real_confidence']['weighted_loss'])*.5)
    result['hard_positive_gradient_audit'] = {name: side['pcl_gradient_audit']
                                            for name, side in (('A', a), ('B', b))}
    result['meta']['objective'] = 'V5.1 CE/PCL/confidence + (A_Guidance+Triplet + B_Guidance+Triplet)/2'
    return result
