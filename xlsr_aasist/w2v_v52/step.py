"""V5.2 atomic Online-heavy CE, reduced noisy PCL and paired contrast."""
import copy
import math
import torch
from w2v_v3191.meta import (Transaction, NonfiniteGradient, validate_units, module_norms, _update_norms, _summary)
from w2v_v3191.precision import scaler_for
from .ce import classification_weights
from .side import side_pass
from .contrastive import bank_for, next_bank, device_bank


def train_step(model, optimizer, units, cfg, alpha=0., beta=1., joint=True, audit=False):
    """One view-weighted, class-balanced AdamW update, no adaptation."""
    ce_weights = classification_weights(cfg)
    if float(alpha) != 0. or not joint:
        raise ValueError('V5.2 uses joint classification without virtual adaptation or head warmup')
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
    bank = bank_for(model, runtime)
    runtime['_v52_device_bank'] = device_bank(bank, transaction.parameters[0][1].device)
    scaler.scale(torch.zeros((), device=transaction.parameters[0][1].device))
    encoded = 0
    try:
        selected = units
        selection = dict(arm='baseline', scoring_encoded_views=0, candidate_scoring=False)
        sides = validate_units(selected)
        a = side_pass(model, sides['A'], runtime, scaler, backward=True, backward_factor=1/(1+beta), audit=audit)
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
        proposed_bank = next_bank(bank, a.pop('_contrastive_pending'), runtime)
        if b.pop('_contrastive_pending'):
            raise ValueError('Side B must not enqueue contrastive references')
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
    model._v52_contrastive_bank = proposed_bank
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
                            objective='(A_CE60_20_20 + A_reduced_PCL + A_contrast + A_confidence + B_CE60_20_20 + B_confidence)/2',
                            virtual_update_norm=0., module_outer_gradient_norms=norms,
                            module_outer_update_norms=_update_norms(optimizer, transaction) if audit else {}))
    # Report actual contributions to the committed total, including A's 1/2.
    result['pcl_losses'] = {kind: dict(item, weighted_loss=item['weighted_loss']/(1+beta))
                            for kind, item in a['pcl_losses'].items()}
    result['pcl_variance'] = dict(a['pcl_variance'], weighted_loss=a['pcl_variance']['weighted_loss']/(1+beta))
    result['en_real_confidence'] = dict(a['en_real_confidence'],
        weighted_loss=(a['en_real_confidence']['weighted_loss']+beta*b['en_real_confidence']['weighted_loss'])/(1+beta))
    result['contrastive'] = dict(a['contrastive'], weighted_loss=a['contrastive']['weighted_loss']/(1+beta),
        outer_side_weight=1/(1+beta), bank_entries_after=len(proposed_bank['entries']))
    result['auxiliary_gradient_audit'] = a['pcl_gradient_audit']
    return result
