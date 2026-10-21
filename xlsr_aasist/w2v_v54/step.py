"""Paired CE, retained platform PCL, and A-side repair with global coefficients."""
import copy
import math
import torch
from w2v_v3191.ce import classification_weights
from .side import side_pass
from w2v_v3191.meta import (Transaction, NonfiniteGradient, validate_units, module_norms,
                            _update_norms, _summary, rng_state)
from w2v_v3191.precision import scaler_for


class CommittedStepState(Transaction):
    """Forward mutates buffers/RNG, but never parameters before AdamW commits.

    Keep parameter copies only for occasional exact update diagnostics. This
    state must not be used by historical virtual/meta-update implementations.
    """
    def __init__(self, model, *, audit=False):
        if getattr(model, '_v31810_virtual_active', False):
            raise ValueError('V5.4 cannot run on temporary adapted parameters')
        self.model = model
        self.parameters = [(name, p) for name,p in model.named_parameters() if p.requires_grad]
        if not self.parameters or any(p.dtype != torch.float32 for _,p in self.parameters):
            raise ValueError('V5.4 requires FP32 master trainable parameters')
        self.original = [p.detach().clone() for _,p in self.parameters] if audit else []
        self.buffers = [(b, b.detach().clone()) for b in model.buffers()]
        self.modes = [(m, m.training) for m in model.modules()]
        self.rng = rng_state()

    @property
    def snapshot_bytes(self):
        return sum(p.numel()*p.element_size() for p in self.original)


def _train_step(model, optimizer, units, cfg, alpha=0., beta=1., joint=True, audit=False):
    ce_weights = classification_weights(cfg)
    if float(alpha) != 0. or not joint or float(beta) != 1.:
        raise ValueError('V5.4 uses V5.1 joint CE/PCL, equal A/B weight and no virtual update')
    if cfg.get('robust_arm') != 'baseline' or cfg.get('contrastive_weight', 0.) != 0.:
        raise ValueError('V5.4 has no candidate mining or contrastive objective')
    sides = validate_units(units)
    model.set_training_stage(bool(getattr(optimizer, '_v3189_joint', True)))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    runtime = dict(cfg, current_update=int(getattr(optimizer, '_v3189_joint_update',
                                                 cfg.get('current_update', 1))))
    runtime['_repair_warm_steps'] = int(getattr(optimizer, '_v54_warm_steps', cfg['repair_warmup_max']))
    scaler = scaler_for(optimizer, runtime)
    initial_scaler = copy.deepcopy(scaler.state_dict())
    scale = float(scaler.get_scale())
    transaction = CommittedStepState(model, audit=audit)
    scaler.scale(torch.zeros((), device=transaction.parameters[0][1].device))
    encoded = 0
    try:
        a = side_pass(model, sides['A'], runtime, scaler, backward=True, backward_factor=.5)
        encoded += a['encoded_views']
        b = side_pass(model, sides['B'], runtime, scaler, backward=True, backward_factor=.5)
        encoded += b['encoded_views']
        scaler.unscale_(optimizer)
        norms = module_norms(optimizer, transaction, [p.grad for _,p in transaction.parameters]) if audit else {}
        parameters = [p for _,p in transaction.parameters if p.grad is not None]
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(cfg.get('max_grad_norm', 1.)),
                                                  error_if_nonfinite=False))
        if not math.isfinite(norm):
            raise NonfiniteGradient('Nonfinite paired-classification gradient')
    except (NonfiniteGradient, FloatingPointError) as exc:
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled():
            scaler.load_state_dict(initial_scaler)
            raise
        scaler.update(new_scale=scale*float(scaler.get_backoff_factor()))
        return dict(loss=None, classification_loss=None, encoded_views=encoded, views=len(units),
            classification_ce_weights=ce_weights, cells={},
            amp=dict(enabled=True, previous_scale=scale, scale=scaler.get_scale(),
                     skipped_update=True, reason=str(exc)))
    except BaseException:
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.update(new_scale=scale)
        scaler.load_state_dict(initial_scaler)
        raise
    # The first parameter/Adam-state mutation. A failure inside this commit
    # requires the last disk checkpoint, as in the original V5.1 implementation.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    result = _summary(units, a, b, 1.)
    from .telemetry import collect
    collect(a, b, runtime, result)
    result['repair_stage'] = model.repair_stage
    result.update(views=a['views']+b['views'], classification_source_views=len(units),
        classification_ce_weights=ce_weights,
        paired_classification_views=a['paired_classification_views']+b['paired_classification_views'],
        auxiliary_reference_views=a['auxiliary_reference_views']+b['auxiliary_reference_views'],
        encoded_views=encoded, training_encoded_views=encoded, grad_norm=norm,
        selection=dict(arm='baseline', scoring_encoded_views=0, candidate_scoring=False),
        amp=dict(enabled=scaler.is_enabled(), previous_scale=scale, scale=scaler.get_scale(),
                 skipped_update=scaler.get_scale() < scale),
        lrs={g.get('name', str(i)): g['lr'] for i,g in enumerate(optimizer.param_groups)},
        execution=dict(parameter_snapshot_bytes=transaction.snapshot_bytes,
                       contrastive_reference_views=0, contrastive_bank_entries=0),
        meta=dict(alpha=0., beta=1., A_loss=a['loss'], B_loss=b['loss'],
                  objective='(A_CE + A_platform_PCL + A_confidence + B_CE + B_confidence)/2 + lambdaF*LF_A + lambdaD*LD_A; baseline keeps original noise PCL',
                  virtual_update_norm=0., module_outer_gradient_norms=norms,
                  module_outer_update_norms=_update_norms(optimizer, transaction) if audit else {}))
    # Report effective contributions, including the A side's half weight.
    result['pcl_losses'] = {kind: dict(item, weighted_loss=item['weighted_loss']*.5)
                            for kind,item in a['pcl_losses'].items()}
    result['pcl_variance'] = dict(a['pcl_variance'], weighted_loss=a['pcl_variance']['weighted_loss']*.5)
    result['en_real_confidence'] = dict(a['en_real_confidence'],
        weighted_loss=(a['en_real_confidence']['weighted_loss']+b['en_real_confidence']['weighted_loss'])*.5)
    return result


def train_step(model, optimizer, units, cfg, alpha=0., beta=1., joint=True, audit=False):
    fusion = model.fusion
    fusion.reset_observations(enabled=audit)
    before = [p.detach().clone() for p in fusion.conditioner.parameters()] if audit else []
    repair_before = {k: p.detach().clone() for k, p in model.repair.named_parameters()} if audit else {}
    try:
        result = _train_step(model, optimizer, units, cfg, alpha, beta, joint, audit)
        if audit:
            with torch.no_grad():
                change = sum((p.detach()-old).float().square().sum()
                             for p,old in zip(fusion.conditioner.parameters(), before))
                result['adaptive_fusion'] = dict(fusion.observations(),
                    conditioner_update_norm=float(change.sqrt()), mode=fusion.mode,
                    committed=not result['amp']['skipped_update'], extra_encoder_passes=0)
                result['repair_update'] = {part: float(sum((p.detach()-repair_before[n]).square().sum()
                    for n, p in model.repair.named_parameters() if n.startswith('gate.') == (part == 'gate')).sqrt())
                    for part in ('processor', 'gate')}
        return result
    finally:
        # Never retain diagnostic tensors across updates or AMP retries.
        fusion.reset_observations()
