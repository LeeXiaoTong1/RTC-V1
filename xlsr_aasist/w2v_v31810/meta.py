"""Transactional first-order cross-condition updates on a single detector.

Only trainable FP32 tensors are copied: no 3B-base copy and no second-order
graph. Inner SGD uses clipped A gradients; the outer update uses RAW A
gradients plus B gradients evaluated at the temporary parameters.
"""
from collections import Counter, defaultdict
import copy
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from .precision import scaler_for


CELLS = tuple((stream, language, label) for stream in ('online', 'noisy')
              for language in ('en', 'zh') for label in (0, 1))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


class Transaction:
    """Restore transient changes without copying frozen backbone or Adam state.

The optimizer must NOT be called inside the temporary phase. Consequently an
exception there cannot alter its moments. A successful final AdamW step is
the commit point; fatal device failures inside that step require disk resume.
"""
    def __init__(self, model):
        self.parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError('No trainable detector parameters')
        if any(p.dtype != torch.float32 for _, p in self.parameters):
            raise ValueError('Meta updates require FP32 master trainable parameters')
        self.original = [p.detach().clone() for _, p in self.parameters]
        self.buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
        self.modes = [(module, module.training) for module in model.modules()]
        self.rng = rng_state()

    def restore_parameters(self):
        with torch.no_grad():
            for (_, p), old in zip(self.parameters, self.original):
                p.copy_(old)

    def restore_buffers(self):
        with torch.no_grad():
            for buffer, old in self.buffers:
                buffer.copy_(old)

    def restore(self, *, parameters=True, rng=True, modes=True):
        if parameters:
            self.restore_parameters()
        self.restore_buffers()
        if rng:
            restore_rng(self.rng)
        if modes:
            for module, training in self.modes:
                module.training = training


class NonfiniteGradient(FloatingPointError):
    def __init__(self, message, *, encoded_views=0):
        super().__init__(message)
        self.encoded_views = encoded_views


def cell(row):
    role = row['role']
    if role not in ('online', 'noisy'):
        raise ValueError('Only Online and simulated Noisy classification rows are allowed')
    return role, row['language'], row['label']


def validate_units(units):
    if not units or any(len(unit) != 1 for unit in units):
        raise ValueError('Each meta unit must contain exactly one classification audio')
    identities = []
    for unit in units:
        row = unit[0]
        if row.get('split', 'train') != 'train':
            raise ValueError('Only Train audio may enter meta updates')
        if row.get('meta_side') not in ('A', 'B') or cell(row) not in CELLS:
            raise ValueError('Invalid meta side or classification cell')
        identity = row.get('canonical_source', row.get('group_id', row.get('source_id')))
        if identity is None:
            raise ValueError('Canonical source identity is required')
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError('Meta task repeats an original source across classification views')
    sides = {side: [unit for unit in units if unit[0]['meta_side'] == side] for side in ('A', 'B')}
    for side, side_units in sides.items():
        missing = set(CELLS) - {cell(unit[0]) for unit in side_units}
        if missing:
            raise ValueError(f'Meta side {side} is missing classification cells: {sorted(missing)}')
    return sides


def side_pass(model, units, cfg, scaler, *, backward):
    """Accumulate the same 8-cell objective regardless of physical packing."""
    from w2v_v3189.data import microbatches
    counts = Counter(cell(unit[0]) for unit in units)
    if set(counts) != set(CELLS):
        raise ValueError('Each loss must contain all eight balanced classification cells')
    metadata, records = [], []
    encoded = 0
    for micro in microbatches(units, int(cfg.get('microbatch', 12)),
                              int(cfg.get('frame_budget', 9000)), weights={'noise': 0., 'platform': 0.}):
        rows = [unit[0] for unit in micro]
        with torch.set_grad_enabled(backward):
            try:
                logits = model([row['wave'] for row in rows]).float()
            except FloatingPointError as exc:
                # The production detector performs its own finite checks.
                raise NonfiniteGradient(str(exc), encoded_views=encoded + len(rows)) from exc
            if logits.shape != (len(rows), 2) or not bool(torch.isfinite(logits).all()):
                raise NonfiniteGradient('Invalid/nonfinite binary classification logits', encoded_views=encoded + len(rows))
            targets = torch.tensor([row['label'] for row in rows], device=logits.device)
            raw = F.cross_entropy(logits, targets, reduction='none',
                                  label_smoothing=float(cfg.get('label_smoothing', .02)))
            weights = raw.new_tensor([1. / (8 * counts[cell(row)]) for row in rows])
            weighted = raw * weights
            loss = weighted.sum()
            if not bool(torch.isfinite(loss)):
                raise NonfiniteGradient('Nonfinite classification loss', encoded_views=encoded + len(rows))
            if backward:
                scaler.scale(loss).backward()
        predictions = logits.detach().argmax(1)
        records.append(torch.stack((weighted.detach(), raw.detach(),
                                   (predictions == targets).float(), (predictions == 0).float()), 1))
        metadata.extend(cell(row) for row in rows)
        encoded += len(rows)
    values = torch.cat(records).cpu().tolist()
    cells = defaultdict(lambda: dict(objective=0., raw_ce_sum=0., views=0, correct=0, predicted_fake=0))
    for key, (weighted, raw, correct, fake) in zip(metadata, values):
        target = cells['/'.join(map(str, key))]
        for name, value in dict(objective=weighted, raw_ce_sum=raw, views=1,
                                correct=int(correct), predicted_fake=int(fake)).items():
            target[name] += value
    return dict(loss=sum(value[0] for value in values), raw_ce_sum=sum(value[1] for value in values),
                max_example_ce=max(value[1] for value in values), cells=dict(cells), views=encoded)


def raw_gradients(transaction, scale):
    """Detach and unscale in place, then release parameter .grad ownership."""
    gradients = []
    with torch.no_grad():
        for _, parameter in transaction.parameters:
            gradient = parameter.grad
            if gradient is not None:
                gradient = gradient.detach()
                gradient.div_(scale)
            gradients.append(gradient)
            parameter.grad = None
    norm = gradient_norm(gradients)
    if not math.isfinite(norm):
        raise NonfiniteGradient('Nonfinite accumulated side gradient')
    return gradients, norm


def gradient_norm(gradients):
    terms = [torch.linalg.vector_norm(g.detach().float()) for g in gradients if g is not None]
    if not terms:
        raise ValueError('Classification objective produced no trainable gradients')
    # FP64 only for the small list of tensor norms, not full gradient tensors.
    return float(torch.linalg.vector_norm(torch.stack(terms).double()))


def virtual_step(transaction, gradients, alpha, maximum=1., *, norm=None):
    norm = gradient_norm(gradients) if norm is None else norm
    if not math.isfinite(norm):
        raise NonfiniteGradient('Nonfinite inner gradient norm')
    factor = min(1., maximum / (norm + 1e-6))
    with torch.no_grad():
        for (_, parameter), original, gradient in zip(transaction.parameters, transaction.original, gradients):
            parameter.copy_(original)
            if gradient is not None and alpha:
                parameter.add_(gradient, alpha=-alpha * factor)
    return norm, factor


def module_norms(optimizer, transaction, gradients):
    lookup = {id(p): g for (_, p), g in zip(transaction.parameters, gradients)}
    result, selected_by_name = {}, defaultdict(list)
    for i, group in enumerate(optimizer.param_groups):
        selected_by_name[group.get('name', str(i))].extend(
            lookup[id(p)] for p in group['params'] if id(p) in lookup and lookup[id(p)] is not None)
    for name, selected in selected_by_name.items():
        if selected:
            value = gradient_norm(selected)
            result[name] = value if math.isfinite(value) else None
    return result


def _update_norms(optimizer, transaction):
    originals = {id(p): value for (_, p), value in zip(transaction.parameters, transaction.original)}
    result, terms_by_name = {}, defaultdict(list)
    with torch.no_grad():
        for i, group in enumerate(optimizer.param_groups):
            terms_by_name[group.get('name', str(i))].extend(
                torch.linalg.vector_norm((p - originals[id(p)]).float())
                for p in group['params'] if id(p) in originals)
        for name, terms in terms_by_name.items():
            if terms:
                result[name] = float(torch.linalg.vector_norm(torch.stack(terms).double()))
    return result


def _summary(units, a, b, beta, *, warm=False):
    result = dict(cells={}, predicted_fake_by_true_class={'fake': [0, 0], 'real': [0, 0]})
    for side, factor in ((a, 1. if warm else 1. / (1. + beta)),
                         (b, 0. if warm else beta / (1. + beta))):
        if side is None:
            continue
        for key, values in side['cells'].items():
            target = result['cells'].setdefault(key, dict(objective=0., raw_ce_sum=0., views=0, correct=0))
            for name in target:
                target[name] += values[name] * (factor if name == 'objective' else 1)
            pred = result['predicted_fake_by_true_class']['fake' if key.endswith('/0') else 'real']
            pred[0] += values['predicted_fake']
            pred[1] += values['views']
    result.update(loss=a['loss'] if warm else (a['loss'] + beta * b['loss']) / (1 + beta),
                  raw_ce_sum=a['raw_ce_sum'] + (b['raw_ce_sum'] if b else 0),
                  max_example_ce=max(a['max_example_ce'], b['max_example_ce'] if b else 0),
                  views=len(units), unique_original_sources=len(units),
                  auxiliary_reference_views=0, phase='head_warmup' if warm else 'joint')
    result['classification_loss'] = result['loss']
    return result


def train_step(model, optimizer, units, cfg, alpha, beta, joint=True, audit=False):
    """One complete optimizer update; exceptions during virtual work roll back.

    E0 uses alpha=0 with the SAME beta side weighting and role schedule. It is
    not generally equivalent to an equally weighted 48-example batch. Warmup
    does use one balanced whole-task CE; beta=0 joint tasks still forward B
    for exposure accounting but deliberately do not backpropagate its loss.
    """
    alpha, beta = float(alpha), float(beta)
    if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(beta) or beta < 0:
        raise ValueError('Nonnegative finite alpha and beta required')
    sides = validate_units(units)
    if hasattr(model, 'set_phase') and getattr(model, 'joint', None) != joint:
        model.set_phase(joint)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = scaler_for(optimizer, cfg)
    scaler_initial = copy.deepcopy(scaler.state_dict())
    scale = float(scaler.get_scale())
    transaction = Transaction(model)
    # Initialize the lazy scaler even if a first forward fails before loss.
    scaler.scale(torch.zeros((), device=transaction.parameters[0][1].device))
    a = b = None
    diagnostics = {}
    encoded = 0
    try:
        if not joint:
            a = side_pass(model, units, cfg, scaler, backward=True)
            encoded = len(units)
            gradients, norm_a = raw_gradients(transaction, scale)
            gradient_audit = module_norms(optimizer, transaction, gradients) if audit else {}
            # GradScaler unscales only once, after all logical accumulation.
            for (_, p), g in zip(transaction.parameters, gradients):
                if g is not None:
                    p.grad = g.mul_(scale)
            diagnostics.update(A_loss=None, B_loss=None, inner_grad_norm=None, inner_clip_factor=None)
        else:
            a = side_pass(model, sides['A'], cfg, scaler, backward=True)
            encoded += len(sides['A'])
            gradients, norm_a = raw_gradients(transaction, scale)
            gradient_audit = module_norms(optimizer, transaction, gradients) if audit else {}
            # Sparse matched-state diagnostic. Never regenerate audio or spend
            # an extra base encoding on normal, unaudited training updates.
            a_after = b_before = None
            b_rng = rng_state()
            post_a_buffers = [(buffer, buffer.detach().clone()) for buffer, _ in transaction.buffers]
            if audit:
                restore_rng(transaction.rng)
                transaction.restore_buffers()
                virtual_step(transaction, gradients, alpha, float(cfg.get('max_grad_norm', 1.)), norm=norm_a)
                a_after = side_pass(model, sides['A'], cfg, scaler, backward=False)['loss']
                encoded += len(sides['A'])
                transaction.restore_parameters()
                restore_rng(b_rng)
                with torch.no_grad():
                    for buffer, old in post_a_buffers:
                        buffer.copy_(old)
                b_before = side_pass(model, sides['B'], cfg, scaler, backward=False)['loss']
                encoded += len(sides['B'])
                restore_rng(b_rng)
                with torch.no_grad():
                    for buffer, old in post_a_buffers:
                        buffer.copy_(old)
            inner_norm, inner_factor = virtual_step(transaction, gradients, alpha,
                                                    float(cfg.get('max_grad_norm', 1.)), norm=norm_a)
            b = side_pass(model, sides['B'], cfg, scaler, backward=beta > 0)
            encoded += len(sides['B'])
            diagnostics.update(A_loss=a['loss'], B_loss=b['loss'], A_after=a_after, B_before=b_before,
                               delta_A=None if a_after is None else a_after - a['loss'],
                               delta_B=None if b_before is None else b['loss'] - b_before,
                               inner_grad_norm=inner_norm, inner_clip_factor=inner_factor,
                               virtual_update_norm=alpha * inner_factor * inner_norm,
                               module_A_gradient_norms=gradient_audit)
            if audit:
                diagnostics['module_virtual_update_norms'] = {
                    name: alpha * inner_factor * value if value is not None else None
                    for name, value in gradient_audit.items()}
                diagnostics['trainable_parameter_elements'] = sum(p.numel() for _, p in transaction.parameters)
                diagnostics['original_parameter_cache_bytes'] = sum(p.numel() * p.element_size()
                                                                     for p in transaction.original)
            if beta > 0:
                # Keep B gradients scaled until the one public unscale_ call.
                # Cached A gradients remain raw for the outer combination.
                if audit:
                    b_gradients = [p.grad.detach() if p.grad is not None else None for _, p in transaction.parameters]
                    diagnostics['module_B_gradient_norms'] = {
                        key: value / scale if value is not None else None
                        for key, value in module_norms(optimizer, transaction, b_gradients).items()}
            transaction.restore_parameters()
            with torch.no_grad():
                for (_, p), g_a in zip(transaction.parameters, gradients):
                    if p.grad is not None:
                        p.grad.mul_(beta / (1 + beta))
                        if g_a is not None:
                            p.grad.add_(g_a, alpha=scale / (1 + beta))
                    elif g_a is not None:
                        p.grad = g_a.mul_(scale / (1 + beta))
            del gradients
        # Buffers must not encode a temporary-model adaptation at deployment.
        # Production has GN/LN; this also protects future mutable buffers.
        transaction.restore_buffers()
        scaler.unscale_(optimizer)
        parameters = [p for _, p in transaction.parameters if p.grad is not None]
        if audit:
            diagnostics['module_outer_gradient_norms'] = module_norms(
                optimizer, transaction, [p.grad for _, p in transaction.parameters])
        # clip_grad_norm_ already computes the full norm. Reuse its return
        # instead of traversing every trainable gradient a second time.
        outer_norm = float(torch.nn.utils.clip_grad_norm_(
            parameters, float(cfg.get('max_grad_norm', 1.)), error_if_nonfinite=False))
        if not math.isfinite(outer_norm):
            raise NonfiniteGradient('Nonfinite combined outer gradient')
    except NonfiniteGradient as exc:
        encoded += exc.encoded_views
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled():
            scaler.load_state_dict(scaler_initial)
            raise
        # Public update(new_scale) resets per-optimizer unscale bookkeeping.
        # An overflow skips the ENTIRE task, never a partially updated model.
        scaler.update(new_scale=scale * float(scaler.get_backoff_factor()))
        return dict(loss=None, classification_loss=None, views=len(units), encoded_views=encoded,
                    auxiliary_reference_views=0, unique_original_sources=len(units), cells={},
                    predicted_fake_by_true_class={'fake': [0, 0], 'real': [0, 0]},
                    raw_ce_sum=None, max_example_ce=None, grad_norm=None,
                    phase='joint' if joint else 'head_warmup',
                    lrs={g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)},
                    amp=dict(enabled=True, previous_scale=scale, scale=scaler.get_scale(),
                             skipped_update=True, reason=str(exc)),
                    meta=dict(alpha=alpha, beta=beta, **diagnostics))
    except BaseException:
        transaction.restore()
        optimizer.zero_grad(set_to_none=True)
        # Reset transient unscale bookkeeping even if an audit/clip failed.
        if scaler.is_enabled():
            scaler.update(new_scale=scale)
        scaler.load_state_dict(scaler_initial)
        raise
    # COMMIT POINT. All temporary parameters have been restored and all
    # gradients validated. This is the only call that can change Adam state.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if audit:
        diagnostics['module_outer_update_norms'] = _update_norms(optimizer, transaction)
    result = _summary(units, a, b, beta, warm=not joint)
    result.update(encoded_views=encoded, grad_norm=outer_norm,
                  amp=dict(enabled=scaler.is_enabled(), previous_scale=scale,
                           scale=scaler.get_scale(), skipped_update=scaler.get_scale() < scale),
                  lrs={g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)},
                  meta=dict(alpha=alpha if joint else 0., beta=beta if joint else 0.,
                            objective='whole_task_balanced_ce' if not joint else '(A + beta*B_virtual)/(1+beta)',
                            diagnostic_extra_views=encoded - len(units), **diagnostics))
    return result
