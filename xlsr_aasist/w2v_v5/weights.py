"""Convex checkpoint parameter averaging for one V5 detector.

This is a model soup / SWA operation in the *parameter coordinates* of the
unchanged architecture.  In particular, LoRA A and B are averaged separately:
``mean(B) @ mean(A)`` is generally NOT ``mean(B @ A)``.  The resulting detector
must be evaluated directly; an average of old prediction scores is not a
substitute.  No prediction ensemble or rank expansion is created here.

Use ``invariant_keys(model, states[0])`` at call sites.  It protects every
registered buffer and every frozen parameter, including the retained prefix
adapters.  PCL state and standard normalization running statistics are also
protected without that explicit set, so calibration observations cannot be
accidentally blended into a fictitious calibration state.
"""
import math
from collections.abc import Mapping

import torch


def invariant_keys(model, state=None):
    """Return saved keys which must remain identical across active snapshots.

    ``state`` should normally be the partial detector snapshot.  Omitting it
    uses all model keys; no tensor is copied in either case.
    """
    keys = set(model.state_dict()) if state is None else set(state)
    buffers = {name for name, _ in model.named_buffers(remove_duplicate=False)}
    frozen = {name for name, parameter in model.named_parameters(remove_duplicate=False)
              if not parameter.requires_grad}
    return keys & (buffers | frozen)


def _protected(name, value, constant_keys):
    return (name in constant_keys or name.startswith("pcl_scale.")
            or not value.is_floating_point()
            or name.rsplit(".", 1)[-1] in
            ("running_mean", "running_var", "num_batches_tracked"))


def average_states(states, coefficients, *, constant_keys=()):
    """Return a detached CPU snapshot with unchanged keys, shapes and dtypes.

    Coefficients must be finite and nonnegative, with positive total mass.
    They are normalized internally.  Zero-mass snapshots are still validated
    for compatible tensor geometry, but need not share invariant values.
    Positive-mass snapshots must agree exactly on all protected values.
    Inputs are never mutated; even an endpoint returns independent clones.
    """
    states, coefficients = list(states), list(coefficients)
    if not states or len(states) != len(coefficients):
        raise ValueError("Checkpoint snapshots and coefficients must have equal nonzero length")
    try:
        weights = [float(value) for value in coefficients]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Checkpoint coefficients must be finite nonnegative scalars") from error
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("Checkpoint coefficients must be finite nonnegative scalars")
    try:
        total = math.fsum(weights)
    except OverflowError as error:
        raise ValueError("Checkpoint coefficients need a finite positive total") from error
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Checkpoint coefficients need a finite positive total")
    weights = [value / total for value in weights]
    active = [index for index, value in enumerate(weights) if value > 0]
    if not isinstance(states[0], Mapping) or not states[0]:
        raise ValueError("Checkpoint snapshots must be nonempty tensor mappings")
    keys = set(states[0])
    if not all(isinstance(key, str) for key in keys):
        raise ValueError("Checkpoint tensor names must be strings")
    constant_keys = set(constant_keys)
    if not all(isinstance(key, str) for key in constant_keys):
        raise ValueError("Invariant tensor names must be strings")
    if not constant_keys <= keys:
        raise ValueError("Invariant keys are missing from checkpoint: "
                         + ", ".join(sorted(constant_keys - keys)))

    # Validate every snapshot before allocating a result.  Do not silently
    # intersect keys, cast dtypes, or average partial/mismatched checkpoints.
    for state in states:
        if not isinstance(state, Mapping) or set(state) != keys:
            raise ValueError("Checkpoint tensor key sets differ")
        for name, reference in states[0].items():
            value = state[name]
            if (not isinstance(reference, torch.Tensor)
                    or not isinstance(value, torch.Tensor)
                    or value.layout != torch.strided or reference.layout != torch.strided
                    or value.is_complex() or reference.is_complex()
                    or value.is_quantized or reference.is_quantized):
                raise ValueError("Unsupported checkpoint tensor: " + name)
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError("Checkpoint tensor shape or dtype differs: " + name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite checkpoint tensor: " + name)

    result = {}
    anchor = states[active[0]]
    for name in states[0]:
        reference = anchor[name].detach().cpu()
        if _protected(name, reference, constant_keys):
            for index in active[1:]:
                if not torch.equal(reference, states[index][name].detach().cpu()):
                    raise ValueError("Invariant checkpoint tensor differs: " + name)
            result[name] = reference.clone()
        elif len(active) == 1:
            # Preserve exact predictions for parent-only / continuation-only
            # endpoints, including low precision values and signed zeros.
            result[name] = reference.clone()
        else:
            accumulated = torch.zeros(reference.shape, dtype=torch.float64)
            for index in active:
                accumulated.add_(states[index][name].detach().to(
                    device="cpu", dtype=torch.float64), alpha=weights[index])
            value = accumulated.to(reference.dtype)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Averaged checkpoint tensor became nonfinite: " + name)
            result[name] = value
    return result
