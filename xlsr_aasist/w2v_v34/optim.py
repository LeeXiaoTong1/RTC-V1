"""Explicit encoder-block adaptation with persisted layer-wise learning rates.

The input projection and all non-block backbone parameters remain frozen. Update
diagnostics sample fixed coordinates; their norms are not whole-model norms.
"""
import math
import torch


def _layers(model):
    layers = model.backbone.encoder.layers
    if not 1 <= len(layers) <= 24:
        raise ValueError('V3.4 expects between 1 and 24 encoder blocks')
    return layers


def _count(model, cfg):
    count = cfg.get('trainable_layers', 24)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(_layers(model)):
        raise ValueError('trainable_layers must select 1..number of encoder blocks')
    return count


def configure_model(model, cfg):
    """Enable the last N complete encoder blocks and the complete head only."""
    layers = _layers(model)
    count = _count(model, cfg)
    model.configure_trainable_layers(count)
    expected = {id(p) for layer in layers[-count:] for p in layer.parameters()}
    expected.update(id(p) for p in model.head.parameters())
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != expected:
        raise RuntimeError('Trainable parameters differ from the requested blocks and complete head')
    return model


def optimizer_for(model, cfg):
    """Create AdamW groups with stable names and no frozen or repeated tensors."""
    configure_model(model, cfg)
    encoder_lr = cfg.get('encoder_lr', 5e-7)
    head_lr = cfg.get('joint_head_lr', cfg.get('head_lr', 2e-6))
    layer_decay = cfg.get('layer_decay', .9)
    weight_decay = cfg.get('weight_decay', 1e-4)
    for value, name in ((encoder_lr, 'encoder_lr'), (head_lr, 'head_lr')):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if isinstance(layer_decay, bool) or not math.isfinite(layer_decay) or not 0 < layer_decay <= 1:
        raise ValueError('layer_decay must be finite and in (0,1]')
    if isinstance(weight_decay, bool) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    layers = _layers(model)
    groups, seen = [], set()
    modules = [(f'encoder.layer_{i:02d}', layer,
                encoder_lr * layer_decay ** (len(layers) - 1 - i))
               for i, layer in enumerate(layers)]
    modules.append(('head', model.head, head_lr))
    for prefix, module, base_lr in modules:
        named = list(module.named_parameters(remove_duplicate=False))
        for decay in (True, False):
            params = []
            for name, parameter in named:
                if not parameter.requires_grad or (parameter.ndim > 1 and not name.endswith('bias')) != decay:
                    continue
                if id(parameter) in seen:
                    raise ValueError('A parameter is shared between optimizer groups: ' + prefix + '.' + name)
                params.append(parameter)
                seen.add(id(parameter))
            if params:
                groups.append(dict(params=params, name=prefix + ('.decay' if decay else '.no_decay'),
                                   lr=base_lr, base_lr=base_lr,
                                   weight_decay=weight_decay if decay else 0.))
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected:
        raise RuntimeError('Optimizer groups omit eligible parameters or contain foreign parameters')
    return torch.optim.AdamW(groups, eps=1e-8)


def apply_learning_rates(optimizer, scale):
    """Apply one schedule multiplier without flattening the layer-wise ratios."""
    if isinstance(scale, bool) or not math.isfinite(scale) or scale < 0:
        raise ValueError('Learning-rate scale must be finite and nonnegative')
    names = [g.get('name') for g in optimizer.param_groups]
    if any(not isinstance(n, str) or not n for n in names) or len(names) != len(set(names)):
        raise ValueError('Unique named parameter groups are required')
    rates = {}
    for group in optimizer.param_groups:
        base = group.get('base_lr')
        if base is None or not math.isfinite(base) or base <= 0:
            raise ValueError('Optimizer checkpoint lacks a valid persisted base_lr')
        rates[group['name']] = float(base * scale)
    for group in optimizer.param_groups:
        group['lr'] = rates[group['name']]
    return rates


class UpdateDiagnostics:
    """Small, RNG-free warm-origin samples and optional last-update gradients.

    Capture once after warm initialization. Save/load this object's state with
    resume checkpoints so relative deltas retain the same reference. Call
    record_gradients after an optimizer step and before the next zero_grad, only
    at diagnostic boundaries; gradients are then the actual post-clip gradients.
    Sampling cannot establish that every coordinate updated.
    """
    SCHEMA = 'v34_fixed_coordinate_updates_v1'

    def __init__(self, model, samples_per_parameter=16):
        if isinstance(samples_per_parameter, bool) or not isinstance(samples_per_parameter, int) or samples_per_parameter < 1:
            raise ValueError('Positive integer samples_per_parameter required')
        self.samples_per_parameter = samples_per_parameter
        self.layout = []
        self._groups = {}
        layer_ids = {id(p): f'encoder.layer_{i:02d}' for i, layer in enumerate(_layers(model))
                     for p in layer.parameters()}
        head_ids = {id(p) for p in model.head.parameters()}
        for name, parameter in model.named_parameters():
            group = layer_ids.get(id(parameter), 'head' if id(parameter) in head_ids else 'backbone.frozen')
            count = min(samples_per_parameter, parameter.numel())
            if not count:
                continue
            indices = (torch.arange(count, dtype=torch.long) * (parameter.numel() - 1) // max(count - 1, 1))
            self.layout.append(dict(name=name, shape=list(parameter.shape), group=group, count=count))
            self._groups.setdefault(group, []).append((name, indices))
        self.origin = self._sample(model)
        self.last_gradients = {}

    def _sample(self, model, gradients=False):
        parameters = dict(model.named_parameters())
        for entry in self.layout:
            parameter = parameters.get(entry['name'])
            if parameter is None or list(parameter.shape) != entry['shape']:
                raise ValueError('Model parameter layout differs from diagnostic origin')
        result = {}
        for group, entries in self._groups.items():
            values = []
            for name, indices in entries:
                parameter = parameters[name]
                value = parameter.grad if gradients else parameter
                if value is None:
                    selected = torch.zeros(len(indices), device=parameter.device)
                else:
                    selected = value.detach().reshape(-1).index_select(0, indices.to(value.device)).float()
                values.append(selected)
            result[group] = torch.cat(values).to(device='cpu', dtype=torch.float64)
        return result

    def record_gradients(self, model):
        samples = self._sample(model, gradients=True)
        parameters = dict(model.named_parameters())
        self.last_gradients = {
            group: dict(sampled_gradient_l2=float(value.norm()),
                        parameters_with_grad=sum(parameters[name].grad is not None for name, _ in self._groups[group]))
            for group, value in samples.items()}
        return self.last_gradients

    def report(self, model):
        current = self._sample(model)
        parameters = dict(model.named_parameters())
        groups = {}
        for group, value in current.items():
            origin = self.origin[group]
            delta = value - origin
            delta_norm, origin_norm = float(delta.norm()), float(origin.norm())
            groups[group] = dict(sampled_coordinates=value.numel(), sampled_origin_l2=origin_norm,
                sampled_delta_l2=delta_norm, sampled_relative_delta=delta_norm / max(origin_norm, 1e-12),
                sampled_changed_coordinates=int(torch.count_nonzero(delta)),
                parameter_count=len(self._groups[group]),
                trainable_parameter_count=sum(parameters[name].requires_grad for name, _ in self._groups[group]),
                **self.last_gradients.get(group, {}))
        return dict(schema=self.SCHEMA, reference='warm initialization',
                    method='fixed evenly spaced coordinates per parameter; sampled norms, not full norms',
                    samples_per_parameter=self.samples_per_parameter, groups=groups)

    def state_dict(self):
        return dict(schema=self.SCHEMA, samples_per_parameter=self.samples_per_parameter,
                    layout=self.layout, origin={k: v.clone() for k, v in self.origin.items()},
                    last_gradients=self.last_gradients)

    def load_state_dict(self, state):
        if (state.get('schema') != self.SCHEMA or state.get('samples_per_parameter') != self.samples_per_parameter
                or state.get('layout') != self.layout or set(state.get('origin', {})) != set(self.origin)):
            raise ValueError('Update diagnostic checkpoint differs from the current parameter layout')
        restored = {}
        for group, value in state['origin'].items():
            if not isinstance(value, torch.Tensor) or value.shape != self.origin[group].shape or not torch.isfinite(value).all():
                raise ValueError('Invalid update diagnostic warm-origin samples')
            restored[group] = value.detach().to(device='cpu', dtype=torch.float64).clone()
        self.origin = restored
        self.last_gradients = dict(state.get('last_gradients', {}))
