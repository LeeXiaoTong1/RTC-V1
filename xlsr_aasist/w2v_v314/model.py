"""Restore the existing adapter first; only then unfreeze the output projection."""
import torch

from w2v_v313.model import load_model as source_model


def joint_mode(model, cfg):
    model.configure_trainable_layers(cfg['trainable_layers'])
    model.head.requires_grad_(True)
    return model


def load_model(cfg, device=None, training=True):
    # V3.12's strict partial loader expects its output Linear to be frozen.
    # Changing that before restoration would invalidate its parameter inventory.
    model = source_model(cfg, device, training)
    return joint_mode(model, cfg)


def optimizer_for(model, cfg):
    groups = []
    count = len(model.backbone.encoder.layers)
    for index, layer in enumerate(model.backbone.encoder.layers):
        params = [p for p in layer.parameters() if p.requires_grad]
        if params:
            lr = cfg['encoder_lr'] * cfg['layer_decay'] ** (count - 1 - index)
            groups.append(dict(params=params, lr=lr, initial_lr=lr, name=f'encoder_{index}'))
    adapter = list(model.head.classifier.adapter.parameters())
    output = list(model.head.classifier[-1].parameters())
    special = {id(p) for p in adapter + output}
    backbone_head = [p for p in model.head.parameters() if p.requires_grad and id(p) not in special]
    for name, params, lr in (('multiconv_and_first_fc', backbone_head, cfg['head_lr']),
                              ('feature_adapter', adapter, cfg['adapter_lr']),
                              ('final_binary_projection', output, cfg['output_lr'])):
        groups.append(dict(name=name, params=params, lr=lr, initial_lr=lr))
    actual = [p for g in groups for p in g['params']]
    expected = [p for p in model.parameters() if p.requires_grad]
    if len({id(p) for p in actual}) != len(actual) or {id(p) for p in actual} != {id(p) for p in expected}:
        raise ValueError('Optimizer must cover every trainable parameter exactly once')
    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
