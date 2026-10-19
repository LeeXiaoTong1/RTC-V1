"""V4 checkpoint tensors include adapters AND the inherited frozen LoRA."""
from w2v_v31810.common import *


def detector_key(name):
    return name.startswith(('head.', 'fusion.', 'adapters.')) or name.endswith(('.lora_a', '.lora_b'))


def partial_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if detector_key(k)}


def apply_partial(model, values, *, parent=False):
    target = {k: v for k, v in model.state_dict().items() if detector_key(k)}
    expected = {k: v for k, v in target.items() if not (parent and k.startswith('adapters.'))}
    if set(values) != set(expected):
        raise ValueError('V4 checkpoint key mismatch; adapters may be absent only for parent initialization')
    for name, value in values.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid V4 checkpoint tensor: ' + name)
    with torch.no_grad():
        for name, value in values.items():
            target[name].copy_(value)
