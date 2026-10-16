"""Scoped FP16 backend compute with FP32 parameters and accumulated gradients."""
import math
import torch


def backend_context(device, precision):
    if precision not in ('fp16', 'fp32'):
        raise ValueError('Unknown AASIST precision: '+str(precision))
    device = torch.device(device)
    return torch.autocast(device.type, dtype=torch.float16,
                          enabled=device.type == 'cuda' and precision == 'fp16')


def scaler_for(optimizer, cfg):
    if not hasattr(optimizer, '_v3189_scaler'):
        enabled = str(cfg.get('device', 'cpu')).startswith('cuda') and cfg.get('backend_precision') == 'fp16'
        optimizer._v3189_scaler = torch.amp.GradScaler('cuda', enabled=enabled, init_scale=256.)
    return optimizer._v3189_scaler


def unscale_and_clip(optimizer, scaler, maximum, *, unscale=True):
    # Exactly once per logical update, after ALL microbatch backwards.
    if unscale:scaler.unscale_(optimizer)
    parameters = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    norm = torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=not scaler.is_enabled())
    value = float(norm)
    if not math.isfinite(value) and scaler.is_enabled():
        # GradScaler detects nonfinite elements, not overflow of a finite norm
        # reduction. Fail rather than commit an invalid clipped update in that
        # exceedingly large-gradient case. Nonfinite elements skip via scaler.
        if all(bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError('Gradient norm overflow with finite elements')
    return value if math.isfinite(value) else None


def commit(optimizer, scaler):
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    after = scaler.get_scale()
    return dict(enabled=scaler.is_enabled(), scale=after, previous_scale=before,
                skipped_update=after < before)
