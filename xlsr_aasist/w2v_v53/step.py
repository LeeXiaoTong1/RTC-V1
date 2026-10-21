"""Unchanged V5.2 objectives, with infrequent conditioner diagnostics."""
import torch
from w2v_v52.step import train_step as inherited_step


def train_step(model, optimizer, units, cfg, alpha=0., beta=1., joint=True, audit=False):
    fusion = model.fusion
    fusion.reset_observations(enabled=audit)
    before = [p.detach().clone() for p in fusion.conditioner.parameters()] if audit else []
    try:
        result = inherited_step(model, optimizer, units, cfg, alpha, beta, joint, audit)
        if audit:
            with torch.no_grad():
                change = sum((p.detach()-old).float().square().sum()
                             for p,old in zip(fusion.conditioner.parameters(), before))
                result['adaptive_fusion'] = dict(fusion.observations(),
                    conditioner_update_norm=float(change.sqrt()), mode=fusion.mode,
                    committed=not result['amp']['skipped_update'], extra_encoder_passes=0)
        return result
    finally:
        # Never retain diagnostic tensors across updates or AMP retries.
        fusion.reset_observations()
