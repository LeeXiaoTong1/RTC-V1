"""Masked temporal repair with an identity-initialized evidence bypass."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F


def length_mask(x, lengths):
    if len(lengths) != len(x) or any(n <= 0 or n > x.shape[1] for n in lengths):
        raise ValueError("One valid positive length per recording is required")
    return torch.arange(x.shape[1], device=x.device)[None] < torch.as_tensor(lengths, device=x.device)[:, None]


def masked(x, mask):
    # Multiplication would leave NaN padding in place.
    return x.masked_fill(~mask[..., None], 0.)


class TemporalBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        hidden = 2 * channels
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, hidden)
        self.filters = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel, padding=kernel//2,
                                               groups=hidden) for kernel in (3, 7, 15))
        self.contract = nn.Linear(hidden, channels)

    def forward(self, x, valid):
        x = masked(x, valid)
        z = masked(F.silu(self.expand(masked(self.norm(x), valid))), valid)
        # Mask after every biased temporal block, not just at its final output.
        branches = [masked(layer(z.transpose(1, 2)).transpose(1, 2), valid) for layer in self.filters]
        z = sum(branches) / len(branches)
        return masked(x + self.contract(F.silu(z)), valid)


class FeatureRepair(nn.Module):
    def __init__(self, channels=128, seed=54001, initial_gate=.1):
        super().__init__()
        if not 0 < initial_gate < 1:
            raise ValueError("Gate must start nonzero")
        # Initialization must not consume the continuing run's sampling RNG.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed would also reseed CUDA generators outside this
            # CPU-only fork. Initialize on CPU, then let the caller move us.
            cpu_seed = torch.Generator(device='cpu').manual_seed(seed)
            torch.set_rng_state(cpu_seed.get_state())
            self.blocks = nn.ModuleList([TemporalBlock(channels), TemporalBlock(channels)])
            self.out_norm = nn.LayerNorm(channels)
            self.out = nn.Linear(channels, channels)
            self.gate = nn.Sequential(nn.LayerNorm(3*channels), nn.Linear(3*channels, 64),
                                      nn.SiLU(), nn.Linear(64, channels))
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, math.log(initial_gate/(1-initial_gate)))
        self.register_buffer("frame_scale", torch.ones(channels))
        self.register_buffer("scale_ready", torch.tensor(False))

    def repair(self, x, valid):
        x = masked(x, valid)
        z = x
        for block in self.blocks:
            z = block(z, valid)
        delta = masked(self.out(masked(self.out_norm(z), valid)), valid)
        return x + delta, delta

    def forward(self, x, lengths):
        valid = length_mask(x, lengths)
        raw = masked(x, valid)
        repaired, delta = self.repair(raw, valid)
        gate = masked(self.gate(torch.cat([raw, repaired, delta], dim=-1)).sigmoid(), valid)
        fused = masked(raw + gate*delta, valid)
        return dict(fused=fused, repaired=repaired, delta=delta, gate=gate)

    def auxiliary_prediction(self, x, lengths):
        # Only this cheap processor is recomputed, never the Omni encoder.
        return self.repair(x.detach(), length_mask(x, lengths))[0]


def aligned_reference(clean, clean_valid, noisy_frames, shift_frames):
    """Positive shift means Noisy is delayed: noisy[t] maps to clean[t-shift].

    shift_frames = (noisy_delay_samples - clean_delay_samples) / 320.
    Both interpolation neighbours must be reliable; exact integer positions
    require only that frame. Caller adds Noisy-side reliability/activity mask.
    """
    q = torch.arange(noisy_frames, device=clean.device, dtype=torch.float32) - shift_frames
    lower, upper = q.floor().long(), q.ceil().long()
    valid = (lower >= 0) & (upper < len(clean))
    lo, hi = lower.clamp(0, len(clean)-1), upper.clamp(0, len(clean)-1)
    valid &= clean_valid[lo] & clean_valid[hi]
    alpha = (q-lower).unsqueeze(-1)
    target = (1-alpha)*clean.detach()[lo] + alpha*clean.detach()[hi]
    return target.masked_fill(~valid[:, None], 0.), valid


def source_feature_loss(prediction, reference, mask, scale, min_frames=4):
    if torch.any(scale <= 0) or not torch.isfinite(scale).all():
        raise ValueError("Positive fixed frame scale required")
    active = mask.sum(1) >= min_frames
    effective = mask & active[:, None]
    # Replace invalid frames before arithmetic; they may contain padding NaNs.
    residual = (masked(prediction, effective) - masked(reference.detach(), effective)) / scale.detach()
    element = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none", beta=1.)
    per_frame = element.mean(-1).masked_fill(~effective, 0.)
    values = per_frame.sum(1) / effective.sum(1).clamp_min(1)
    return values, active


def planned_group_mean(values, group_ids, budgets):
    """Four groups each have 1/4 mass, even when some eligible pairs are absent.

    Values for invalid sources are zero. Never renormalize over survivors or
    frames, which would overweight a lone usable source or long recording.
    """
    if len(budgets) != 4 or any(q <= 0 for q in budgets):
        raise ValueError("Positive planned budgets for four groups required")
    if len(group_ids) != len(values) or any(g not in range(4) for g in group_ids):
        raise ValueError("Invalid source group")
    weights = values.new_tensor([1./(4*budgets[g]) for g in group_ids])
    return (weights*values).sum()


def source_kd_loss(student_logits, teacher_logits, labels, pair_valid,
                   temperature=3., confidence=.9, advantage=.02):
    teacher = teacher_logits.detach().float()
    student = student_logits.float()
    with torch.no_grad():
        pc = teacher.softmax(-1)
        pn = student.softmax(-1)
        truth_c = pc.gather(1, labels[:, None]).squeeze(1)
        truth_n = pn.gather(1, labels[:, None]).squeeze(1)
        active = pair_valid & (pc.argmax(-1) == labels) & (truth_c >= confidence)
        active &= truth_c >= truth_n + advantage
    loss = F.kl_div(F.log_softmax(student/temperature, -1),
                    F.softmax(teacher/temperature, -1), reduction="none").sum(-1) * temperature**2
    return loss.clamp_min(0).masked_fill(~active, 0.), active


@contextmanager
def deterministic_head(head):
    """Evaluation mode still permits input/parameter gradients. Restore all modes."""
    modes = [(module, module.training) for module in head.modules()]
    try:
        head.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def migrated_anchor(old, initialized):
    """New state keys may only belong to repair; no non-strict load shortcut."""
    if set(old) - set(initialized):
        raise ValueError("Unexpected parent keys")
    additions = set(initialized) - set(old)
    if any(not k.startswith("repair.") for k in additions):
        raise ValueError("Missing existing model parameters")
    result = {}
    for key, target in initialized.items():
        value = old.get(key, target)
        if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
            raise ValueError("Invalid parameter: " + key)
        result[key] = value.detach().clone()
    return result
