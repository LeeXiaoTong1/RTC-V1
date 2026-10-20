"""Symmetric PCGrad on complete, balanced Online and Noisy CE gradients."""
import torch


KINDS = ('online', 'noisy', 'official', 'noise')


class GradientWorkspace:
    """Persistent FP32 accumulators; parameter gradients view the merged buffer."""
    def __init__(self, params):
        self.signature = tuple(id(p) for p in params)
        self.params = params
        count = sum(p.numel() for p in params)
        self.storage = torch.empty((7, count), device=params[0].device, dtype=torch.float32)
        self.buffers = dict(zip(KINDS, self.storage[:4]))
        self.views = {k: list(v.split([p.numel() for p in params])) for k, v in self.buffers.items()}
        self.used = [False]*len(params)

    def reset(self):
        self.storage[:4].zero_(); self.used[:] = [False]*len(self.params)

    def add(self, kind, gradients):
        targets, values = [], []
        for i, (target, grad) in enumerate(zip(self.views[kind], gradients)):
            if grad is not None:
                targets.append(target.view_as(grad)); values.append(grad.detach()); self.used[i] = True
        if targets: torch._foreach_add_(targets, values)

    def coordinate(self, enabled, scale, epsilon=1e-12):
        # Unscale before reductions, preserving V4's overflow/PCGrad convention.
        self.storage[:4].div_(scale)
        c, n, official, noise = self.storage[:4]
        auxiliary, combined, correction = self.storage[4:]
        auxiliary.copy_(official).add_(noise)
        finite = torch.isfinite(self.storage[:4]).all()
        cc, nn, dot = torch.dot(c, c), torch.dot(n, n), torch.dot(c, n)
        cn, n_norm, an = cc.sqrt(), nn.sqrt(), auxiliary.norm()
        valid = finite & (cn > epsilon) & (n_norm > epsilon)
        applied = valid & (dot < 0) & enabled
        zero = dot.new_zeros(())
        nc = torch.where(applied, -dot/nn.clamp_min(epsilon**2), zero)
        cc_coeff = torch.where(applied, -dot/cc.clamp_min(epsilon**2), zero)
        correction.copy_(n).mul_(nc).addcmul_(c, cc_coeff)
        # On a nonfinite input avoid inf*0 altering the original skip semantics.
        correction.masked_fill_(~applied, 0.)
        combined.copy_(c).add_(n).add_(correction)
        denom = combined.norm()*an
        nan = dot.new_full((), float('nan'))
        cosine = torch.where(valid, (dot/(cn*n_norm).clamp_min(epsilon**2)).clamp(-1,1), nan)
        aux_cos = torch.where(finite & (denom > epsilon), (torch.dot(combined,auxiliary)/denom.clamp_min(epsilon)).clamp(-1,1), nan)
        stats = torch.stack((finite.float(), applied.float(), cosine, cn, n_norm, an,
                             correction.norm(), aux_cos, official.norm(), noise.norm()))
        combined.add_(auxiliary).mul_(scale)
        offset = 0
        for used, p in zip(self.used, self.params):
            size = p.numel(); p.grad = combined[offset:offset+size].view_as(p) if used else None; offset += size
        return stats


def workspace_for(optimizer, params):
    value = getattr(optimizer, '_v41_workspace', None)
    if value is None or value.signature != tuple(id(p) for p in params):
        value = optimizer._v41_workspace = GradientWorkspace(params)
    value.reset()
    return value


def unpack_stats(values):
    import math
    finite, applied, cosine, cn, nn, an, correction, aux_cos = values[:8]
    clean = lambda v: v if math.isfinite(v) else None
    return dict(finite=bool(finite), applied=bool(applied), cosine=clean(cosine),
        clean_norm=clean(cn), noisy_norm=clean(nn), auxiliary_norm=clean(an),
        correction_norm=clean(correction), auxiliary_cosine=clean(aux_cos))


def coordinate(clean, noisy, auxiliary, *, enabled=True, scale=1., epsilon=1e-12):
    # Work unscaled in FP32. Missing parameter gradients are explicit zeros.
    c, n, a = (torch.cat([g.reshape(-1) for g in values]).float()/scale
               for values in (clean, noisy, auxiliary))
    stats = dict(applied=False, finite=True, cosine=None, clean_norm=None, noisy_norm=None,
                 auxiliary_norm=None, correction_norm=None, auxiliary_cosine=None)
    if not all(bool(torch.isfinite(g).all()) for g in (c, n, a)):
        stats['finite'] = False
        return (c+n+a)*scale, stats  # let GradScaler skip the ENTIRE update
    cc, nn, dot = torch.dot(c, c), torch.dot(n, n), torch.dot(c, n)
    cn, n_norm = cc.sqrt(), nn.sqrt()
    stats.update(clean_norm=float(cn), noisy_norm=float(n_norm), auxiliary_norm=float(a.norm()))
    valid = bool((cn > epsilon) & (n_norm > epsilon))
    if valid:
        stats['cosine'] = float((dot/(cn*n_norm)).clamp(-1, 1))
    combined = c+n
    if enabled and valid and bool(dot < 0):
        # Both projections use ORIGINAL gradients, never an already projected one.
        correction = -(dot/nn)*n - (dot/cc)*c
        combined = combined + correction
        stats.update(applied=True, correction_norm=float(correction.norm()))
    else:
        stats['correction_norm'] = 0.
    denom = combined.norm()*a.norm()
    if bool(denom > epsilon):
        stats['auxiliary_cosine'] = float((torch.dot(combined, a)/denom).clamp(-1, 1))
    # Auxiliaries remain separate. No claim of conflict-free final Adam update.
    return (combined+a)*scale, stats
