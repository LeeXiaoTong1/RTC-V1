"""Symmetric PCGrad on complete, balanced Online and Noisy CE gradients."""
import torch


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
