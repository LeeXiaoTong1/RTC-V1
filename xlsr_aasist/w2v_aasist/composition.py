"""Same-source, label-preserving RAM compositions; never modify cache files."""
import numpy as np

CUT = 64600
MODES = ('single', 'single', 'switch', 'prefix_tail')


def compose(first, second, original, mode, rng, fade=320):
    first = np.asarray(first, dtype=np.float32)
    if first.shape != (CUT,) or not np.isfinite(first).all():
        raise ValueError('Invalid cached view')
    if mode == 'single':
        return first.copy(), mode
    if mode == 'switch':
        second = np.asarray(second, dtype=np.float32)
        if second.shape != first.shape or not np.isfinite(second).all():
            raise ValueError('A second aligned cached view is required')
        point = int(rng.integers(CUT // 4, 3 * CUT // 4))
        start, end = point - fade // 2, point + fade // 2
        alpha = np.linspace(0., 1., end - start, dtype=np.float32)
        out = first.copy()
        out[start:end] = (1 - alpha) * first[start:end] + alpha * second[start:end]
        out[end:] = second[end:]
        return out, mode
    if mode == 'prefix_tail':
        original = np.asarray(original, dtype=np.float32)
        if original.ndim != 1 or not np.isfinite(original).all():
            raise ValueError('Original source is invalid')
        if len(original) <= CUT:
            return first.copy(), 'single_short_source'
        out = original.copy()
        out[:CUT] = first
        alpha = np.linspace(0., 1., fade, dtype=np.float32)
        out[CUT-fade:CUT] = (1 - alpha) * first[-fade:] + alpha * original[CUT-fade:CUT]
        return out, mode
    raise ValueError('Unknown composition mode')
