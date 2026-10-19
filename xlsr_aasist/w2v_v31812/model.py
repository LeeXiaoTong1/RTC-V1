"""One encoder, one original classifier; exact parent behavior at initialization."""
import copy
import torch
from torch import nn
from w2v_v31810.precision import backend_context


def global_statistics(sequences, original_samples=None):
    """Population moments before windowing; exclude batch and reflected padding.

    For sub-400-sample inputs no fully real CNN frame exists. Their global
    evidence is explicitly zero; the unchanged local path retains its context.
    The transformer can still attend to short-input reflection, as in the parent.
    """
    if original_samples is not None and len(original_samples) != len(sequences):
        raise ValueError('Length metadata must match recordings')
    values = []
    for i, seq in enumerate(sequences):
        if seq.ndim != 2 or seq.shape[1] != 128 or not len(seq):
            raise ValueError('Expected unpadded T x 128 fusion features')
        n = len(seq) if original_samples is None else min(len(seq), max(0, (int(original_samples[i])-400)//320+1))
        x = seq[:n].float()
        values.append(torch.cat((x.mean(0), x.std(0, correction=0))) if n else seq.new_zeros(256).float())
    return torch.stack(values)


def evidence(parent, waves, original_samples=None, correction=None):
    """Capture the exact classifier input in the sole backend pass.

    No fork/copy of AASIST equations. A short-lived pre-hook is always removed,
    including on exceptions. This wrapper is single-threaded like the parent.
    """
    seq = parent.encode(waves)
    glob = global_statistics(seq, original_samples)
    classifier = parent.head.aasist.out_layer
    if classifier.in_features != 384 or classifier.out_features != 2:
        raise ValueError('V3.18.12 requires the complete parent 384 -> 2 classifier')
    captured = []
    def before(module, inputs):
        local = inputs[0].float()
        captured.append(local)
        if correction is not None:
            with torch.autocast(local.device.type, enabled=False):
                local = local + correction(glob, local)
            return (local,)
    handle = classifier.register_forward_pre_hook(before)
    try:
        logits = parent.classify_sequences(seq)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (len(waves), 384):
        raise RuntimeError('Expected one final classifier invocation per recording batch')
    return glob, captured[0], logits


class Residual(nn.Module):
    def __init__(self, mean=None, scale=None, hidden=128, dropout=.1):
        super().__init__()
        self.register_buffer('mean', torch.zeros(640) if mean is None else mean.float().clone())
        self.register_buffer('scale', torch.ones(640) if scale is None else scale.float().clamp_min(1e-4).clone())
        self.net = nn.Sequential(nn.Linear(640, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 384))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, glob, local):
        return self.net((torch.cat((glob.float(), local.float()), -1)-self.mean)/self.scale)


class CachedDetector(nn.Module):
    def __init__(self, classifier, residual, precision='fp32'):
        super().__init__()
        self.classifier = copy.deepcopy(classifier).float().requires_grad_(False)
        self.residual, self.precision = residual, precision

    def forward(self, glob, local):
        with torch.autocast(local.device.type, enabled=False):
            value = local.float() + self.residual(glob, local)
        with backend_context(local.device, self.precision):
            return self.classifier(value).float()


class Detector(nn.Module):
    """Deployment path: frozen parent plus a single learned residual, no ensemble."""
    def __init__(self, parent, residual):
        super().__init__()
        self.parent = parent.requires_grad_(False).eval()
        self.residual = residual

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, waves):
        from w2v_v3189.data import ready
        lengths = [len(w) for w in waves]
        return evidence(self.parent, [ready(w) for w in waves], lengths, self.residual)[2]
