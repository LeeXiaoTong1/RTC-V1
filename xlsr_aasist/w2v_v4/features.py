"""Official w2v-BERT filterbanks with explicit full-waveform time geometry."""
import numpy as np
import torch

SAMPLE_RATE = 16000
MIN_SAMPLES = 3920
FRAME_SUPPORT = 560
FRAME_STRIDE = 320


def frame_count(samples, *, extend_short=True):
    samples = max(MIN_SAMPLES, int(samples)) if extend_short else int(samples)
    return max(0, (samples - FRAME_SUPPORT) // FRAME_STRIDE + 1)


def frame_intervals(length, sample_rate=SAMPLE_RATE):
    starts = np.arange(int(length), dtype=np.float64) * FRAME_STRIDE
    return np.stack((starts, starts + FRAME_SUPPORT), -1) / sample_rate


def prepare_wave(wave):
    """Retain the complete recording; only mirror genuinely short inputs."""
    value = np.asarray(wave, dtype=np.float32)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Expected a nonempty finite mono waveform')
    original = len(value)
    if original < MIN_SAMPLES:
        value = np.pad(value, (0, MIN_SAMPLES-original), mode='reflect' if original > 1 else 'edge')
    return np.ascontiguousarray(value), original


class OfficialFeatures:
    def __init__(self, extractor):
        if (extractor.sampling_rate, extractor.num_mel_bins, extractor.stride) != (16000, 80, 2):
            raise ValueError('V4 requires the official 16 kHz / 80 mel / stride-2 extractor')
        self.extractor = extractor

    @classmethod
    def from_pretrained(cls, path):
        from transformers import AutoFeatureExtractor
        return cls(AutoFeatureExtractor.from_pretrained(path, local_files_only=True))

    def __call__(self, waves, device=None):
        if not waves:
            raise ValueError('Cannot encode an empty waveform batch')
        prepared = [prepare_wave(w) for w in waves]
        # Extraction computes CMVN independently BEFORE padding. Odd final
        # fbank frames are masked according to the official stride-2 rule.
        result = self.extractor([w for w, _ in prepared], sampling_rate=16000,
            padding=True, pad_to_multiple_of=2, truncation=False,
            return_attention_mask=True, return_tensors='pt')
        features = result['input_features'].float()
        mask = result['attention_mask'].bool()
        lengths = mask.sum(-1).tolist()
        expected = [frame_count(len(w), extend_short=False) for w, _ in prepared]
        if lengths != expected or features.shape[-1] != 160:
            raise ValueError('Official feature geometry differs from the V4 time contract')
        if not bool(torch.isfinite(features).all()) or not all(lengths):
            raise ValueError('Invalid official filterbank output')
        if bool((~mask[:, :-1] & mask[:, 1:]).any()):
            raise ValueError('Feature masks must contain a valid prefix')
        features = features[:, :max(lengths)].contiguous()
        mask = mask[:, :max(lengths)].contiguous()
        if device is not None:
            features, mask = features.to(device), mask.to(device)
        return features, mask, lengths
