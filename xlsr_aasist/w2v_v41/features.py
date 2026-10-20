"""Official w2v-BERT filterbanks with explicit full-waveform time geometry."""
import numpy as np
import torch
from collections import OrderedDict

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

    def one(self, wave):
        """Keep the masked odd tail too: it can be present in a longer batch."""
        wave, _ = prepare_wave(wave)
        result = self.extractor([wave], sampling_rate=16000, padding=True,
            pad_to_multiple_of=2, truncation=False, return_attention_mask=True, return_tensors='pt')
        value = result['input_features'][0].float().contiguous()
        mask = result['attention_mask'][0].bool()
        length = int(mask.sum())
        if length != frame_count(len(wave), extend_short=False) or value.shape[-1] != 160:
            raise ValueError('Invalid worker feature geometry')
        if not torch.isfinite(value).all() or not length or bool((~mask[:-1] & mask[1:]).any()):
            raise ValueError('Invalid worker filterbanks')
        return value, length

    def pack(self, rows, device):
        lengths = [r['mel_length'] for r in rows]
        if not lengths or min(lengths) < 1:
            raise ValueError('Invalid cached frame lengths')
        maximum = max(lengths)
        # Reproduce the official padding value AND the partially padded odd tail.
        features = torch.full((len(rows), maximum, 160), float(self.extractor.padding_value), dtype=torch.float32, device='cpu')
        mask = torch.arange(maximum)[None] < torch.tensor(lengths)[:, None]
        for i, row in enumerate(rows):
            value = row['mel']
            if value.device.type != 'cpu' or value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != 160 or len(value) < lengths[i]:
                raise ValueError('Invalid prepared CPU features')
            n = min(maximum, len(value)); features[i, :n].copy_(value[:n])
        if torch.device(device).type == 'cuda':
            features, mask = features.pin_memory(), mask.pin_memory()
        return features.to(device, non_blocking=True), mask.to(device, non_blocking=True), lengths


class FeatureCache:
    """Process-local bounded LRU, only for verified unaugmented references."""
    def __init__(self, features, cap_mib=64):
        self.features, self.cap = features, int(cap_mib)*1024**2
        self.values, self.bytes, self.hits, self.misses = OrderedDict(), 0, 0, 0

    def attach(self, row, static=False):
        key = (row.get('audio', row.get('id')), row.get('audio_size'),
               row.get('audio_mtime_ns'), row['original_samples']) if static else None
        if key is not None and key in self.values:
            value, length = self.values[key]; self.values.move_to_end(key); self.hits += 1
        else:
            value, length = self.features.one(row['wave']); self.misses += 1
            size = value.numel()*value.element_size()
            if key is not None and size <= self.cap:
                while self.values and self.bytes+size > self.cap:
                    _, (old, _) = self.values.popitem(last=False); self.bytes -= old.numel()*old.element_size()
                self.values[key] = value, length; self.bytes += size
        row.update(mel=value, mel_length=length)
        return row


class PreparedFeatures:
    """Lazy initialization inside each spawned CPU worker; no CUDA access."""
    def attach_features(self, out):
        if not self.cfg.get('worker_features', False): return out
        if not hasattr(self, '_feature_cache'):
            self._feature_cache = FeatureCache(OfficialFeatures.from_pretrained(self.cfg['bert_model_dir']),
                                               self.cfg.get('feature_cache_mib', 64))
        for row in out:
            recipe = row.get('base_augmentation', {}).get('recipe', {})
            static = row.get('role') != 'noisy' and (not recipe or recipe.get('family') == 'original')
            self._feature_cache.attach(row, static=static)
            pair = row.get('pcl')
            if pair:
                self._feature_cache.attach(pair['reference'], static=True)
                if not pair['reuse_received']: self._feature_cache.attach(pair['received'], static=True)
        return out
