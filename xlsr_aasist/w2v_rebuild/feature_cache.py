"""Lossless content-addressed cache of fixed official input features, never SSL states."""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch


class FeatureCache:
    def __init__(self, root, model_directory, extractor):
        config = Path(model_directory) / 'preprocessor_config.json'
        identity = {
            'schema': 'w2v_official_fixed_pcm_v1', 'cut': 64600, 'sr': 16000,
            'config': hashlib.sha256(config.read_bytes()).hexdigest(),
            'extractor': hashlib.sha256(inspect.getsource(type(extractor)).encode()).hexdigest(),
            'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'collator': hashlib.sha256(Path(__file__).with_name('data.py').read_bytes()).hexdigest(),
            'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'transformers')},
        }
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.root = Path(root) / namespace

    def path(self, waveform):
        raw = np.asarray(waveform.numpy(), dtype='<f4').tobytes()
        key = hashlib.sha256(raw).hexdigest()
        return self.root / key[:2] / (key + '.npy')

    @staticmethod
    def _validate(value):
        if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != 160 or value.shape[0] < 12:
            raise ValueError('Invalid feature-cache shape/dtype; remove the corrupt entry')
        if not np.isfinite(value).all():
            raise ValueError('Non-finite cached features')
        return torch.from_numpy(value)

    def transform(self, waves, extract):
        paths = [self.path(w.float()) for w in waves]
        values, missing = {}, {}
        for path, wave in zip(paths, waves):
            if path in values or path in missing:
                continue
            if path.is_file():
                values[path] = self._validate(np.load(path, allow_pickle=False))
            else:
                missing[path] = wave
        if missing:
            features, mask = extract(list(missing.values()))
            if not bool(mask.bool().all()):
                raise ValueError('Only trimmed all-valid fixed features may be cached')
            for path, feature in zip(missing, features):
                value = np.array(feature.numpy(), dtype=np.float32, copy=True)
                self._validate(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                # Workers may discover the same content; publish only complete files.
                fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        np.save(stream, value, allow_pickle=False)
                    os.replace(tmp, path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                values[path] = torch.from_numpy(value)
        features = torch.stack([values[p] for p in paths])
        return features, torch.ones(features.shape[:2], dtype=torch.long)
