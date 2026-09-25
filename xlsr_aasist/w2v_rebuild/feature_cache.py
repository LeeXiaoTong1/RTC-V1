"""Lossless content-addressed cache of fixed official input features, never SSL states."""
import errno
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from .storage import available_bytes

# Preserve the original Linux cache namespace: only the write policy changed.
# Bump this identity if the cached values/layout change. Extractor, collator,
# dependency versions and input content continue to be independently hashed.
FEATURE_FORMAT_HASH = '5a860e08445450b6345679282ac0a956911471bdcae7bc9d2ca1365b9534f116'


class FeatureCache:
    def __init__(self, root, model_directory, extractor):
        config = Path(model_directory) / 'preprocessor_config.json'
        identity = {
            'schema': 'w2v_official_fixed_pcm_v1', 'cut': 64600, 'sr': 16000,
            'config': hashlib.sha256(config.read_bytes()).hexdigest(),
            'extractor': hashlib.sha256(inspect.getsource(type(extractor)).encode()).hexdigest(),
            'implementation': FEATURE_FORMAT_HASH,
            'collator': hashlib.sha256(Path(__file__).with_name('data.py').read_bytes()).hexdigest(),
            'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'transformers')},
        }
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.root = Path(root) / namespace
        self.min_free_bytes = int(os.environ.get('W2V_FEATURE_CACHE_MIN_FREE_BYTES', '0'))
        if self.min_free_bytes < 0:
            raise ValueError('Feature-cache free-space floor must be nonnegative')
        self.warned = False

    def skip_write(self):
        if not self.warned:
            print('Feature cache: skip new disk writes; recompute missing inputs with unchanged values.', flush=True)
            self.warned = True

    def publish(self, path, value):
        # Extra slack covers concurrent DataLoader workers (each entry is ~0.13 MB).
        if available_bytes(path.parent) < self.min_free_bytes + value.nbytes + 16*1024**2:
            self.skip_write()
            return
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
            with os.fdopen(fd, 'wb') as stream:
                np.save(stream, value, allow_pickle=False)
            os.replace(tmp, path)
        except OSError as exc:
            if exc.errno not in (errno.ENOSPC, getattr(errno, 'EDQUOT', 122)):
                raise
            self.skip_write()
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)

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
                self.publish(path, value)
                values[path] = torch.from_numpy(value)
        features = torch.stack([values[p] for p in paths])
        return features, torch.ones(features.shape[:2], dtype=torch.long)
