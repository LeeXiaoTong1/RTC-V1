"""Frozen generic LID teacher; neither task labels nor file names enter this model.

Architecture: the official immutable VoxLingua107 hyperparams.yaml.  We do not
execute that YAML, instantiate its classifier, or download training/audio data.
The teacher is used on official Train views only, never by submission inference.
"""
from contextlib import contextmanager
from importlib.metadata import version
import math
from pathlib import Path

import numpy as np
import torch

from w2v_aasist.runtime import atomic_json, sha256

REPO_ID = 'speechbrain/lang-id-voxlingua107-ecapa'
REVISION = '0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9'
WEIGHT_SHA256 = 'ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a'
WEIGHT_BYTES = 84474355
SPEECHBRAIN_VERSION = '1.0.2'
DIM = 256
SAMPLE_RATE = 16000
MIN_SAMPLES = 1600
ASSET_FORMAT = 'rtc_v37_frozen_lid_assets_v1'
MODE = 'ecapa-fbank60-fp32-eval-balanced-full-wave-segments-unit-duration-mean-unit-v1'
ASSET_FILES = ('embedding_model.ckpt', 'hyperparams.yaml', 'README.md')


def ensure_language_assets(cfg):
    """Reuse HF's single cached copy; download only three explicitly named files.

    An immutable revision and the published LFS SHA256/size pin the checkpoint.
    Local cache hits are tried first even online.  The manifest records SHA256
    for every small provenance file, but no YAML/Python from HF is executed.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    from .language_bundle import imported_paths

    if cfg.get('language_repo_id', REPO_ID) != REPO_ID or cfg.get('language_revision', REVISION) != REVISION:
        raise ValueError('V3.7 requires the pinned official LID repository/revision')
    kwargs = dict(repo_id=REPO_ID, revision=REVISION)
    if cfg.get('language_cache_dir'):
        kwargs['cache_dir'] = str(Path(cfg['language_cache_dir']).expanduser().resolve())
    imported = imported_paths(kwargs.get('cache_dir'))
    files = {}
    for name in ASSET_FILES:
        if imported is not None:
            path = imported[name]
        else:
            try:
                path = hf_hub_download(filename=name, local_files_only=True, **kwargs)
            except LocalEntryNotFoundError:
                if cfg.get('language_offline', False):
                    raise RuntimeError('Pinned LID asset missing from offline cache: ' + name +
                        '. Import the verified ZIP with python -m w2v_v37.language '
                        '--cache-dir models/v37_language_teacher --import-bundle /path/to/v37_language_teacher.zip --offline') from None
                try:
                    path = hf_hub_download(filename=name, local_files_only=False, **kwargs)
                except Exception as exc:
                    raise RuntimeError('Cannot download pinned V3.7 language teacher file '+name+
                        '. This happened before model training. Import the verified offline ZIP with '
                        'python -m w2v_v37.language --cache-dir models/v37_language_teacher '
                        '--import-bundle /path/to/v37_language_teacher.zip --offline. '
                        'Use your configured language cache path if different. Original error: '+str(exc)) from exc
        path = Path(path).resolve()
        files[name] = dict(path=str(path), sha256=sha256(path), size=path.stat().st_size)
    assets = dict(format=ASSET_FORMAT, repo_id=REPO_ID, revision=REVISION,
                  license='apache-2.0', speechbrain_version=SPEECHBRAIN_VERSION,
                  files=files, embedding_dim=DIM, sample_rate=SAMPLE_RATE,
                  encoding_policy=encoding_policy(cfg))
    verify_language_assets(assets)
    if cfg.get('language_assets_manifest'):
        atomic_json(Path(cfg['language_assets_manifest']), assets)
    return assets


def verify_language_assets(assets):
    """Check provenance and actual bytes before any checkpoint is deserialized."""
    if (assets.get('format') != ASSET_FORMAT or assets.get('repo_id') != REPO_ID
            or assets.get('revision') != REVISION or assets.get('speechbrain_version') != SPEECHBRAIN_VERSION
            or assets.get('embedding_dim') != DIM or assets.get('sample_rate') != SAMPLE_RATE
            or set(assets.get('files', {})) != set(ASSET_FILES)):
        raise ValueError('Unknown or incomplete frozen LID asset identity')
    policy = assets.get('encoding_policy', {})
    seconds = policy.get('segment_samples', 0)/SAMPLE_RATE
    if policy != encoding_policy(dict(language_segment_seconds=seconds)):
        raise ValueError('Unknown frozen LID encoding policy')
    for name, info in assets['files'].items():
        path = Path(info['path'])
        if not path.is_file() or path.stat().st_size != info['size'] or sha256(path) != info['sha256']:
            raise ValueError('Frozen LID asset changed: ' + name)
    weights = assets['files']['embedding_model.ckpt']
    if weights['sha256'] != WEIGHT_SHA256 or weights['size'] != WEIGHT_BYTES:
        raise ValueError('LID checkpoint does not match the official pinned SHA256/size')
    return assets


def assets_identity(assets):
    """Portable provenance for the small fitted patch; no local cache paths."""
    return {**{k: assets[k] for k in ('format', 'repo_id', 'revision', 'license',
                                     'speechbrain_version', 'embedding_dim', 'sample_rate', 'encoding_policy')},
            'files': {name: {k: info[k] for k in ('sha256', 'size')}
                      for name, info in assets['files'].items()}}


def encoding_policy(cfg):
    seconds = float(cfg.get('language_segment_seconds', 8.0))
    if not math.isfinite(seconds) or not .1 <= seconds <= 8.0:
        raise ValueError('LID segment duration must be between 0.1 and 8 seconds')
    return dict(mode=MODE, sample_rate=SAMPLE_RATE, segment_samples=round(seconds*SAMPLE_RATE),
                short_wave_repeat_min_samples=MIN_SAMPLES, embedding_dim=DIM,
                partition='balanced-contiguous-nonoverlap-cover-every-sample',
                pooling='L2 each segment; original-duration weighted mean; L2 result',
                frontend='Fbank60; sentence mean normalization; no std normalization',
                role='frozen official-Train-only teacher; excluded from submission')


def language_identity(cfg, assets, device=None):
    return dict(assets=assets_identity(assets), policy=encoding_policy(cfg),
                runtime={name: version(name) for name in
                         ('speechbrain', 'torch', 'torchaudio', 'numpy', 'scipy')},
                device_type=torch.device(device or cfg.get('device', 'cpu')).type)


def _components():
    """Construct documented classes directly, without remote hyperparameter code."""
    if version('speechbrain') != SPEECHBRAIN_VERSION:
        raise RuntimeError('Frozen LID extraction requires speechbrain==' + SPEECHBRAIN_VERSION)
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization
    features = Fbank(n_mels=60, left_frames=0, right_frames=0, deltas=False,
                     sample_rate=SAMPLE_RATE)
    normalizer = InputNormalization(norm_type='sentence', std_norm=False)
    model = ECAPA_TDNN(input_size=60, channels=[1024, 1024, 1024, 1024, 3072],
                      kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                      attention_channels=128, lin_neurons=DIM)
    return features, normalizer, model


@contextmanager
def _fp32(device):
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class LanguageEncoder:
    """Exact-length segments bound activation memory independently of file length."""
    def __init__(self, cfg, device=None):
        # Passing the asset manifest is convenient for offline smoke verification.
        if cfg.get('format') == ASSET_FORMAT:
            self.assets, cfg = cfg, {'language_segment_seconds': cfg['encoding_policy']['segment_samples']/SAMPLE_RATE}
        else:
            self.assets = cfg.get('language_assets') or ensure_language_assets(cfg)
        verify_language_assets(self.assets)
        self.device = torch.device(device or cfg.get('device', 'cpu'))
        self.policy = encoding_policy(cfg)
        if self.policy != self.assets['encoding_policy']:
            raise ValueError('Requested LID encoding policy differs from asset manifest')
        # SpeechBrain's package import may enable its allow_tf32 quirk globally.
        # Restore caller settings; the teacher forward explicitly disables TF32.
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            self.features, self.normalizer, self.model = _components()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
        # weights_only=True accepts this official tensor-only legacy torch file;
        # no retry with unrestricted unpickling is permitted.
        state = torch.load(self.assets['files']['embedding_model.ckpt']['path'],
                           map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        for module in (self.features, self.normalizer, self.model):
            # InputNormalization.to accepts only device (unlike nn.Module.to).
            module.float().to(self.device).eval().requires_grad_(False)
        self.identity = language_identity(cfg, self.assets, self.device)

    def _segment(self, samples):
        if len(samples) < MIN_SAMPLES:
            samples = np.tile(samples, math.ceil(MIN_SAMPLES/len(samples)))[:MIN_SAMPLES]
        wave = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0).to(self.device)
        lens = torch.ones(1, dtype=torch.float32, device=self.device)
        embedding = self.model(self.normalizer(self.features(wave), lens), lens).reshape(-1)
        if embedding.shape != (DIM,) or not bool(torch.isfinite(embedding).all()):
            raise FloatingPointError('Nonfinite or malformed LID segment embedding')
        norm = torch.linalg.vector_norm(embedding)
        if not bool(norm > 1e-12):
            raise FloatingPointError('Zero LID segment embedding')
        return (embedding/norm).cpu().numpy()

    def encode_waveforms(self, waveforms):
        """16 kHz mono arrays in, unit FP32 [N,256] out; no metadata input."""
        result = np.empty((len(waveforms), DIM), dtype=np.float32)
        with _fp32(self.device):
            for i, value in enumerate(waveforms):
                wave = np.asarray(value, dtype=np.float32)
                if wave.ndim != 1 or len(wave) == 0 or not np.isfinite(wave).all():
                    raise ValueError('LID requires nonempty finite mono waveforms at 16000 Hz')
                count = math.ceil(len(wave)/self.policy['segment_samples'])
                pooled = np.zeros(DIM, dtype=np.float64)
                for segment in range(count):
                    start, stop = segment*len(wave)//count, (segment+1)*len(wave)//count
                    pooled += self._segment(wave[start:stop]).astype(np.float64)*(stop-start)
                norm = np.linalg.norm(pooled)
                if not np.isfinite(norm) or norm <= 1e-12:
                    raise FloatingPointError('Nonfinite or zero pooled LID embedding')
                result[i] = pooled/norm
        return result


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Prefetch/check the pinned V3.7 Train-only LID teacher')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--segment-seconds', type=float, default=8.)
    bundle = parser.add_mutually_exclusive_group()
    bundle.add_argument('--import-bundle', help='Import a SHA256-pinned public teacher ZIP without network access')
    bundle.add_argument('--export-bundle', help='Package the existing public teacher files for offline transfer')
    args = parser.parse_args(argv)
    cfg = dict(language_cache_dir=args.cache_dir, language_assets_manifest=args.manifest,
               language_offline=args.offline, language_segment_seconds=args.segment_seconds, device=args.device)
    if args.import_bundle:
        from .language_bundle import import_bundle
        import_bundle(args.import_bundle, args.cache_dir)
        cfg['language_offline'] = True
    if args.export_bundle:
        from .language_bundle import export_bundle
        export_bundle(args.cache_dir, args.export_bundle)
    assets = ensure_language_assets(cfg)
    print(json.dumps(assets, sort_keys=True, indent=2), flush=True)
    if args.smoke:
        encoder = LanguageEncoder(assets, device=args.device)
        wave = np.random.default_rng(37).normal(0, .02, SAMPLE_RATE).astype(np.float32)
        first, second = encoder.encode_waveforms([wave]), encoder.encode_waveforms([wave])
        np.testing.assert_allclose(first, second, rtol=0, atol=0)
        print(f'V37_LID_SMOKE_OK shape={first.shape} norm={np.linalg.norm(first):.8f}', flush=True)


if __name__ == '__main__':
    main()
