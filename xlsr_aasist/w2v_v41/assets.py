"""Pinned official w2v-BERT assets; downloads occur only in explicit preparation."""
import hashlib
import json
from pathlib import Path

MODEL_ID = 'facebook/w2v-bert-2.0'


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def _files(directory):
    directory = Path(directory).expanduser().resolve()
    required = [directory/'config.json', directory/'preprocessor_config.json']
    weight = next((directory/name for name in ('model.safetensors', 'pytorch_model.bin') if (directory/name).is_file()), None)
    if weight is not None:
        required.append(weight)
    else:
        index = next((directory/name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json') if (directory/name).is_file()), None)
        if index is None:
            raise FileNotFoundError('No complete official w2v-BERT weight file in '+str(directory))
        required.append(index)
        mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
        for name in sorted(set(mapping.values())):
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Weight index escapes model directory')
            # Official Hub snapshots legitimately symlink shard files to a
            # sibling blobs store. Validate the index's lexical path, not the
            # target of a legitimate downloaded cache symlink.
            path = directory/relative
            required.append(path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('Missing official model asset: '+str(path))
    return required


def prepare_assets(model_dir=None, *, download=True, revision=None):
    if model_dir is None:
        if not download:
            raise ValueError('A local bert_model_dir is required when downloading is disabled')
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(MODEL_ID, revision=revision,
            allow_patterns=['config.json', 'preprocessor_config.json', '*.safetensors', '*.safetensors.index.json'])
    elif download:
        try:
            _files(model_dir)
        except FileNotFoundError:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(MODEL_ID, revision=revision, local_dir=str(model_dir),
                allow_patterns=['config.json', 'preprocessor_config.json', '*.safetensors', '*.safetensors.index.json'])
    files = _files(model_dir)
    config = json.loads((Path(model_dir)/'config.json').read_text(encoding='utf-8'))
    if (config.get('model_type'), config.get('hidden_size'), config.get('num_hidden_layers'),
            config.get('feature_projection_input_dim'), config.get('add_adapter', False)) != ('wav2vec2-bert', 1024, 24, 160, False):
        raise ValueError('Expected the complete official 24-layer w2v-BERT 2.0 encoder')
    processor = json.loads((Path(model_dir)/'preprocessor_config.json').read_text(encoding='utf-8'))
    expected = dict(feature_extractor_type='SeamlessM4TFeatureExtractor', feature_size=80,
        num_mel_bins=80, sampling_rate=16000, stride=2, return_attention_mask=True, padding_side='right', padding_value=1)
    if any(processor.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected the official w2v-BERT 2.0 preprocessor configuration')
    return dict(bert_model_id=MODEL_ID, bert_model_dir=str(Path(model_dir).resolve()),
        bert_revision=revision, bert_files={str(path): _sha(path) for path in files}, transformers_version='4.38.2')


def verify_assets(cfg):
    files = _files(cfg['bert_model_dir'])
    expected = cfg.get('bert_files')
    if not isinstance(expected, dict) or set(expected) != {str(p) for p in files}:
        raise ValueError('Missing or incomplete pinned w2v-BERT asset inventory')
    for path in files:
        if _sha(path) != expected[str(path)]:
            raise ValueError('Official model asset changed: '+str(path))
    return files
