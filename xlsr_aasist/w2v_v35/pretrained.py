"""Bind fresh initialization to the public generic model, never an anti-spoof checkpoint."""
import json
from pathlib import Path

from w2v_aasist.runtime import sha256

MODEL_ID = 'facebook/w2v-bert-2.0'
REVISION = '6b1c0b3a98343376f0cffd7cfbbe3ce4db45d459'
MODEL_SHA256 = 'eb890c9660ed6e3414b6812e27257b8ce5454365d5490d3ad581ea60b93be043'


def validate_directory(folder, expected_sha=MODEL_SHA256):
    folder = Path(folder).expanduser().resolve()
    paths = {name: folder/name for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')}
    if any(not p.is_file() for p in paths.values()):
        raise ValueError('Generic pretrained folder needs config, feature extractor, and model.safetensors: '+str(folder))
    if sha256(paths['model.safetensors']) != expected_sha:
        raise ValueError('Encoder weights are not the pinned generic w2v-BERT 2.0; refusing fine-tuned or unknown initialization')
    architecture = json.loads(paths['config.json'].read_text(encoding='utf-8'))
    required = dict(model_type='wav2vec2-bert', hidden_size=1024, num_hidden_layers=24,
                    feature_projection_input_dim=160, num_attention_heads=16, intermediate_size=4096,
                    add_adapter=False, position_embeddings_type='relative_key')
    if any(architecture.get(k) != v for k,v in required.items()):
        raise ValueError('Pretrained encoder architecture differs from the public 24-layer model')
    return {str(p): sha256(p) for p in paths.values()}


def prepare(cfg):
    """Reuse a verified existing download; download only three required files otherwise."""
    explicit = cfg.get('pretrained_path')
    candidate = explicit or cfg.get('pretrained_candidate_path')
    fingerprints = None
    if candidate:
        try:
            fingerprints = validate_directory(candidate)
        except ValueError:
            if explicit:
                raise
    if fingerprints is None:
        from huggingface_hub import snapshot_download
        print('V35_PRETRAINED: downloading/reusing the pinned generic encoder (2.32 GB if not cached).', flush=True)
        candidate = snapshot_download(MODEL_ID, revision=REVISION,
            allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
        fingerprints = validate_directory(candidate)
    cfg['pretrained_path'] = str(Path(candidate).resolve())
    cfg['ssl_path'] = cfg['pretrained_path']
    reference_processor = Path(cfg['reference_ssl_path'])/'preprocessor_config.json'
    if sha256(reference_processor) != sha256(Path(candidate)/'preprocessor_config.json'):
        raise ValueError('Generic and submitted model feature extractors differ; baseline comparison must use identical features')
    cfg['pretrained_fingerprints'] = fingerprints
    cfg['pretrained_origin'] = dict(model_id=MODEL_ID, revision=REVISION, weight_sha256=MODEL_SHA256,
                                    role='generic speech pretraining, not an anti-spoof checkpoint')
    print('V35_PRETRAINED_SHA256='+MODEL_SHA256, flush=True)
    return cfg


def verify(cfg):
    for path, digest in cfg.get('pretrained_fingerprints', {}).items():
        if sha256(path) != digest:
            raise ValueError('Pretrained artifact changed: '+path)
    if cfg.get('production_layout', True) and not cfg.get('pretrained_fingerprints'):
        raise ValueError('Missing verified generic initialization provenance')


def initialize(cfg):
    verify(cfg)
    import transformers
    if transformers.__version__ != '4.38.2':
        raise RuntimeError('Use transformers==4.38.2')
    from transformers import Wav2Vec2BertModel
    from w2v_v33.model import Detector
    from w2v_v3.model import HeadConfig
    model, loaded = Wav2Vec2BertModel.from_pretrained(cfg['pretrained_path'], local_files_only=True,
                                            use_safetensors=True, output_loading_info=True)
    if any(loaded.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Generic encoder did not load exactly: '+str(loaded))
    model.config.layerdrop = 0.
    model.encoder.config.layerdrop = 0.
    model.config.apply_spec_augment = False
    if cfg.get('checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return Detector(model, HeadConfig(**cfg['head_config']) if cfg.get('head_config') else None)


def reference_model(cfg):
    import torch
    from w2v_v33.model import Detector
    path = cfg['reference_checkpoint']
    if sha256(path) != cfg['reference_checkpoint_sha256']:
        raise ValueError('Submitted reference checkpoint changed')
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if state.get('kind') != 'weights' or state.get('tag') != cfg['reference_tag']:
        raise ValueError('Submitted reference tag/kind differs')
    return Detector.from_checkpoint(state, checkpointing=False)
