"""Pinned all-layer Omni reader with whole-utterance native AASIST classification."""
from pathlib import Path
import torch
from w2v_v3183r2.model import Detector as R2Detector, LoRALinear, RecomputedLayer
from w2v_v3183r2.common import require_finite
from .regions import WholeUtteranceHead


class Detector(R2Detector):
    def __init__(self, frontend, encoder, cfg, layout_factory):
        # Reuse the pinned encoder/LoRA/reader constructor. The temporary old
        # head is discarded completely: no contribution/ASP parameters remain.
        constructor_cfg = dict(cfg, local_evidence=False, window=128, hop=64,
                               window_batch=cfg.get('whole_batch', 8))
        super().__init__(frontend, encoder, constructor_cfg, layout_factory)
        self.cfg = cfg
        self.head = WholeUtteranceHead(self.head.aasist, cfg.get('whole_max_frames', 3000),
            cfg.get('whole_graph_budget', 1_000_000), cfg.get('whole_batch', 8))
        self.set_phase(True)

    def set_phase(self, joint):
        self.joint = joint
        for module in self.encoder.modules():
            if isinstance(module, LoRALinear):
                module.enabled = True
                module.lora_a.requires_grad_(joint)
                module.lora_b.requires_grad_(joint)
            if isinstance(module, RecomputedLayer):
                module.enabled = joint and self.cfg['checkpointing']
        self.train(self.training)

    def classify_sequences(self, sequences):
        with torch.autocast(next(self.head.parameters()).device.type, enabled=False):
            logits = self.head(sequences)
        require_finite(logits, 'whole-utterance logits')
        return logits

    def encode(self, waves):
        maximum = self.cfg.get('whole_max_frames', 3000)
        for wave in waves:
            frames = (len(wave) - 400) // 320 + 1
            if frames > maximum:
                raise ValueError(f'Whole-utterance frontend limit exceeded: {frames} > {maximum} frames; '
                                 'raise whole_max_frames after checking GPU memory; no crop fallback')
        return super().encode(waves)

    def forward(self, waves):
        return self.classify_sequences(self.encode(waves))


def load_model(cfg):
    from w2v_v3184.assets import configured_spec
    from w2v_v3184.runtime import require_version
    item = configured_spec(cfg)
    require_version('fairseq2', '0.6.0')
    import omnilingual_asr  # noqa: F401; registers the pinned architecture.
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    hub = get_wav2vec2_model_hub()
    arch = hub.get_arch_config(item['arch'])
    ec = arch.encoder_config
    if (ec.model_dim, ec.num_encoder_layers) != (item['encoder_dim'], item['encoder_layers']):
        raise ValueError('Installed Omni architecture changed')
    device = torch.device(cfg['device'])
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    ssl = hub.load_custom_model(Path(cfg['omni_checkpoint']), arch, device=device, dtype=dtype,
                               mmap=True, restrict=True)
    if len(ssl.encoder.layers) != item['encoder_layers']:
        raise ValueError('Wrong encoder depth')
    model = Detector(ssl.encoder_frontend, ssl.encoder, cfg, BatchLayout)
    del ssl
    return model.to(device)


def optimizer_for(model, cfg):
    parts = [
        ('lora', [p for n, p in model.named_parameters() if n.endswith(('.lora_a', '.lora_b'))], cfg['lora_lr']),
        ('aasist', [p for n, p in model.head.aasist.named_parameters() if not n.startswith('out_layer.')], cfg['head_lr']),
        ('classifier', list(model.head.aasist.out_layer.parameters()), cfg['classifier_lr']),
        ('fusion', [p for n, p in model.fusion.named_parameters() if n != 'layer_logits'], cfg['fusion_lr']),
        ('layer_mix', [model.fusion.layer_logits], cfg['layer_mix_lr']),
    ]
    groups = []
    for name, parameters, rate in parts:
        for decay in (False, True):
            selected = [p for p in parameters if (p.ndim >= 2) == decay]
            if selected:
                groups.append(dict(name=name, params=selected, lr=rate, initial_lr=rate,
                                   weight_decay=cfg['weight_decay'] if decay else 0.))
    actual = [p for group in groups for p in group['params']]
    expected = {id(p) for n, p in model.named_parameters()
                if p.requires_grad or n.endswith(('.lora_a', '.lora_b'))}
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != expected:
        raise ValueError('Optimizer inventory mismatch')
    return torch.optim.AdamW(groups)
