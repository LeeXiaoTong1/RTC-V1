"""Shape-compatible complete ASP detector for strict parent-best transfer."""
from pathlib import Path
import torch
from torch import nn
from w2v_v3183r2.model import Detector as R2Detector, RecomputedLayer
from w2v_v3183r2.common import require_finite
from .aasist import SSLAASIST
from .regions import StatisticsHead


class Detector(R2Detector):
    def __init__(self, frontend, encoder, cfg, layout_factory):
        if cfg.get('pooling') != 'asp' or not cfg.get('local_evidence'):
            raise ValueError('V3.18.8 requires overlapping-window ASP')
        count = cfg.get('checkpoint_layers', cfg['lora_layers'])
        if not isinstance(count, int) or not 0 <= count <= cfg['lora_layers']:
            raise ValueError('checkpoint_layers must be between zero and lora_layers')
        super().__init__(frontend, encoder, cfg, layout_factory)
        self.head.aasist = SSLAASIST(128)
        self.head.aasist.LL = nn.Identity()
        self.head = StatisticsHead(self.head, temporal_stats=cfg.get('temporal_stats', True))
        # Fresh means both mean and standard-deviation columns learn from the
        # beginning. Never slice a parent classifier or silently zero half of a
        # pretrained classifier. Compatible transfer loads all state afterward.
        from .pcl_state import PCLScale
        self.pcl_scale = PCLScale(dim=128,
            floor_fraction=cfg.get('pcl_scale_floor_fraction', .1),
            min_floor=cfg.get('pcl_scale_min_floor', 1e-4))
        self.fusion.checkpointing = bool(cfg.get('fusion_checkpointing', True))
        self.set_phase(cfg.get('warm_epochs', 1) == 0)

    def set_phase(self, joint):
        super().set_phase(joint)
        wrappers = [layer for layer in self.encoder.layers if isinstance(layer, RecomputedLayer)]
        count = self.cfg.get('checkpoint_layers', self.cfg['lora_layers'])
        # Keep checkpointing on the deepest selected adapter layers. This is a
        # runtime compute/memory choice; it does not change checkpoint keys.
        for index, layer in enumerate(wrappers):
            layer.enabled = bool(joint and self.cfg.get('checkpointing', True) and index >= len(wrappers)-count)

    def classify_sequences(self, sequences):
        # PCL references never enter the classification backend.
        with torch.autocast(next(self.head.parameters()).device.type, enabled=False):
            logits = self.head(sequences)
        require_finite(logits, 'window ASP logits')
        return logits

    def encode(self, waves):
        maximum = self.cfg.get('whole_max_frames', 3000)
        frames_list=[max(1,(len(wave)-400)//320+1) for wave in waves]
        self.last_encode_geometry=dict(encoded_views=len(waves),frames=frames_list,
            padded_frames=len(waves)*max(frames_list,default=0))
        for wave in waves:
            frames = (len(wave)-400)//320+1
            if frames > maximum:
                raise ValueError(f'Frontend frame limit exceeded: {frames} > {maximum} frames; '
                                 'raise whole_max_frames after checking GPU memory; no crop fallback')
        return super().encode(waves)

    def forward(self, waves):
        return self.classify_sequences(self.encode(waves))


def load_model(cfg):
    from w2v_v3184.assets import configured_spec
    from w2v_v3184.runtime import require_version
    item = configured_spec(cfg)
    require_version('fairseq2', '0.6.0')
    import omnilingual_asr  # noqa: F401; register the pinned architecture.
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    hub = get_wav2vec2_model_hub()
    arch = hub.get_arch_config(item['arch'])
    ec = arch.encoder_config
    if (ec.model_dim, ec.num_encoder_layers) != (item['encoder_dim'], item['encoder_layers']):
        raise ValueError('Installed Omni architecture changed')
    device = torch.device(cfg['device'])
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    # This loads the pretrained SSL backbone, never randomly reinitializes it.
    ssl = hub.load_custom_model(Path(cfg['omni_checkpoint']), arch, device=device, dtype=dtype,
                               mmap=True, restrict=True)
    if len(ssl.encoder.layers) != item['encoder_layers']:
        raise ValueError('Wrong encoder depth')
    model = Detector(ssl.encoder_frontend, ssl.encoder, cfg, BatchLayout)
    del ssl
    return model.to(device)


def optimizer_for(model, cfg):
    depth = len(model.encoder.layers)
    adapters = {'early': [], 'middle': [], 'late': []}
    for name, parameter in model.named_parameters():
        if name.endswith(('.lora_a', '.lora_b')):
            parts = name.split('.')
            if parts[:2] != ['encoder', 'layers']:
                raise ValueError('Unexpected LoRA parameter location: ' + name)
            bucket = ('early', 'middle', 'late')[min(2, int(parts[2]) * 3 // depth)]
            adapters[bucket].append(parameter)
    rates = (cfg.get('lora_early_lr', cfg['lora_lr'] * .25),
             cfg.get('lora_middle_lr', cfg['lora_lr'] * .5), cfg['lora_lr'])
    parts = [('lora_' + name, adapters[name], rate, 'lora')
             for name, rate in zip(('early', 'middle', 'late'), rates)]
    parts += [('aasist', [p for n, p in model.head.aasist.named_parameters()
                         if not n.startswith('out_layer.')], cfg['head_lr'], 'head'),
              ('classifier', list(model.head.aasist.out_layer.parameters()), cfg['classifier_lr'], 'head'),
              ('evidence', list(model.head.contribution.parameters()), cfg['evidence_lr'], 'head'),
              ('pooling', list(model.head.attention_delta.parameters()), cfg['pool_lr'], 'head'),
              ('fusion', [p for n, p in model.fusion.named_parameters() if n != 'layer_logits'], cfg['fusion_lr'], 'fusion'),
              ('layer_mix', [model.fusion.layer_logits], cfg['layer_mix_lr'], 'fusion')]
    groups = []
    for name, parameters, rate, role in parts:
        for decay in (False, True):
            selected = [p for p in parameters if (p.ndim >= 2) == decay]
            if selected:
                groups.append(dict(name=name, role=role, params=selected, lr=rate, initial_lr=rate,
                                   weight_decay=cfg['weight_decay'] if decay else 0.))
    actual = [p for group in groups for p in group['params']]
    expected = {id(p) for n, p in model.named_parameters()
                if p.requires_grad or n.endswith(('.lora_a', '.lora_b'))}
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != expected:
        raise ValueError('Optimizer inventory mismatch')
    return torch.optim.AdamW(groups)
