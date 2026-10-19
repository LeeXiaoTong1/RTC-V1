"""Keep the complete inherited detector; adapt its last attention layers."""
from pathlib import Path
import torch
from w2v_v31810.model import Detector as BaseDetector
from w2v_v3183r2.model import LoRALinear, RecomputedLayer
from .adapters import attach


class Detector(BaseDetector):
    def __init__(self, frontend, encoder, cfg, layout_factory):
        super().__init__(frontend, encoder, cfg, layout_factory)
        attach(self)
        self.set_phase(True)

    def set_phase(self, joint):
        super().set_phase(False)
        if hasattr(self, 'adapters'):
            self.adapters.requires_grad_(True)
            for i, layer in enumerate(self.encoder.layers):
                if isinstance(layer, RecomputedLayer):
                    layer.enabled = bool(str(i) in self.adapters and self.cfg.get('checkpointing', True))
        # Existing LoRA remains enabled, deterministic and frozen in every epoch.
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, LoRALinear):
                module.drop.eval()
        return self


def load_model(cfg):
    from w2v_v3184.assets import configured_spec
    from w2v_v3184.runtime import require_version
    spec = configured_spec(cfg)
    require_version('fairseq2', '0.6.0')
    import omnilingual_asr  # registers the pinned official architecture
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    hub = get_wav2vec2_model_hub()
    arch = hub.get_arch_config(spec['arch'])
    ec = arch.encoder_config
    if (ec.model_dim, ec.num_encoder_layers) != (spec['encoder_dim'], spec['encoder_layers']):
        raise ValueError('Official Omni dimensions changed')
    device = torch.device(cfg['device'])
    ssl = hub.load_custom_model(Path(cfg['omni_checkpoint']), arch, device=device,
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, mmap=True, restrict=True)
    model = Detector(ssl.encoder_frontend, ssl.encoder, cfg, BatchLayout)
    return model.to(device)


def optimizer_for(model, cfg):
    parts = [('adapter', list(model.adapters.parameters()), cfg['adapter_lr']),
        ('aasist', [p for n, p in model.head.aasist.named_parameters() if not n.startswith('out_layer.')], cfg['head_lr']),
        ('classifier', list(model.head.aasist.out_layer.parameters()), cfg['classifier_lr']),
        ('evidence', list(model.head.contribution.parameters()), cfg['evidence_lr']),
        ('pooling', list(model.head.attention_delta.parameters()), cfg['pool_lr']),
        ('fusion', [p for n, p in model.fusion.named_parameters() if n != 'layer_logits'], cfg['fusion_lr']),
        ('layer_mix', [model.fusion.layer_logits], cfg['layer_mix_lr'])]
    groups = []
    for name, parameters, rate in parts:
        for decay in (False, True):
            chosen = [p for p in parameters if p.requires_grad and (p.ndim >= 2) == decay]
            if chosen:
                groups.append(dict(name=name, role=name, params=chosen, lr=rate, initial_lr=rate,
                    weight_decay=cfg['weight_decay'] if decay else 0.))
    actual = [id(p) for g in groups for p in g['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('V4 optimizer misses or duplicates trainable parameters')
    return torch.optim.AdamW(groups)


def schedule(optimizer, cfg, epoch, step, steps):
    import math
    update = epoch * steps + step
    total = cfg['epochs'] * steps
    factor = cfg['lr_floor'] + (1-cfg['lr_floor']) * .5 * (1+math.cos(math.pi*update/max(1, total-1)))
    warm = min(1., (update+1)/max(1, cfg.get('lr_warmup_updates',200)))
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * factor * warm
