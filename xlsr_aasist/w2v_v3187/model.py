"""Official Omni plus a fresh complete ASP detector, with selective recompute."""
from pathlib import Path
import torch
from w2v_v3183r2.model import Detector as R2Detector, RecomputedLayer
from w2v_v3183r2.common import require_finite
from w2v_v3184_aug5.model import optimizer_for as asp_optimizer_for
from .regions import StatisticsHead


class Detector(R2Detector):
    def __init__(self, frontend, encoder, cfg, layout_factory):
        if cfg.get('pooling') != 'asp' or not cfg.get('local_evidence'):
            raise ValueError('V3.18.7 requires overlapping-window ASP')
        count = cfg.get('checkpoint_layers', min(8, cfg['lora_layers']))
        if not isinstance(count, int) or not 0 <= count <= cfg['lora_layers']:
            raise ValueError('checkpoint_layers must be between zero and lora_layers')
        super().__init__(frontend, encoder, cfg, layout_factory)
        self.head = StatisticsHead(self.head)
        # Fresh means both mean and standard-deviation columns learn from the
        # beginning. Never slice a parent classifier or silently zero half of a
        # pretrained classifier. Compatible transfer loads all state afterward.
        self.head.aasist.out_layer.reset_parameters()
        self.fusion.checkpointing = bool(cfg.get('fusion_checkpointing', True))
        self.set_phase(cfg.get('warm_epochs', 1) == 0)

    def set_phase(self, joint):
        super().set_phase(joint)
        wrappers = [layer for layer in self.encoder.layers if isinstance(layer, RecomputedLayer)]
        count = self.cfg.get('checkpoint_layers', min(8, self.cfg['lora_layers']))
        # Keep checkpointing on the deepest selected adapter layers. This is a
        # runtime compute/memory choice; it does not change checkpoint keys.
        for index, layer in enumerate(wrappers):
            layer.enabled = bool(joint and self.cfg.get('checkpointing', True) and index >= len(wrappers)-count)

    def classify_sequences(self, sequences):
        # PCL references never reach this method, the BN, or the classifier.
        with torch.autocast(next(self.head.parameters()).device.type, enabled=False):
            logits = self.head(sequences)
        require_finite(logits, 'window ASP logits')
        return logits

    def encode(self, waves):
        maximum = self.cfg.get('whole_max_frames', 3000)
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
    return asp_optimizer_for(model, cfg)
