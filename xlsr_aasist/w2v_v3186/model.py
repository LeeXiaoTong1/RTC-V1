"""Pinned Omni reader and inherited overlapping-window attention statistics head."""
from pathlib import Path
import torch
from w2v_v3184_aug5.model import Detector as ASPDetector, optimizer_for as asp_optimizer_for
from w2v_v3183r2.common import require_finite


class Detector(ASPDetector):
    def __init__(self, frontend, encoder, cfg, layout_factory):
        # Reuse the exact parent architecture and all 320 classifier columns.
        # Window geometry must come from the validated parent configuration.
        if cfg.get('pooling') != 'asp' or not cfg.get('local_evidence'):
            raise ValueError('V3.18.6 requires inherited overlapping-window ASP')
        super().__init__(frontend, encoder, cfg, layout_factory)
        self.set_phase(True)

    def classify_sequences(self, sequences):
        # Only main CE views reach this method. Auxiliary PCL references share
        # encode(), but do not enter AASIST, its BN, or the classifier.
        with torch.autocast(next(self.head.parameters()).device.type, enabled=False):
            logits = self.head(sequences)
        require_finite(logits, 'window ASP logits')
        return logits

    def encode(self, waves):
        # Keep the frontend memory guard; graph work is now bounded by windows.
        maximum = self.cfg.get('whole_max_frames', 3000)
        for wave in waves:
            frames = (len(wave) - 400) // 320 + 1
            if frames > maximum:
                raise ValueError(f'Frontend frame limit exceeded: {frames} > {maximum} frames; '
                                 'raise whole_max_frames after checking GPU memory; no crop fallback')
        return super().encode(waves)

    def forward(self, waves):
        # Encode each supplied waveform once, then slice the fused frame tensor.
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
    # Includes contribution and attention_delta, with exhaustive inventory and
    # duplicate checks maintained by the parent's ASP optimizer.
    return asp_optimizer_for(model, cfg)
