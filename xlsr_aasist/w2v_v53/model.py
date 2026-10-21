"""Single-pass Omni with sample-conditioned mixing and the inherited AASIST."""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from w2v_v3191.model import Detector as ParentDetector, adapter_inventory
from w2v_v3183r2.common import require_finite
from .fusion import AdaptiveLayerFusion


def parent_anchor(model, values):
    """Upgrade only an old parent, never loosen V5.3 resume/export validation.

    Use deterministic INITIAL conditioner tensors, even if this model has
    already trained. Averaging against live trained tensors would not be a
    zero-output parent anchor.
    """
    target = {k: v for k,v in model.state_dict().items()
              if k.startswith(('head.', 'fusion.', 'pcl_scale.')) or k.endswith(('.lora_a', '.lora_b'))}
    additions = {'fusion.conditioner.'+k: v.clone()
                 for k,v in model.fusion._initial_conditioner.items()}
    if set(values) != set(target) - set(additions):
        raise ValueError('Parent detector keys differ beyond the new conditioner')
    result = dict(values, **additions)
    for name, value in result.items():
        if value.shape != target[name].shape or value.dtype != target[name].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid parent tensor: ' + name)
    return result


def optimizer_for(model, cfg):
    """Preserve V5.2 rates/groups while excluding the static control's frozen gate."""
    model.set_phase(True)
    depth = len(model.encoder.layers)
    adapters = {'early': [], 'middle': [], 'late': []}
    for name, parameter in model.named_parameters():
        if name.endswith(('.lora_a', '.lora_b')):
            parts = name.split('.')
            if parts[:2] != ['encoder', 'layers']:
                raise ValueError('Unexpected LoRA parameter location: ' + name)
            bucket = ('early', 'middle', 'late')[min(2, int(parts[2])*3//depth)]
            adapters[bucket].append(parameter)
    rates = (cfg.get('lora_early_lr', cfg['lora_lr']*.25),
             cfg.get('lora_middle_lr', cfg['lora_lr']*.5), cfg['lora_lr'])
    parts = [('lora_'+name, adapters[name], rate, 'lora')
             for name,rate in zip(('early', 'middle', 'late'), rates)]
    parts += [('aasist', [p for n,p in model.head.aasist.named_parameters() if not n.startswith('out_layer.')], cfg['head_lr'], 'head'),
              ('classifier', list(model.head.aasist.out_layer.parameters()), cfg['classifier_lr'], 'head'),
              ('evidence', list(model.head.contribution.parameters()), cfg['evidence_lr'], 'head'),
              ('pooling', list(model.head.attention_delta.parameters()), cfg['pool_lr'], 'head'),
              ('fusion', [p for n,p in model.fusion.named_parameters() if n != 'layer_logits'], cfg['fusion_lr'], 'fusion'),
              ('layer_mix', [model.fusion.layer_logits], cfg['layer_mix_lr'], 'fusion')]
    groups = []
    for name, parameters, rate, role in parts:
        for decay in (False, True):
            active = [p for p in parameters if p.requires_grad and (p.ndim >= 2) == decay]
            if active:
                groups.append(dict(name=name, role=role, params=active, lr=rate, initial_lr=rate,
                                   weight_decay=cfg['weight_decay'] if decay else 0.))
    actual = [p for group in groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(expected) or {id(p) for p in actual} != expected:
        raise ValueError('Optimizer must contain exactly the suffix, fusion and backend')
    return torch.optim.AdamW(groups)


class Detector(ParentDetector):
    def __init__(self, frontend, encoder, cfg, layout_factory):
        super().__init__(frontend, encoder, cfg, layout_factory)
        self.fusion = AdaptiveLayerFusion(self.fusion, width=cfg['adaptive_fusion_width'],
            mode=cfg['adaptive_fusion_mode'], seed=cfg['adaptive_fusion_seed'])

    def encode(self, waves):
        maximum = self.cfg.get('whole_max_frames', 3000)
        frames = [max(1, (len(w)-400)//320+1) for w in waves]
        if not waves or max(frames) > maximum:
            raise ValueError('Empty batch or full recording exceeds Omni frame limit; no crop fallback')
        self.last_encode_geometry = dict(encoded_views=len(waves), frames=frames,
            padded_frames=len(waves)*max(frames))
        parameter = next(self.frontend.parameters())
        device, dtype = parameter.device, parameter.dtype
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with amp:
            groups, sequences = {}, [None]*len(waves)
            for i,w in enumerate(waves):
                groups.setdefault(len(w), []).append(i)
            with torch.no_grad():
                for n,indices in groups.items():
                    if n < 3920:
                        raise ValueError('Waveform must have >=12 Omni frames; extend short audio explicitly')
                    x = torch.stack([torch.as_tensor(waves[i], device=device, dtype=torch.float32) for i in indices])
                    require_finite(x, 'waveform')
                    x = F.layer_norm(x, (n,)).to(dtype)
                    y,layout = self.frontend(x, self.layout_factory(x.shape, seq_lens=[n]*len(indices), device=device))
                    for j,i in enumerate(indices):
                        length = int(layout.seq_lens[j])
                        if length != (n-400)//320+1:
                            raise ValueError('Omni frame geometry changed')
                        sequences[i] = y[j,:length]
            lengths = [len(s) for s in sequences]
            padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            layout = self.layout_factory(padded.shape[:2], seq_lens=lengths, device=device)
            # Conditions are available BEFORE the Transformer. All layer
            # contributions continue to stream; no second encoder or LxTxD cache.
            weights = self.fusion.weights(padded, lengths)
            mixed, seen, handles = None, set(), []
            def capture(index):
                def hook(module, inputs, output):
                    nonlocal mixed
                    if index in seen:
                        raise RuntimeError('Encoder layer reader executed twice in one forward')
                    seen.add(index)
                    value = self.fusion.project_layer(output, index, weights)
                    mixed = value if mixed is None else mixed + value
                return hook
            try:
                for index in self.fusion.layers[:-1]:
                    handles.append(self.encoder.layers[index-1].register_forward_hook(capture(index)))
                final = self.encoder(padded, layout)
            finally:
                for handle in handles:
                    handle.remove()
            if seen != set(self.fusion.layers[:-1]):
                raise RuntimeError('An encoder layer did not reach the fusion reader')
            last = self.fusion.project_layer(final, self.fusion.layers[-1], weights)
            fused = last if mixed is None else mixed + last
            require_finite(fused, 'adaptive fused features')
            return [fused[i,:n] for i,n in enumerate(lengths)]


def load_model(cfg):
    from w2v_v3184.assets import configured_spec
    from w2v_v3184.runtime import require_version
    require_version('fairseq2', '0.6.0')
    import omnilingual_asr  # noqa: F401
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    item = configured_spec(cfg)
    hub = get_wav2vec2_model_hub()
    arch = hub.get_arch_config(item['arch'])
    ec = arch.encoder_config
    if (ec.model_dim, ec.num_encoder_layers) != (cfg['encoder_dim'], cfg['encoder_layers']):
        raise ValueError('Pinned Omni geometry changed')
    device = torch.device(cfg['device'])
    ssl = hub.load_custom_model(Path(cfg['omni_checkpoint']), arch, device=device,
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32, mmap=True, restrict=True)
    model = Detector(ssl.encoder_frontend, ssl.encoder, cfg, BatchLayout).to(device)
    adapter_inventory(model)
    return model
