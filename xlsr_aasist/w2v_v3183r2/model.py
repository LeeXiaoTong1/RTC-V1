"""One Omni pass, full-depth learned reader, and the retained AASIST backend."""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .aasist import SSLAASIST
from .regions import RegionHead
from .common import require_finite
from .fusion import LayerFusion


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32., dropout=.05):
        super().__init__()
        self.base = base.requires_grad_(False)
        out_dim, in_dim = base.weight.shape
        self.lora_a = nn.Parameter(torch.empty(rank, in_dim, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(out_dim, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5**.5)
        self.scale, self.drop, self.enabled = alpha/rank, nn.Dropout(dropout), True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + F.linear(F.linear(self.drop(x), self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))*self.scale


def inject(module, cfg):
    found = []
    # fairseq2 FFNs also have output_proj (model_dim x ffn_inner_dim).
    # Recognize a complete attention group before touching any projection.
    if all(hasattr(module, name) for name in ('q_proj', 'k_proj', 'v_proj')):
        for name in ('q_proj', 'k_proj', 'v_proj', 'output_proj'):
            child = getattr(module, name, None)
            if child is None or not hasattr(child, 'weight'):
                raise ValueError('Incomplete attention projection group: '+name)
            if child.weight.shape != (cfg['encoder_dim'], cfg['encoder_dim']):
                raise ValueError('Unexpected attention projection shape: '+name)
            setattr(module, name, LoRALinear(child, cfg['lora_rank'], cfg['lora_alpha'], cfg['lora_dropout']))
            found.append(name)
        return found
    for name, child in list(module.named_children()):
        found.extend(name+'.'+s for s in inject(child, cfg))
    return found


class RecomputedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__(); self.layer, self.enabled = layer, False

    def forward(self, x, layout, bias_cache):
        if self.enabled and torch.is_grad_enabled():
            return checkpoint(self.layer, x, layout, bias_cache, use_reentrant=False)
        return self.layer(x, layout, bias_cache)


class Detector(nn.Module):
    def __init__(self, frontend, encoder, cfg, layout_factory):
        super().__init__()
        self.frontend, self.encoder = frontend.requires_grad_(False), encoder.requires_grad_(False)
        self.cfg, self.layout_factory = cfg, layout_factory
        self.joint, self.lora_inventory = False, {}
        count = cfg['lora_layers']
        if not 1 <= count <= len(encoder.layers):
            raise ValueError('Invalid number of LoRA layers')
        for i in range(len(encoder.layers)-count, len(encoder.layers)):
            names = inject(encoder.layers[i], cfg)
            if len(names) != 4:
                raise ValueError(f'Layer {i}: expected four Q/K/V/O projections, found {names}')
            self.lora_inventory[str(i)] = names
            encoder.layers[i] = RecomputedLayer(encoder.layers[i])
        if cfg.get('fusion_dim', 128) != 128:
            raise ValueError('AASIST spectral geometry requires a 128-dimensional fused representation')
        if tuple(cfg['fusion_layers']) != tuple(range(1, len(encoder.layers)+1)):
            raise ValueError('R2 requires every encoder layer in the learned reader')
        self.fusion = LayerFusion(cfg['encoder_dim'], len(encoder.layers), 128, cfg['checkpointing'])
        aasist = SSLAASIST(128)
        # The reader now owns the old SSL->128 projection. Applying LL again
        # would introduce an unintended second projection and reset geometry.
        aasist.LL = nn.Identity()
        self.head = RegionHead(aasist, cfg['local_evidence'], cfg['window'], cfg['hop'], cfg['window_batch'])
        for module in self.head.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.momentum = cfg.get('bn_momentum', .01)
        self.set_phase(False)

    def set_phase(self, joint):
        self.joint = joint
        self.head.learn_contribution = True
        for m in self.encoder.modules():
            if isinstance(m, LoRALinear):
                # Freezing inherited adapters must NOT remove their learned
                # contribution from the warmup feature representation.
                m.enabled = True
                m.lora_a.requires_grad_(joint); m.lora_b.requires_grad_(joint)
            if isinstance(m, RecomputedLayer): m.enabled = joint and self.cfg['checkpointing']
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval(); self.encoder.eval()
        for m in self.encoder.modules():
            if isinstance(m, LoRALinear): m.drop.train(mode and self.joint)
        # The full-depth reader continues changing in joint training. BN keeps
        # updating from training waveforms only; eval uses the saved buffers.
        return self

    def reset_bn_statistics(self):
        for module in self.head.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()

    def encode(self, waves):
        parameter = next(self.frontend.parameters())
        device, dtype = parameter.device, parameter.dtype
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with amp:
            groups, sequences = {}, [None]*len(waves)
            for i, w in enumerate(waves): groups.setdefault(len(w), []).append(i)
            with torch.no_grad():
                for n, indices in groups.items():
                    if n < 3920: raise ValueError('Waveform must have >=12 Omni frames; extend short audio explicitly')
                    x = torch.stack([torch.as_tensor(waves[i], device=device, dtype=torch.float32) for i in indices])
                    require_finite(x, 'waveform')
                    x = F.layer_norm(x, (n,)).to(dtype)
                    y, layout = self.frontend(x, self.layout_factory(x.shape, seq_lens=[n]*len(indices), device=device))
                    for j, i in enumerate(indices):
                        length = int(layout.seq_lens[j])
                        if length != (n-400)//320+1: raise ValueError('Omni frame geometry changed')
                        sequences[i] = y[j, :length]
            lengths = [len(s) for s in sequences]
            padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            # BatchLayout describes batch/time only; the feature axis is not a sequence axis.
            layout = self.layout_factory(padded.shape[:2], seq_lens=lengths, device=device)
            weights, mixed, seen, handles = self.fusion.weights(), None, set(), []
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
                # Hooks attach to the recomputation wrapper, not its inner layer.
                # Reader transforms have their own cheap checkpointing. Hooks
                # are removed before encoder backward recomputation.
                for index in self.fusion.layers[:-1]:
                    handles.append(self.encoder.layers[index-1].register_forward_hook(capture(index)))
                final = self.encoder(padded, layout)
            finally:
                for handle in handles: handle.remove()
            if seen != set(self.fusion.layers[:-1]):
                raise RuntimeError('An encoder layer did not reach the fusion reader')
            # Native encoder final normalization occurs AFTER its last layer
            # hook. Consume the actual final output as layer 60 exactly once.
            last = self.fusion.project_layer(final, self.fusion.layers[-1], weights)
            fused = last if mixed is None else mixed + last
            return [fused[i, :n] for i, n in enumerate(lengths)]

    def forward(self, waves):
        sequences = self.encode(waves)
        # BN, graph scores and classification remain FP32, independent of SSL BF16.
        with torch.autocast(next(self.head.parameters()).device.type, enabled=False):
            logits = self.head(sequences)
        require_finite(logits, 'detector logits')
        return logits


def load_model(cfg):
    from .assets import configured_spec
    item=configured_spec(cfg)
    from .runtime import require_version
    require_version('fairseq2','0.6.0')
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    hub = get_wav2vec2_model_hub(); arch = hub.get_arch_config(item['arch'])
    ec = arch.encoder_config
    if ec.model_dim != item['encoder_dim'] or ec.num_encoder_layers != item['encoder_layers']:
        raise ValueError('Installed Omni architecture disagrees with the pinned '+item['arch']+' contract')
    device = torch.device(cfg['device'])
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    ssl = hub.load_custom_model(Path(cfg['omni_checkpoint']), arch, device=device, dtype=dtype, mmap=True, restrict=True)
    if len(ssl.encoder.layers) != item['encoder_layers']: raise ValueError('Wrong SSL encoder')
    model = Detector(ssl.encoder_frontend, ssl.encoder, cfg, BatchLayout)
    del ssl
    return model.to(device)


def optimizer_for(model, cfg):
    parts = [('lora', [p for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))], cfg['lora_lr']),
             ('aasist', list(model.head.aasist.parameters()), cfg['head_lr']),
             ('evidence', [p for p in model.head.contribution.parameters() if p.requires_grad], cfg['evidence_lr']),
             ('fusion', [p for n,p in model.fusion.named_parameters() if n!='layer_logits'], cfg['fusion_lr']),
             ('layer_mix', [model.fusion.layer_logits], cfg['layer_mix_lr'])]
    groups = []
    for kind, parameters, rate in parts:
        for decay in (False, True):
            selected = [p for p in parameters if (p.ndim >= 2) == decay]
            if selected: groups.append(dict(name=kind, params=selected, lr=rate, initial_lr=rate, weight_decay=cfg['weight_decay'] if decay else 0.))
    actual = [p for g in groups for p in g['params']]
    expected = {id(p) for n,p in model.named_parameters() if p.requires_grad or n.endswith(('.lora_a','.lora_b'))}
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != expected:
        raise ValueError('Optimizer inventory mismatch')
    return torch.optim.AdamW(groups)
