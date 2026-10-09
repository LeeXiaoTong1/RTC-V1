"""Official W2V 1B encoder only; fresh SSL-AASIST, final-layer Q/K/V/O LoRA."""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .aasist import SSLAASIST
from .regions import RegionHead
from .common import require_finite


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
    for name, child in list(module.named_children()):
        if name in ('q_proj', 'k_proj', 'v_proj', 'output_proj') and hasattr(child, 'weight'):
            if child.weight.shape != (cfg['encoder_dim'], cfg['encoder_dim']):
                raise ValueError('Unexpected attention projection shape: '+name)
            setattr(module, name, LoRALinear(child, cfg['lora_rank'], cfg['lora_alpha'], cfg['lora_dropout']))
            found.append(name)
        else:
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
        self.head = RegionHead(SSLAASIST(cfg['encoder_dim']), cfg['local_evidence'], cfg['window'], cfg['hop'], cfg['window_batch'])
        self.set_phase(False)

    def set_phase(self, joint):
        self.joint = joint
        self.head.learn_contribution = joint
        for m in self.encoder.modules():
            if isinstance(m, LoRALinear): m.enabled = joint
            if isinstance(m, RecomputedLayer): m.enabled = joint and self.cfg['checkpointing']
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval(); self.encoder.eval()
        for m in self.encoder.modules():
            if isinstance(m, LoRALinear): m.drop.train(mode and self.joint)
        if self.joint:
            # Preserve affine learning while fixing running statistics after warmup.
            for m in self.head.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()
        return self

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
            layout = self.layout_factory(padded.shape, seq_lens=lengths, device=device)
            final = self.encoder(padded, layout)
            return [final[i, :n].float() for i, n in enumerate(lengths)]

    def forward(self, waves):
        sequences = self.encode(waves)
        # BN, graph scores and classification remain FP32, independent of SSL BF16.
        with torch.autocast(next(self.head.parameters()).device.type, enabled=False):
            logits = self.head(sequences)
        require_finite(logits, 'detector logits')
        return logits


def load_model(cfg):
    from importlib.metadata import version
    if version('fairseq2') != '0.6.0': raise RuntimeError('Use sdd-v318 with fairseq2==0.6.0')
    import omnilingual_asr
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    hub = get_wav2vec2_model_hub(); arch = hub.get_arch_config('1b')
    ec = arch.encoder_config
    if ec.model_dim != 1280 or ec.num_encoder_layers != 48:
        raise ValueError('Expected official Omni W2V 1B: 48 layers, 1280 dimensions')
    device = torch.device(cfg['device'])
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    ssl = hub.load_custom_model(Path(cfg['omni_checkpoint']), arch, device=device, dtype=dtype, mmap=True, restrict=True)
    if len(ssl.encoder.layers) != 48: raise ValueError('Wrong SSL encoder')
    model = Detector(ssl.encoder_frontend, ssl.encoder, cfg, BatchLayout)
    del ssl
    return model.to(device)


def optimizer_for(model, cfg):
    parts = [('lora', [p for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))], cfg['lora_lr']),
             ('aasist', list(model.head.aasist.parameters()), cfg['head_lr']),
             ('evidence', [p for p in model.head.contribution.parameters() if p.requires_grad], cfg['evidence_lr'])]
    groups = []
    for kind, parameters, rate in parts:
        for decay in (False, True):
            selected = [p for p in parameters if (p.ndim >= 2) == decay]
            if selected: groups.append(dict(name=kind, params=selected, lr=rate, initial_lr=rate, weight_decay=cfg['weight_decay'] if decay else 0.))
    actual = [p for g in groups for p in g['params']]
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Optimizer inventory mismatch')
    return torch.optim.AdamW(groups)
