"""Direct SSL encoder integration, never ASR decoding or SSL masking.

fairseq2 0.6 uses BatchLayout, not Hugging Face attention_mask APIs.
Only the selected upper layers receive LoRA. The immutable prefix stays eval.
"""
from contextlib import nullcontext
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from w2v_v3.model import HeadConfig, MultiConvHead, mask_frames


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32.):
        super().__init__()
        if rank < 1 or base.weight.ndim != 2:
            raise ValueError('LoRA requires a matrix projection and positive rank')
        self.base = base.requires_grad_(False)
        out_dim, in_dim = base.weight.shape
        self.a = nn.Parameter(torch.empty(rank, in_dim, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(out_dim, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5**.5)
        self.scale, self.enabled = alpha/rank, True

    def forward(self, x):
        value = self.base(x)
        if not self.enabled:
            return value
        # FP32 master parameters, compute in the activation dtype under BF16.
        delta = F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))
        return value + delta*self.scale


def add_lora(module, rank, alpha):
    names = []
    for name, child in list(module.named_children()):
        if name in ('q_proj', 'k_proj', 'v_proj', 'output_proj') and hasattr(child, 'weight'):
            setattr(module, name, LoRALinear(child, rank, alpha)); names.append(name)
        else:
            names.extend(name+'.'+n for n in add_lora(child, rank, alpha))
    return names


class RecomputedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__(); self.layer = layer; self.enabled = True

    def forward(self, x, layout, bias_cache):
        if self.enabled and torch.is_grad_enabled():
            return checkpoint(self.layer, x, layout, bias_cache, use_reentrant=False)
        return self.layer(x, layout, bias_cache)


class FusionHead(MultiConvHead):
    def __init__(self, input_dim, count, **kwargs):
        super().__init__(HeadConfig(input_dim=input_dim, **kwargs))
        self.layer_weights = nn.Parameter(torch.zeros(count))
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.fusion_norm = nn.LayerNorm(self.config.projection)

    def forward(self, hidden, valid):
        if len(hidden) != len(self.layer_weights):
            raise ValueError('Selected SSL layer count changed')
        weights = self.layer_weights.softmax(0)
        fused = sum(w*self.gating(self.projection(self.input_norm(h))) for w,h in zip(weights,hidden))
        fused = mask_frames(self.fusion_norm(fused),valid)
        x, frames = fused, []
        for block in self.blocks:
            x = block(x,valid); frames.append(x)
        pooled,_ = self.pool(frames,valid)
        return self.classifier(pooled), fused


class Detector(nn.Module):
    def __init__(self, frontend, encoder, cfg, layout_factory, dim=2048, head_kwargs=None):
        super().__init__()
        self.frontend, self.encoder = frontend,encoder
        self.frontend.requires_grad_(False); self.encoder.requires_grad_(False)
        self.layout_factory = layout_factory
        layers = encoder.layers
        count = cfg['lora_layers']
        if not 1 <= count <= len(layers):
            raise ValueError('LoRA layer count outside encoder')
        self.lora_inventory = {}
        for i in range(len(layers)-count,len(layers)):
            names = add_lora(layers[i],cfg['lora_rank'],cfg['lora_alpha'])
            if len(names) != 4:
                raise ValueError(f'Layer {i}: expected Q/K/V/output projections, found {names}')
            self.lora_inventory[str(i)] = names
            layers[i] = RecomputedLayer(layers[i])
        # Distributed depths retain acoustic cues without retaining 128 activations.
        self.feature_layers = tuple(cfg['feature_layers'])
        if (not self.feature_layers or len(set(self.feature_layers)) != len(self.feature_layers)
                or min(self.feature_layers)<0 or max(self.feature_layers)>=len(layers)):
            raise ValueError('Invalid feature layer indices')
        self.head = FusionHead(dim,len(self.feature_layers),**(head_kwargs or {}))
        self.amp = cfg.get('amp','bf16')
        self.set_phase(True,cfg.get('checkpointing',True))

    def set_phase(self, joint, recompute=True):
        for m in self.modules():
            if isinstance(m,LoRALinear): m.enabled = joint
            if isinstance(m,RecomputedLayer): m.enabled = joint and recompute

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval(); self.encoder.eval()
        return self

    def forward(self, waves):
        device = next(self.frontend.parameters()).device
        dtype = next(self.frontend.parameters()).dtype
        context = torch.autocast(device.type,dtype=torch.bfloat16) if self.amp=='bf16' else nullcontext()
        with context:
            # Exact-length CNN/positional frontend avoids padding-dependent GroupNorm.
            # Normalize the complete waveform before padding, as in official inference.
            groups = {}
            for i,w in enumerate(waves): groups.setdefault(len(w),[]).append(i)
            seqs = [None]*len(waves)
            with torch.no_grad():
                for n,indices in groups.items():
                    x = torch.stack([torch.as_tensor(waves[i],device=device,dtype=torch.float32) for i in indices])
                    if n<400 or not bool(torch.isfinite(x).all()):
                        raise ValueError('Empty, too short or nonfinite waveform')
                    x = F.layer_norm(x,(n,)).to(dtype)
                    y,layout = self.frontend(x,self.layout_factory(x.shape,seq_lens=[n]*len(indices),device=device))
                    for j,i in enumerate(indices): seqs[i] = y[j,:layout.seq_lens[j]]
            lengths = [len(s) for s in seqs]
            x = nn.utils.rnn.pad_sequence(seqs,batch_first=True)
            layout = self.layout_factory(x.shape,seq_lens=lengths,device=device)
            valid = torch.arange(x.shape[1],device=device)[None] < torch.tensor(lengths,device=device)[:,None]
            saved, handles = {}, []
            def remember(index):
                def hook(_m,_args,out): saved[index] = out
                return hook
            try:
                for index in self.feature_layers:
                    handles.append(self.encoder.layers[index].register_forward_hook(remember(index)))
                final = self.encoder(x,layout)
                # Last selected layer uses the encoder's final normalization.
                if len(self.encoder.layers)-1 in saved: saved[len(self.encoder.layers)-1] = final
                logits,fused = self.head([saved[i] for i in self.feature_layers],valid)
            finally:
                for h in handles: h.remove()
            return logits,fused,valid,lengths


def load_model(cfg):
    from importlib.metadata import version
    from packaging.version import Version
    if Version(version('fairseq2')) != Version('0.6.0'):
        raise RuntimeError('Use the separate V3.16 environment with fairseq2==0.6.0')
    import omnilingual_asr  # fairseq2 extension registers the 7b architecture
    from pathlib import Path
    from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
    from fairseq2.nn.batch_layout import BatchLayout
    hub = get_wav2vec2_model_hub()
    arch = hub.get_arch_config('7b')
    ssl = hub.load_custom_model(Path(cfg['omni_checkpoint']),arch,device=torch.device(cfg['device']),
        dtype=torch.bfloat16,mmap=True,restrict=True)
    if len(ssl.encoder.layers)!=128 or arch.encoder_config.model_dim!=2048:
        raise ValueError('Expected omniASR W2V 7B SSL encoder')
    model = Detector(ssl.encoder_frontend,ssl.encoder,cfg,BatchLayout)
    del ssl  # Discard pretraining quantizer/masker/projections; never copy them to checkpoints.
    model.head.to(cfg['device'])
    return model.to(cfg['device'])


def optimizer_for(model,auxiliary,cfg):
    lora = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.')]
    groups = [dict(name='lora',params=lora,lr=cfg['lora_lr'],initial_lr=cfg['lora_lr']),
              dict(name='head',params=list(model.head.parameters()),lr=cfg['head_lr'],initial_lr=cfg['head_lr']),
              dict(name='tfcl',params=list(auxiliary.parameters()),lr=cfg['tfcl_lr'],initial_lr=cfg['tfcl_lr'])]
    actual = [p for g in groups for p in g['params']]
    expected = [p for m in (model,auxiliary) for p in m.parameters() if p.requires_grad]
    if len({id(p) for p in actual})!=len(actual) or {id(p) for p in actual}!={id(p) for p in expected}:
        raise ValueError('Optimizer inventory mismatch')
    return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
