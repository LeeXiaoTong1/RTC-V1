"""Adapted from Hoan My TRAN's MIT-licensed MultiConv implementation.

See THIRD_PARTY_LICENSES/MultiConv.txt. Checkpoint-compatible with RTC-V1
ab937d6, with exact-length encoder batches, differentiable global CKA, and
masked temporal pooling. This is a w2v-BERT adaptation, not a paper reproduction.
"""
from dataclasses import asdict, dataclass
import itertools
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HeadConfig:
    input_dim: int = 1024
    projection: int = 128
    expansion: int = 1024
    blocks: int = 4
    kernels: tuple = (3, 7, 11, 15)
    merge_kernel: int = 15
    dropout: float = .1

    def __post_init__(self):
        if min(self.input_dim, self.projection, self.expansion, self.blocks) < 1:
            raise ValueError('Positive head dimensions and block count required')
        if not self.kernels or not 0 <= self.dropout < 1:
            raise ValueError('Nonempty kernels and dropout in [0,1) required')


def mask_frames(x, mask):
    return x.masked_fill(~mask.bool().unsqueeze(-1), 0)


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        return F.silu(self.gate(x)) * self.value(x)


class MultiConvBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg.expansion // 2
        if cfg.expansion % 2 or width % len(cfg.kernels):
            raise ValueError('Expansion/2 must be divisible by the kernel count')
        if any(k < 1 or k % 2 == 0 for k in (*cfg.kernels, cfg.merge_kernel)):
            raise ValueError('Convolution kernels must be positive odd integers')
        self.up = nn.Linear(cfg.projection, cfg.expansion)
        self.norm = nn.LayerNorm(width)
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width // len(cfg.kernels), k, padding=k // 2,
                      groups=width // len(cfg.kernels)) for k in cfg.kernels])
        self.merge = nn.Conv1d(width, width, cfg.merge_kernel,
                               padding=cfg.merge_kernel // 2, groups=width)
        self.linear = nn.Linear(width, width)
        self.drop = nn.Dropout(cfg.dropout)
        self.down = nn.Linear(width, cfg.projection)

    def forward(self, x, mask):
        a, b = F.gelu(self.up(x)).chunk(2, dim=-1)
        b = mask_frames(self.norm(b), mask).transpose(1, 2)
        b = torch.cat([conv(b) for conv in self.convs], dim=1)
        b = b.masked_fill(~mask.bool().unsqueeze(1), 0)
        b = b + self.merge(b)
        b = F.silu(self.linear(b.transpose(1, 2)))
        return mask_frames(self.down(self.drop(a * b)), mask)


class BlockStatisticsPool(nn.Module):
    """One attention head per block; concatenate all means, then all stds."""
    def __init__(self, blocks, dim):
        super().__init__()
        self.blocks, self.dim = blocks, dim
        self.attention = nn.Conv1d(blocks * dim, blocks, 1, groups=blocks, bias=False)

    def forward(self, frames, mask):
        # Preserve B,T,(block,channel). A direct view of B,block,T,D is incorrect.
        x = torch.cat(frames, dim=-1).transpose(1, 2)
        alpha = self.attention(x).float().masked_fill(~mask.bool().unsqueeze(1), -torch.inf)
        alpha = alpha.softmax(-1).unsqueeze(2)
        grouped = x.float().reshape(x.shape[0], self.blocks, self.dim, x.shape[-1])
        mean = (grouped * alpha).sum(-1)
        variance = ((grouped - mean.unsqueeze(-1)).square() * alpha).sum(-1)
        std = variance.clamp_min(1e-10).sqrt()
        return torch.cat((mean.flatten(1), std.flatten(1)), dim=1), mean


class MultiConvHead(nn.Module):
    def __init__(self, cfg=HeadConfig()):
        super().__init__()
        self.config = cfg
        self.projection = nn.Linear(cfg.input_dim, cfg.projection)
        self.gating = SwiGLU(cfg.projection)
        self.blocks = nn.ModuleList([MultiConvBlock(cfg) for _ in range(cfg.blocks)])
        self.pool = BlockStatisticsPool(cfg.blocks, cfg.projection)
        width = cfg.blocks * cfg.projection
        self.classifier = nn.Sequential(nn.Linear(2 * width, width), nn.SELU(), nn.Linear(width, 2))

    def forward(self, hidden_states, mask):
        if not hidden_states or mask.ndim != 2 or not bool(mask.bool().any(1).all()):
            raise ValueError('Nonempty hidden states and valid frames are required')
        x = None
        for h in hidden_states:
            if h.shape[:2] != mask.shape or h.shape[-1] != self.config.input_dim:
                raise ValueError('SSL hidden-state shape differs from MultiConv input')
            z = self.gating(self.projection(h))
            x = z if x is None else x + z
        x = mask_frames(x, mask)
        frames = []
        for block in self.blocks:
            x = block(x, mask)
            frames.append(x)
        stats, block_means = self.pool(frames, mask)
        return self.classifier(stats), block_means


def diversity_cka(block_means):
    """Biased linear CKA across the WHOLE logical batch; never detach the loss."""
    if block_means.ndim != 3:
        raise ValueError('Expected B,blocks,channels')
    if len(block_means) < 3:
        return block_means.sum() * 0
    x = block_means.float() - block_means.float().mean(0, keepdim=True)
    grams = torch.einsum('bkd,ckd->kbc', x, x)
    values = []
    for i, j in itertools.combinations(range(x.shape[1]), 2):
        d = grams[i].square().sum().clamp_min(1e-12).sqrt()
        d = d * grams[j].square().sum().clamp_min(1e-12).sqrt()
        values.append((grams[i] * grams[j]).sum() / d.clamp_min(1e-12))
    return torch.stack(values).mean() if values else x.sum() * 0


class Detector(nn.Module):
    def __init__(self, backbone, head_config=None):
        super().__init__()
        self.backbone = backbone
        self.head = MultiConvHead(head_config or HeadConfig(input_dim=backbone.config.hidden_size))
        self.trainable_layers = 0
        self.configure_trainable_layers(0)

    def configure_trainable_layers(self, count):
        layers = self.backbone.encoder.layers
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(layers):
            raise ValueError('Invalid trainable encoder layer count')
        self.trainable_layers = count
        self.backbone.requires_grad_(False)
        for layer in (layers[-count:] if count else []):
            layer.requires_grad_(True)
        for p in self.backbone.parameters():
            if not p.requires_grad:
                p.grad = None
        self.head.requires_grad_(True)
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.trainable_layers:
            self.backbone.training = True
            self.backbone.encoder.training = True
            for layer in self.backbone.encoder.layers[-self.trainable_layers:]:
                layer.train(True)
        return self

    def forward(self, features, mask):
        if features.ndim != 3 or features.shape[:2] != mask.shape or not bool(mask.bool().all()):
            raise ValueError('Use exact-length encoder microbatches; no padding or truncation')
        if features.shape[1] < 1:
            raise ValueError('At least one acoustic frame is required')
        output = self.backbone(input_features=features, attention_mask=mask,
                               output_hidden_states=True, return_dict=True)
        return self.head(output.hidden_states, mask)

    @classmethod
    def from_checkpoint(cls, state, checkpointing=True):
        if not isinstance(state, dict) or not {'model_config', 'head_config', 'model'} <= state.keys():
            raise ValueError('A MultiConv checkpoint with architecture and weights is required')
        model = cls.from_config(state['model_config'], state['head_config'], checkpointing)
        model.load_state_dict(state['model'], strict=True)
        return model

    @classmethod
    def from_original(cls, state, head_config=None, checkpointing=True):
        """Import the complete original encoder; never import the AASIST head."""
        if state.get('schema') != 'rtc_w2v_rebuild_v1':
            raise ValueError('Expected the original rtc_w2v_rebuild_v1 checkpoint')
        model = cls.from_config(state['model_config'], head_config, checkpointing)
        prefix = 'backbone.'
        backbone = {k[len(prefix):]: v for k, v in state['model'].items() if k.startswith(prefix)}
        if not backbone:
            raise ValueError('Original checkpoint has no encoder weights')
        model.backbone.load_state_dict(backbone, strict=True)
        return model

    @classmethod
    def from_config(cls, model_config, head_config=None, checkpointing=True):
        import transformers
        if transformers.__version__ != '4.38.2':
            raise RuntimeError('Use transformers==4.38.2, matching the original checkpoint')
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
        cfg = Wav2Vec2BertConfig.from_dict(model_config)
        if cfg.add_adapter:
            raise ValueError('Encoder adapters change frame alignment and are unsupported')
        cfg.layerdrop = 0.
        cfg.apply_spec_augment = False
        backbone = Wav2Vec2BertModel(cfg)
        if checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        return cls(backbone, HeadConfig(**head_config) if head_config else None)

    def architecture(self):
        return {'model_config': self.backbone.config.to_dict(), 'head_config': asdict(self.head.config)}


def microbatches(examples, size=4, frame_budget=1600):
    """Each complete waveform is encoded once, grouped only by exact frame length."""
    if size < 1 or frame_budget < 1:
        raise ValueError('Positive microbatch and frame budget required')
    buckets = {}
    for i, example in enumerate(examples):
        features, mask = example['features'], example['mask']
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[:2] != mask.shape:
            raise ValueError('Each example requires one [1,T,D] waveform and [1,T] mask')
        if features.shape[1] < 1 or not bool(mask.bool().all()):
            raise ValueError('Acoustic examples must contain only valid frames')
        buckets.setdefault(features.shape[1], []).append(i)
    for frames, indices in buckets.items():
        count = min(size, max(1, frame_budget // frames))
        for start in range(0, len(indices), count):
            selected = indices[start:start + count]
            yield (selected, torch.cat([examples[i]['features'] for i in selected]),
                   torch.cat([examples[i]['mask'] for i in selected]))
