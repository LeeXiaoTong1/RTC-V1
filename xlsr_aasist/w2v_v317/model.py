"""Frozen public encoder, Q/V LoRA and a single constrained detection path."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from w2v_v3.model import HeadConfig, MultiConvBlock, SwiGLU, mask_frames
from w2v_v32.model import _TrainableCheckpoint
from w2v_v39.common import verify_files


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16., dropout=.05):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError('LoRA requires Linear and a valid positive rank')
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid LoRA scale/dropout')
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale, self.drop = alpha / rank, nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scale


class ForensicHead(nn.Module):
    def __init__(self, hidden, layers, cfg):
        super().__init__()
        dim = cfg['feature_dim']
        self.config = HeadConfig(input_dim=hidden, projection=dim, expansion=cfg['head_expansion'],
                                 blocks=cfg['head_blocks'], dropout=cfg['block_dropout'])
        self.input_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.projection = nn.Linear(hidden, dim)
        self.gating = SwiGLU(dim)
        self.blocks = nn.ModuleList([MultiConvBlock(self.config) for _ in range(cfg['head_blocks'])])
        self.forensic = nn.Sequential(nn.Linear(cfg['head_blocks'] * dim, dim), nn.LayerNorm(dim))
        self.attention = nn.Linear(dim, 1, bias=False)
        self.classifier = nn.Sequential(nn.Dropout(cfg['classifier_dropout']), nn.Linear(2 * dim, 2))
        self.fusion_chunk_layers = cfg.get('fusion_chunk_layers', 5)

    def forward(self, states, mask):
        if len(states) != len(self.layer_logits) or not bool(mask.bool().any(1).all()):
            raise ValueError('Expected all SSL states and nonempty valid trajectories')
        weights = self.layer_logits.float().softmax(0)
        x = None
        # Projection batching saves launch overhead; full lengths and all layers remain.
        for start in range(0, len(states), self.fusion_chunk_layers):
            chunk = states[start:start+self.fusion_chunk_layers]
            joined = torch.cat(chunk, dim=0) if len(chunk) > 1 else chunk[0]
            z = self.gating(self.projection(self.input_norm(joined)))
            for offset, item in enumerate(z.split(mask.shape[0], dim=0)):
                item = item * weights[start+offset].to(item.dtype)
                x = item if x is None else x + item
        x, frames = mask_frames(x, mask), []
        for block in self.blocks:
            x = block(x, mask)
            frames.append(x)
        forensic = mask_frames(self.forensic(torch.cat(frames, dim=-1)), mask)
        # This is the ONLY path to classification. TFCL receives this same tensor.
        alpha = self.attention(forensic).squeeze(-1).float().masked_fill(~mask.bool(), -torch.inf).softmax(-1)
        values = forensic.float()
        mean = (alpha[..., None] * values).sum(1)
        variance = (alpha[..., None] * (values - mean[:, None]).square()).sum(1)
        stats = torch.cat((mean, variance.clamp_min(1e-10).sqrt()), dim=-1)
        return self.classifier(stats), forensic


class Detector(nn.Module):
    supports_padded_training = True

    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        count = cfg['lora_layers']
        if not 1 <= count <= len(backbone.encoder.layers):
            raise ValueError('LoRA layer count exceeds the encoder')
        backbone.config.layerdrop = 0.
        backbone.config.apply_spec_augment = False
        self.lora_layers = count
        for layer in backbone.encoder.layers[-count:]:
            for name in ('linear_q', 'linear_v'):
                original = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, LoRALinear(original, cfg['lora_rank'],
                    cfg['lora_alpha'], cfg['lora_dropout']))
        self.head = ForensicHead(backbone.config.hidden_size, len(backbone.encoder.layers)+1, cfg)
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        checkpoint = backbone.encoder._gradient_checkpointing_func
        backbone.encoder._gradient_checkpointing_func = _TrainableCheckpoint(checkpoint)
        for layer in backbone.encoder.layers:
            layer._rtc_frozen_checkpoint = not any(p.requires_grad for p in layer.parameters())
        self.backbone.encoder.gradient_checkpointing = cfg['checkpointing']
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode:
            self.backbone.training = self.backbone.encoder.training = True
            for layer in self.backbone.encoder.layers[-self.lora_layers:]:
                layer.train(True)
        return self

    def forward(self, features, mask):
        if (features.ndim != 3 or features.shape[:2] != mask.shape or
                not bool(((mask == 0) | (mask == 1)).all())):
            raise ValueError('Expected feature frames with a binary mask')
        valid = mask.bool()
        if not bool(valid.any(1).all()) or bool((~valid[:, :-1] & valid[:, 1:]).any()):
            raise ValueError('Each waveform must have a nonempty valid prefix')
        out = self.backbone(input_features=features.masked_fill(~valid[..., None], 0),
                            attention_mask=mask, output_hidden_states=True, return_dict=True)
        return self.head(out.hidden_states, mask)


def load_model(cfg, device=None, training=True):
    import transformers
    if transformers.__version__ != '4.38.2':
        raise RuntimeError('Use the existing transformers==4.38.2 runtime')
    from transformers import Wav2Vec2BertModel
    verify_files(cfg['pretrained_fingerprints'])
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(cfg['seed'])
        backbone = Wav2Vec2BertModel.from_pretrained(cfg['pretrained_path'], local_files_only=True,
                                                    torch_dtype=torch.float32)
        if backbone.config.add_adapter:
            raise ValueError('Temporal encoder adapters are unsupported')
        model = Detector(backbone, cfg)
    return model.to(device or cfg['device']).train(training)


def optimizer_for(model, cfg, auxiliary):
    groups = []
    for kind, params, rate in (
        ('lora', [p for n,p in model.backbone.named_parameters() if p.requires_grad], cfg['lora_lr']),
        ('detection_head', list(model.head.parameters()), cfg['head_lr']),
        ('training_only_tfcl', list(auxiliary.parameters()), cfg['tfcl_lr'])):
        groups.append(dict(name=kind, params=params, lr=rate, initial_lr=rate))
    actual = [p for g in groups for p in g['params']]
    expected = [p for module in (model, auxiliary) for p in module.parameters() if p.requires_grad]
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != {id(p) for p in expected}:
        raise ValueError('Optimizer must cover each trainable parameter exactly once')
    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])


def inventory(model, auxiliary):
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    unexpected = [n for n,p in model.backbone.named_parameters() if p.requires_grad and not n.endswith(('lora_A','lora_B'))]
    if unexpected: raise ValueError('Unfrozen base encoder parameters: ' + str(unexpected))
    return dict(lora=count(model.backbone), head=count(model.head), detector=count(model),
                auxiliary=count(auxiliary), total=count(model)+count(auxiliary),
                original_encoder_frozen=True, feature_site='post_multiconv_forensic',
                no_unconstrained_classifier_bypass=True)
