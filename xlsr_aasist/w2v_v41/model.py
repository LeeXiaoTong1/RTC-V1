"""Full-utterance w2v-BERT, 24-layer fusion and four-head statistics pooling.

The public Teffic-Audio report does not publish a head implementation. V4
uses four 256-channel subspaces, whose means/stds concatenate to 2048, as an
explicit implementation choice. There are no AASIST or overlapping windows.
"""
from contextlib import nullcontext
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .features import OfficialFeatures, frame_count


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32., dropout=.05):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 < rank <= min(base.weight.shape):
            raise ValueError('LoRA requires a Linear and a valid rank')
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale, self.drop = float(alpha)/rank, nn.Dropout(dropout)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))*self.scale


class RecomputedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer, self.enabled = layer, False

    def forward(self, *args, **kwargs):
        if self.enabled and torch.is_grad_enabled():
            return checkpoint(self.layer, *args, use_reentrant=False, **kwargs)
        return self.layer(*args, **kwargs)


class LayerFusion(nn.Module):
    def __init__(self, width, depth, checkpointing=True):
        super().__init__()
        self.layers = tuple(range(1, depth+1))
        self.output_dim, self.checkpointing = width, checkpointing
        self.input_norm = nn.LayerNorm(width, elementwise_affine=False)
        initial = torch.full((depth,), .5/max(1, depth-1))
        initial[-1] = .5 if depth > 1 else 1.
        self.layer_logits = nn.Parameter(initial.log())

    def weights(self):
        return self.layer_logits.float().softmax(0)

    def _normalize(self, value):
        with torch.autocast(value.device.type, enabled=False):
            return self.input_norm(value.float())

    def project_layer(self, value, index, weights):
        if index not in self.layers or value.shape[-1] != self.output_dim:
            raise ValueError('Unexpected fusion layer or width')
        normalized = checkpoint(self._normalize, value, use_reentrant=False) if self.checkpointing and torch.is_grad_enabled() else self._normalize(value)
        return normalized * weights[index-1]

    def describe(self):
        with torch.no_grad():
            weights = self.weights().cpu()
            entropy = float(-(weights*weights.clamp_min(1e-30).log()).sum())
            order = weights.argsort(descending=True)[:5].tolist()
            return dict(mode='all_layers_nonaffine_ln_softmax', layers=list(self.layers),
                weights=weights.tolist(), entropy=entropy, effective_layers=math.exp(entropy),
                top_layers=[dict(layer=i+1, weight=float(weights[i])) for i in order],
                final_weight=float(weights[-1]), output_dim=self.output_dim,
                initialization_final_mass=.5, final_bypass=False)


class MultiHeadStatistics(nn.Module):
    def __init__(self, width=1024, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('MHASP width must be divisible by attention heads')
        self.width, self.heads, self.head_dim = width, heads, width//heads
        hidden = max(4, self.head_dim//2)
        self.scorers = nn.ModuleList([nn.Sequential(nn.Linear(self.head_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)) for _ in range(heads)])

    def forward(self, values, mask, validated=False):
        if values.ndim != 3 or values.shape[:2] != mask.shape or values.shape[-1] != self.width:
            raise ValueError('Invalid MHASP sequence geometry')
        mask = mask.bool()
        if not validated and not bool(mask.any(1).all()):
            raise ValueError('MHASP requires at least one valid frame per recording')
        # Pooling adds neither dropout nor batch normalization before moments.
        with torch.autocast(values.device.type, enabled=False):
            pieces = values.float().reshape(*values.shape[:2], self.heads, self.head_dim)
            result = []
            for index, scorer in enumerate(self.scorers):
                x = pieces[:, :, index].masked_fill(~mask[..., None], 0.)
                logits = scorer(x).squeeze(-1).masked_fill(~mask, -torch.inf)
                weights = logits.softmax(-1)
                mean = (weights[..., None]*x).sum(1)
                variance = (weights[..., None]*(x-mean[:, None]).square()).sum(1)
                result.extend((mean, variance.clamp_min(1e-10).sqrt()))
            return torch.cat(result, -1)


class Head(nn.Module):
    def __init__(self, width, cfg):
        super().__init__()
        heads = cfg.get('pooling_heads', cfg.get('pool_heads', 4))
        if 'pool_heads' in cfg and cfg['pool_heads'] != heads:
            raise ValueError('Conflicting MHASP head counts')
        self.pooling = MultiHeadStatistics(width, heads)
        # Small HF fixtures keep the same three-hidden-layer topology.
        hidden = cfg.get('mlp_dims', [1536, 1024, 512])
        if len(hidden) != 3 or any(int(d) < 1 for d in hidden):
            raise ValueError('Expected three positive MLP hidden dimensions')
        self.input_dim, self.output_dim = 2*width, int(hidden[-1])
        blocks, previous = [], self.input_dim
        for size in hidden:
            blocks.extend((nn.Linear(previous, size), nn.LayerNorm(size), nn.GELU(), nn.Dropout(cfg.get('classifier_dropout', .1))))
            previous = size
        self.mlp = nn.Sequential(*blocks)
        self.classifier = nn.Linear(previous, 2)
        self.precision = cfg.get('backend_precision', 'fp16')

    def forward(self, sequences):
        if not sequences or any(len(s) == 0 for s in sequences):
            raise ValueError('Cannot classify empty sequences')
        lengths = torch.tensor([len(s) for s in sequences], device=sequences[0].device)
        values = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        mask = torch.arange(values.shape[1], device=values.device)[None] < lengths[:, None]
        stats = self.pooling(values, mask, validated=True)
        dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
        amp = torch.autocast('cuda', dtype=dtype) if values.is_cuda and self.precision != 'fp32' else nullcontext()
        with amp:
            return self.classifier(self.mlp(stats))


class Detector(nn.Module):
    def __init__(self, backbone, cfg, extractor=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone.requires_grad_(False)
        self.backbone.config.layerdrop = 0.
        self.backbone.config.apply_spec_augment = False
        if self.backbone.config.add_adapter:
            raise ValueError('Temporal adapter is incompatible with V4 timestamps')
        self.backbone.gradient_checkpointing_disable()
        depth, width = len(backbone.encoder.layers), backbone.config.hidden_size
        if cfg.get('lora_layers', depth) != depth:
            raise ValueError('V4 adapts every Conformer layer')
        if cfg.get('encoder_dim', width) != width or cfg.get('encoder_layers', depth) != depth:
            raise ValueError('Configured encoder geometry differs from pretrained weights')
        for index, layer in enumerate(backbone.encoder.layers):
            for name in ('linear_q', 'linear_k', 'linear_v', 'linear_out'):
                original = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, LoRALinear(original, cfg.get('lora_rank', 16), cfg.get('lora_alpha', 32.), cfg.get('lora_dropout', .05)))
            backbone.encoder.layers[index] = RecomputedLayer(layer)
        self.fusion = LayerFusion(width, depth, cfg.get('fusion_checkpointing', True))
        self.head = Head(width, cfg)
        self.features = OfficialFeatures(extractor) if extractor is not None else OfficialFeatures.from_pretrained(cfg['bert_model_dir'])
        self.joint = False
        self._health = None
        self.set_phase(cfg.get('warm_epochs', 1) == 0)

    @property
    def encoder(self):
        return self.backbone.encoder

    def set_phase(self, joint):
        self.joint = bool(joint)
        count = getattr(self, 'checkpoint_override', self.cfg.get('checkpoint_layers', len(self.encoder.layers)))
        if not 0 <= count <= len(self.encoder.layers):
            raise ValueError('Invalid checkpoint layer count')
        for index, layer in enumerate(self.encoder.layers):
            layer.enabled = bool(joint and self.cfg.get('checkpointing', True) and index >= len(self.encoder.layers)-count)
            for module in layer.modules():
                if isinstance(module, LoRALinear):
                    module.lora_a.requires_grad_(joint)
                    module.lora_b.requires_grad_(joint)
        self.train(self.training)

    def set_checkpoint_count(self, count):
        self.checkpoint_override = int(count)
        self.set_phase(self.joint)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.drop.train(bool(mode and self.joint))
        return self

    def encode(self, waves):
        device = next(self.backbone.parameters()).device
        features, mask, lengths = self.features(waves, device)
        return self.encode_features(features, mask, lengths)

    def encode_rows(self, rows):
        if not all('mel' in r for r in rows): return self.encode([r['wave'] for r in rows])
        device = next(self.backbone.parameters()).device
        return self.encode_features(*self.features.pack(rows, device))

    def encode_features(self, features, mask, lengths):
        device = features.device
        maximum = self.cfg.get('whole_max_frames', 3000)
        if any(n > maximum for n in lengths):
            raise ValueError(f'Full waveform exceeds whole_max_frames={maximum}; no crop fallback')
        self.last_encode_geometry = dict(encoded_views=len(lengths), frames=lengths, padded_frames=len(lengths)*max(lengths))
        weights, mixed, seen, handles = self.fusion.weights(), None, set(), []
        def capture(index):
            def hook(_module, _args, output):
                nonlocal mixed
                if index in seen:
                    raise RuntimeError('Fusion captured an encoder layer more than once')
                seen.add(index)
                hidden = output[0] if isinstance(output, tuple) else output
                term = self.fusion.project_layer(hidden, index, weights)
                mixed = term if mixed is None else mixed+term
            return hook
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        try:
            for index, layer in enumerate(self.encoder.layers, 1):
                handles.append(layer.register_forward_hook(capture(index)))
            with amp:
                self.backbone(input_features=features, attention_mask=mask.long(), output_hidden_states=False, output_attentions=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if seen != set(self.fusion.layers) or mixed is None:
            raise ValueError('Incomplete fused encoder output')
        self.check_finite(mixed, 'fused encoder output')
        return [mixed[i, :length] for i, length in enumerate(lengths)]

    def classify_sequences(self, sequences):
        logits = self.head(sequences)
        self.check_finite(logits, 'logits')
        return logits

    def check_finite(self, value, name):
        ok = torch.isfinite(value.detach()).all()
        if self._health is not None: self._health.append(ok)
        elif not bool(ok): raise FloatingPointError('Nonfinite V4.1 '+name)

    def begin_checks(self):
        self._health = []

    def finish_checks(self):
        checks, self._health = self._health, None
        return torch.stack(checks).all() if checks else next(self.parameters()).new_tensor(True, dtype=torch.bool)

    def forward_rows(self, rows):
        return self.classify_sequences(self.encode_rows(rows))

    def forward(self, waves):
        return self.classify_sequences(self.encode(waves))


def load_model(cfg):
    import transformers
    from transformers import Wav2Vec2BertModel
    from .assets import verify_assets
    if transformers.__version__ != cfg.get('transformers_version', '4.38.2'):
        raise RuntimeError('Use the pinned transformers runtime: '+cfg.get('transformers_version', '4.38.2'))
    verify_assets(cfg)
    device = torch.device(cfg['device'])
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(cfg.get('seed', 1234))
        backbone, info = Wav2Vec2BertModel.from_pretrained(cfg['bert_model_dir'], local_files_only=True,
            torch_dtype=dtype, output_loading_info=True)
        invalid = {k: info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') if info.get(k)}
        if invalid or (backbone.config.hidden_size, backbone.config.num_hidden_layers) != (1024, 24):
            raise ValueError('Official w2v-BERT encoder was not fully restored: '+str(invalid))
        model = Detector(backbone, cfg)
    return model.to(device)


def adapter_inventory(model):
    count, parameters = len(model.encoder.layers), []
    for layer in model.encoder.layers:
        adapters = [m for m in layer.modules() if isinstance(m, LoRALinear)]
        if len(adapters) != 4:
            raise ValueError('Each Conformer block must have four Q/K/V/O adapters')
        parameters.extend(p for m in adapters for p in (m.lora_a, m.lora_b))
    for name, p in model.backbone.named_parameters():
        if not name.endswith(('.lora_a', '.lora_b')) and p.requires_grad:
            raise ValueError('An original encoder parameter became trainable: '+name)
    if any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('LoRA master parameters must be FP32')
    return dict(encoder_depth=count, lora_layers=count, lora_projections=count*4,
        lora_layer_indices_zero_based=list(range(count)), fusion_layers_one_based=list(range(1, count+1)),
        lora_parameters=sum(p.numel() for p in parameters), active_lora_parameters=sum(p.numel() for p in parameters if p.requires_grad),
        recomputed_layer_indices_zero_based=[i for i, layer in enumerate(model.encoder.layers) if layer.enabled],
        frozen_prefix_indices_zero_based=[], original_encoder_frozen=True, output_dim=model.fusion.output_dim)


def optimizer_for(model, cfg):
    adapters = {'early': [], 'middle': [], 'late': []}
    for index, layer in enumerate(model.encoder.layers):
        bucket = ('early', 'middle', 'late')[min(2, index*3//len(model.encoder.layers))]
        adapters[bucket].extend(p for name, p in layer.named_parameters() if name.endswith(('.lora_a', '.lora_b')))
    rate = cfg.get('lora_lr', 2e-5)
    rates = (cfg.get('lora_early_lr', rate*.25), cfg.get('lora_middle_lr', rate*.5), rate)
    parts = [('lora_'+name, values, lr, 'lora') for (name, values), lr in zip(adapters.items(), rates)]
    parts += [('pooling', list(model.head.pooling.parameters()), cfg.get('pool_lr', cfg.get('head_lr', 1e-4)), 'head'),
        ('head', list(model.head.mlp.parameters()), cfg.get('head_lr', 1e-4), 'head'),
        ('classifier', list(model.head.classifier.parameters()), cfg.get('classifier_lr', 1e-4), 'head'),
        ('layer_mix', list(model.fusion.parameters()), cfg.get('layer_mix_lr', 1e-4), 'fusion')]
    groups = []
    for name, parameters, lr, role in parts:
        for decay in (False, True):
            selected = [p for p in parameters if (p.ndim >= 2) == decay]
            if selected:
                groups.append(dict(name=name, role=role, params=selected, lr=lr, initial_lr=lr, weight_decay=cfg.get('weight_decay', .01) if decay else 0.))
    actual = [p for g in groups for p in g['params']]
    expected = {id(p) for n, p in model.named_parameters() if p.requires_grad or n.endswith(('.lora_a', '.lora_b'))}
    if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != expected:
        raise ValueError('Optimizer must cover every trainable detector parameter exactly once')
    return torch.optim.AdamW(groups)
