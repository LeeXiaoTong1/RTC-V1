"""w2v-BERT + AASIST topology, with sample-wise (not batch-wise) normalization.

AASIST topology/equations derive from model/model.py at source commit
9d925407af846b2c544c8136433f49694bbcec00, originally Hemlata Tak et al.
BN2d -> GroupNorm; graph BN1d -> LayerNorm. No fairseq or import monkey-patching.
The residual preactivation, discarded in the source, is actually used here.
"""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F


def parameter(*shape):
    p = nn.Parameter(torch.empty(*shape))
    nn.init.xavier_normal_(p)
    return p


def norm2d(channels):
    return nn.GroupNorm(min(8, channels), channels)


class GraphAttention(nn.Module):
    def __init__(self, cin, cout, temperature):
        super().__init__()
        self.att_proj = nn.Linear(cin, cout)
        self.att_weight = parameter(cout, 1)
        self.proj_with_att = nn.Linear(cin, cout)
        self.proj_without_att = nn.Linear(cin, cout)
        self.norm = nn.LayerNorm(cout)
        self.input_drop = nn.Dropout(.2)
        self.temperature = temperature

    def forward(self, x):
        x = self.input_drop(x)
        scores = torch.tanh(self.att_proj(x.unsqueeze(2) * x.unsqueeze(1))) @ self.att_weight
        attention = (scores / self.temperature).softmax(dim=-2).squeeze(-1)
        return F.selu(self.norm(self.proj_with_att(attention @ x) + self.proj_without_att(x)))


class HeterogeneousAttention(nn.Module):
    def __init__(self, cin, cout, temperature=100.):
        super().__init__()
        self.proj_type1, self.proj_type2 = nn.Linear(cin, cin), nn.Linear(cin, cin)
        self.att_proj, self.att_proj_master = nn.Linear(cin, cout), nn.Linear(cin, cout)
        self.w11, self.w22, self.w12, self.wm = [parameter(cout, 1) for _ in range(4)]
        self.proj_with_att, self.proj_without_att = nn.Linear(cin, cout), nn.Linear(cin, cout)
        self.master_with_att, self.master_without_att = nn.Linear(cin, cout), nn.Linear(cin, cout)
        self.norm = nn.LayerNorm(cout)
        self.input_drop = nn.Dropout(.2)
        self.temperature = temperature

    def forward(self, x1, x2, master):
        n = x1.shape[1]
        x = self.input_drop(torch.cat((self.proj_type1(x1), self.proj_type2(x2)), 1))
        a = torch.tanh(self.att_proj(x.unsqueeze(2) * x.unsqueeze(1)))
        upper = torch.cat((a[:, :n, :n] @ self.w11, a[:, :n, n:] @ self.w12), 2)
        lower = torch.cat((a[:, n:, :n] @ self.w12, a[:, n:, n:] @ self.w22), 2)
        attention = (torch.cat((upper, lower), 1) / self.temperature).softmax(-2).squeeze(-1)
        am = ((torch.tanh(self.att_proj_master(x * master)) @ self.wm) / self.temperature).softmax(1)
        master = self.master_with_att(am.transpose(1, 2) @ x) + self.master_without_att(master)
        x = F.selu(self.norm(self.proj_with_att(attention @ x) + self.proj_without_att(x)))
        return x[:, :n], x[:, n:], master


class GraphPool(nn.Module):
    def __init__(self, dim, ratio=.5):
        super().__init__()
        self.proj, self.drop = nn.Linear(dim, 1), nn.Dropout(.3)
        self.ratio = ratio

    def forward(self, x):
        scores = torch.sigmoid(self.proj(self.drop(x)))
        count = max(1, int(x.shape[1] * self.ratio))
        idx = scores.topk(count, dim=1).indices.expand(-1, -1, x.shape[2])
        return (x * scores).gather(1, idx)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, first=False):
        super().__init__()
        self.pre = nn.Identity() if first else nn.Sequential(norm2d(cin), nn.SELU())
        self.conv1 = nn.Conv2d(cin, cout, (2, 3), padding=(1, 1))
        self.norm = norm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, (2, 3), padding=(0, 1))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, (1, 3), padding=(0, 1))

    def forward(self, x):
        # Source bug: conv1(x) discarded bn1/activation. Keep preactivation in graph.
        y = self.conv1(self.pre(x))
        return self.conv2(F.selu(self.norm(y))) + self.skip(x)


class AASISTSampleNorm(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.projection = nn.Linear(input_dim, 128)
        self.first_norm = norm2d(1)
        self.conv = nn.Sequential(ResidualBlock(1, 32, True), ResidualBlock(32, 32),
                                  ResidualBlock(32, 64), *[ResidualBlock(64, 64) for _ in range(3)])
        self.last_norm = norm2d(64)
        self.attention = nn.Sequential(nn.Conv2d(64, 128, 1), nn.SELU(), norm2d(128), nn.Conv2d(128, 64, 1))
        self.pos_s = nn.Parameter(torch.randn(1, 42, 64))
        self.master1, self.master2 = parameter(1, 1, 64), parameter(1, 1, 64)
        self.gat_s, self.gat_t = GraphAttention(64, 64, 2.), GraphAttention(64, 64, 2.)
        self.pool_s, self.pool_t = GraphPool(64), GraphPool(64)
        self.h11, self.h12 = HeterogeneousAttention(64, 32), HeterogeneousAttention(32, 32)
        self.h21, self.h22 = HeterogeneousAttention(64, 32), HeterogeneousAttention(32, 32)
        self.ps1, self.pt1, self.ps2, self.pt2 = [GraphPool(32) for _ in range(4)]
        self.drop_way, self.drop = nn.Dropout(.2), nn.Dropout(.5)
        self.classifier = nn.Linear(160, 2)

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1] < 12:
            raise ValueError('AASIST requires [B,T,D], T >= 12')
        x = self.projection(hidden).transpose(1, 2).unsqueeze(1)
        x = F.selu(self.first_norm(F.max_pool2d(x, (3, 3))))
        x = F.selu(self.last_norm(self.conv(x)))
        a = self.attention(x)
        s = (x * a.softmax(-1)).sum(-1).transpose(1, 2) + self.pos_s
        t = (x * a.softmax(-2)).sum(-2).transpose(1, 2)
        s, t = self.pool_s(self.gat_s(s)), self.pool_t(self.gat_t(t))
        t1, s1, m1 = self.h11(t, s, self.master1)
        t1, s1 = self.pt1(t1), self.ps1(s1)
        dt, ds, dm = self.h12(t1, s1, m1)
        t1, s1, m1 = t1 + dt, s1 + ds, m1 + dm
        t2, s2, m2 = self.h21(t, s, self.master2)
        t2, s2 = self.pt2(t2), self.ps2(s2)
        dt, ds, dm = self.h22(t2, s2, m2)
        t2, s2, m2 = t2 + dt, s2 + ds, m2 + dm
        t = torch.maximum(self.drop_way(t1), self.drop_way(t2))
        s = torch.maximum(self.drop_way(s1), self.drop_way(s2))
        m = torch.maximum(self.drop_way(m1), self.drop_way(m2))
        features = torch.cat((t.abs().amax(1), t.mean(1), s.abs().amax(1), s.mean(1), m.squeeze(1)), 1)
        return self.classifier(self.drop(features)), features


class Detector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = AASISTSampleNorm(backbone.config.hidden_size)
        # Disable only the unused SpecAugment placeholder, not the feature projection/layers.
        if hasattr(backbone, 'masked_spec_embed'):
            backbone.masked_spec_embed.requires_grad_(False)
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.modules()):
            raise RuntimeError('Unexpected BatchNorm: sample-independent microbatching would be invalid')

    @classmethod
    def load(cls, model_dir, config_dict=None, checkpointing=True):
        import transformers
        if transformers.__version__ != "4.38.2":
            raise RuntimeError("This implementation targets transformers==4.38.2; do not silently change encoder APIs")
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
        if config_dict is None:
            config = Wav2Vec2BertConfig.from_pretrained(model_dir, local_files_only=True)
            if (config.hidden_size, config.num_hidden_layers, config.feature_projection_input_dim) != (1024, 24, 160):
                raise ValueError('Expected original facebook/w2v-bert-2.0 configuration')
            config.layerdrop = 0.0
            config.apply_spec_augment = False  # waveform augmentations are already supplied by V2
            backbone = Wav2Vec2BertModel.from_pretrained(model_dir, config=config, local_files_only=True,
                                                        use_safetensors=True)
        else:
            config = Wav2Vec2BertConfig.from_dict(config_dict)
            backbone = Wav2Vec2BertModel(config)
        if config.add_adapter or config.layerdrop != 0 or config.apply_spec_augment:
            raise ValueError('Unexpected encoder adapter / stochastic layer skip / SpecAugment')
        if checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            backbone.gradient_checkpointing_disable()
        return cls(backbone)

    def forward(self, features, mask):
        if features.ndim != 3 or mask.shape != features.shape[:2] or not bool(mask.bool().all()):
            raise ValueError('Fixed-length features must be trimmed to all-valid frames before AASIST')
        h = self.backbone(input_features=features, attention_mask=mask,
                          output_hidden_states=False, return_dict=True).last_hidden_state
        return self.head(h)


def forward_chunks(model, features, mask, microbatch):
    """Keep one logical loss/optimizer step; do not split the contrastive negative set."""
    if microbatch < 1:
        raise ValueError('microbatch must be positive')
    outputs = [model(features[i:i + microbatch], mask[i:i + microbatch])
               for i in range(0, len(features), microbatch)]
    return torch.cat([x[0] for x in outputs]), torch.cat([x[1] for x in outputs])
