"""Full coverage windows, overlap-corrected mass, bounded evidence aggregation."""
from collections import defaultdict
import torch
from torch import nn


def windows(length, size=128, hop=64):
    if length < 12 or size < 12 or not 1 <= hop <= size:
        raise ValueError('Require >=12 valid frames and 0<hop<=window')
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, hop))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(a, a + size) for a in starts]


def coverage_mass(length, ranges, device=None):
    coverage = torch.zeros(length, dtype=torch.float32, device=device)
    for a, b in ranges:
        if not 0 <= a < b <= length:
            raise ValueError('Invalid window bounds')
        coverage[a:b] += 1
    if not (coverage > 0).all():
        raise ValueError('Window plan leaves uncovered frames')
    inverse = coverage.reciprocal()
    return torch.stack([inverse[a:b].sum() for a, b in ranges]) / length


class RegionHead(nn.Module):
    def __init__(self, aasist, local=True, window=128, hop=64, window_batch=32):
        super().__init__()
        self.aasist = aasist
        self.local, self.window, self.hop, self.window_batch = local, window, hop, window_batch
        self.contribution = nn.Sequential(nn.Linear(160, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.contribution[-1].weight); nn.init.zeros_(self.contribution[-1].bias)
        if not local:
            self.contribution.requires_grad_(False)
        self.learn_contribution = True

    def weights(self, evidence, mass):
        # Adjustment is in [0.5,1.5]; normalized weights can differ from mass
        # by [1/3,3]. It is NOT correct to promise [0.5,1.5] after normalization.
        scale = .5 + torch.sigmoid(self.contribution(evidence).squeeze(-1)) if self.learn_contribution else torch.ones_like(mass)
        adjusted = mass * scale
        return adjusted / adjusted.sum()

    def forward(self, sequences, return_weights=False):
        groups = defaultdict(list)
        plans, collected = [], {}
        for i, seq in enumerate(sequences):
            ranges = windows(len(seq), self.window, self.hop) if self.local else [(0, len(seq))]
            plans.append(ranges)
            for j, (a, b) in enumerate(ranges):
                groups[b-a].append((i, j, seq[a:b]))
        for _, records in sorted(groups.items()):
            for start in range(0, len(records), self.window_batch):
                chunk = records[start:start+self.window_batch]
                features = self.aasist.evidence(torch.stack([r[2] for r in chunk]).float())
                for r, feature in zip(chunk, features):
                    collected[r[0], r[1]] = feature
        aggregates, weights = [], []
        for i, (seq, ranges) in enumerate(zip(sequences, plans)):
            evidence = torch.stack([collected[i, j] for j in range(len(ranges))])
            mass = coverage_mass(len(seq), ranges, evidence.device)
            weight = self.weights(evidence, mass)
            aggregates.append((weight[:, None] * evidence).sum(0)); weights.append(weight)
        logits = self.aasist.classify(torch.stack(aggregates))
        return (logits, weights) if return_weights else logits
