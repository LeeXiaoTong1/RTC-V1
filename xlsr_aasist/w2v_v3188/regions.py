"""Full-coverage window ASP plus genuine within-window temporal variation."""
from collections import defaultdict
import torch
from torch import nn
from w2v_v3184_aug5.regions import StatisticsHead as PriorStatisticsHead, moments


class StatisticsHead(PriorStatisticsHead):
    """320d window mean/std plus 64d temporal std, with one AASIST pass.

    Short recordings keep their actual length. A single window correctly has
    zero between-window std, but can carry real temporal variation in its e_T
    nodes. Artificial tiny-input reflection remains classifier context; only
    PCL uses the original-audio mask. No batch padding enters this head.
    """
    def __init__(self, old, temporal_stats=True):
        super().__init__(old)
        self.temporal_stats = bool(temporal_stats)
        previous = self.aasist.out_layer
        # Allocate the full compatible readout before strict parent transfer.
        # Every trained column is loaded; never truncate a parent classifier.
        self.aasist.out_layer = nn.Linear(384 if self.temporal_stats else 320, 2,
            device=previous.weight.device, dtype=previous.weight.dtype)

    def forward(self, sequences, return_weights=False):
        if not sequences:
            raise ValueError('At least one utterance is required')
        groups, plans, collected, temporal = defaultdict(list), [], {}, {}
        for i, seq in enumerate(sequences):
            if seq.ndim != 2 or seq.shape[1] != 128 or len(seq) < 12:
                raise ValueError('AASIST requires >=12 frames and 128 features')
            ranges, mass = self.window_plan(len(seq), seq.device)
            plans.append((ranges, mass))
            for j, (a, b) in enumerate(ranges):
                groups[b-a].append((i, j, seq[a:b]))
        for _, records in sorted(groups.items()):
            records.sort(key=lambda record: (record[1], record[0]))
            for start in range(0, len(records), self.window_batch):
                chunk = records[start:start+self.window_batch]
                inputs = torch.stack([record[2] for record in chunk]).float()
                if self.temporal_stats:
                    features, deviations = self.aasist.evidence_with_temporal(inputs)
                    if deviations.shape != (len(chunk), 64):
                        raise ValueError('AASIST temporal evidence must have 64 channels')
                    for record, deviation in zip(chunk, deviations):
                        temporal[record[0], record[1]] = deviation
                else:
                    features = self.aasist.evidence(inputs)
                for record, feature in zip(chunk, features):
                    collected[record[0], record[1]] = feature
        aggregates, weights = [], []
        for i, (ranges, mass) in enumerate(plans):
            evidence = torch.stack([collected[i, j] for j in range(len(ranges))])
            weight = self.weights(evidence, mass)
            pieces = list(moments(evidence, weight))
            if self.temporal_stats:
                deviations = torch.stack([temporal[i, j] for j in range(len(ranges))])
                pieces.append((weight[:, None] * deviations).sum(0))
            aggregates.append(torch.cat(pieces))
            weights.append(weight)
        logits = self.aasist.classify(torch.stack(aggregates))
        return (logits, weights) if return_weights else logits
