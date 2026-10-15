"""Full-coverage window ASP with source-interleaved physical window batches."""
from collections import defaultdict
import torch
from w2v_v3184_aug5.regions import StatisticsHead as PriorStatisticsHead, moments


class StatisticsHead(PriorStatisticsHead):
    """Same 320-dimensional state geometry; change only window scheduling.

    Round-robin window index ordering prevents a long source from filling an
    entire BN minibatch when windows from other sources are available. Equal
    lengths alone are stacked, so padding never enters unmasked AASIST/BN.
    """
    def forward(self, sequences, return_weights=False):
        if not sequences:
            raise ValueError('At least one utterance is required')
        groups, plans, collected = defaultdict(list), [], {}
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
                features = self.aasist.evidence(torch.stack([record[2] for record in chunk]).float())
                for record, feature in zip(chunk, features):
                    collected[record[0], record[1]] = feature
        aggregates, weights = [], []
        for i, (ranges, mass) in enumerate(plans):
            evidence = torch.stack([collected[i, j] for j in range(len(ranges))])
            weight = self.weights(evidence, mass)
            aggregates.append(torch.cat(moments(evidence, weight)))
            weights.append(weight)
        logits = self.aasist.classify(torch.stack(aggregates))
        return (logits, weights) if return_weights else logits
