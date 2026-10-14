"""Legacy whole-utterance head; unused by the restored-window V3.18.6 recipe."""
from collections import defaultdict
import torch
from torch import nn


class WholeUtteranceHead(nn.Module):
    """Group *equal* lengths; padding is never fed to unmasked AASIST/BN.

    AASIST's first pool reduces time by three. Bound its quadratic temporal
    graph work explicitly; an oversized utterance fails rather than silently
    cropping, chunking, or restoring the removed window aggregator.
    """
    def __init__(self, aasist, max_frames=3000, graph_budget=1_000_000, batch_size=8):
        super().__init__()
        if min(max_frames, graph_budget, batch_size) < 1:
            raise ValueError('Positive whole-utterance limits required')
        self.aasist = aasist
        self.max_frames, self.graph_budget, self.batch_size = max_frames, graph_budget, batch_size

    def forward(self, sequences):
        if not sequences:
            raise ValueError('At least one utterance is required')
        groups = defaultdict(list)
        for index, sequence in enumerate(sequences):
            frames = len(sequence)
            if sequence.ndim != 2 or sequence.shape[1] != 128 or frames < 12:
                raise ValueError('AASIST requires >=12 unpadded frames with 128 features')
            if frames > self.max_frames or (frames // 3) ** 2 > self.graph_budget:
                raise ValueError(f'Whole-utterance AASIST limit exceeded: {frames} frames; '
                                 'raise explicit whole_max_frames/whole_graph_budget after checking GPU memory; no crop fallback')
            groups[frames].append(index)
        output = [None] * len(sequences)
        for frames, indices in groups.items():
            capacity = min(self.batch_size, self.graph_budget // (frames // 3) ** 2)
            for start in range(0, len(indices), capacity):
                selected = indices[start:start + capacity]
                features = self.aasist.evidence(torch.stack([sequences[i] for i in selected]).float())
                logits = self.aasist.classify(features)
                for index, value in zip(selected, logits):
                    output[index] = value
        return torch.stack(output)
