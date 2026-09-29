"""Checkpoint-identical AASIST; whole utterances are processed without time padding."""
import torch
from w2v_rebuild.model import Detector as OriginalDetector


class Detector(OriginalDetector):
    def forward(self, features, mask):
        if features.ndim != 3 or mask.shape != features.shape[:2] or not bool(mask.bool().all()):
            raise ValueError('Use exact-length microbatches; padding changes Conformer/GN boundary statistics')
        if features.shape[1] < 12:
            raise ValueError('AASIST needs at least 12 acoustic frames')
        h = self.backbone(input_features=features, attention_mask=mask,
                          output_hidden_states=False, return_dict=True).last_hidden_state
        return self.head(h)

    def architecture(self):
        return {'model_config': self.backbone.config.to_dict()}


def microbatches(examples, size=4, frame_budget=1600):
    """Group identical frame lengths only; never pad or truncate a waveform."""
    if size < 1 or frame_budget < 1:
        raise ValueError('Positive microbatch and frame budget required')
    buckets = {}
    for i, example in enumerate(examples):
        frames = example['features'].shape[1]
        buckets.setdefault(frames, []).append(i)
    for frames, indices in buckets.items():
        count = min(size, max(1, frame_budget // frames))
        for start in range(0, len(indices), count):
            selected = indices[start:start + count]
            yield (selected, torch.cat([examples[i]['features'] for i in selected]),
                   torch.cat([examples[i]['mask'] for i in selected]))
