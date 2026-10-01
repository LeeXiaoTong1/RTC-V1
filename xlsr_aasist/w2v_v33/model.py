"""Expose shared temporal features without adding parameters or encoder passes."""
import torch
from w2v_v3.model import MultiConvHead, mask_frames
from w2v_v32.model import (Detector as RuntimeDetector, FusedMultiConvHead,
                          install_runtime as install_v32_runtime, microbatches)


class FrameHead(FusedMultiConvHead):
    """The exact classifier's last-block frames also supervise communication pairs."""
    def forward_training(self, hidden_states, mask, *, _validated=False):
        if not hidden_states or mask.ndim != 2 or (not _validated and not bool(mask.bool().any(1).all())):
            raise ValueError('Nonempty hidden states and valid frames are required')
        for h in hidden_states:
            if h.shape[:2] != mask.shape or h.shape[-1] != self.config.input_dim:
                raise ValueError('SSL hidden-state shape differs from MultiConv input')
        x = None
        chunk_layers = getattr(self, 'fusion_chunk_layers', 5)
        for start in range(0, len(hidden_states), chunk_layers):
            chunk = hidden_states[start:start + chunk_layers]
            joined = chunk[0] if len(chunk) == 1 else torch.cat(chunk, dim=0)
            projected = self.gating(self.projection(joined))
            for z in projected.split(mask.shape[0], dim=0):
                x = z if x is None else x + z
        x = mask_frames(x, mask)
        frames = []
        for block in self.blocks:
            x = block(x, mask)
            frames.append(x)
        stats, block_means = self.pool(frames, mask)
        return self.classifier(stats), block_means, frames[-1]

    def forward(self, hidden_states, mask, *, _validated=False):
        if not self.training:
            return MultiConvHead.forward(self, hidden_states, mask)
        z, means, _ = self.forward_training(hidden_states, mask, _validated=_validated)
        return z, means


class Detector(RuntimeDetector):
    def __init__(self, backbone, head_config=None):
        super().__init__(backbone, head_config)
        self.head.__class__ = FrameHead
        self.head.fusion_chunk_layers = 5

    def forward_training(self, features, mask, *, _validated=False):
        if not self.training:
            raise ValueError('forward_training requires model.train(); evaluation uses forward')
        if features.ndim != 3 or mask.ndim != 2 or features.shape[:2] != mask.shape:
            raise ValueError('Expected [B,T,D] acoustic features and [B,T] prefix mask')
        if features.shape[1] < 1 or (not _validated and not bool(((mask == 0) | (mask == 1)).all())):
            raise ValueError('A binary prefix mask and at least one frame are required')
        valid = mask.bool()
        if not _validated and (not bool(valid.any(1).all()) or bool((~valid[:, :-1] & valid[:, 1:]).any())):
            raise ValueError('Each waveform requires one nonempty valid prefix')
        output = self.backbone(input_features=features.masked_fill(~valid.unsqueeze(-1), 0),
                               attention_mask=mask, output_hidden_states=True, return_dict=True)
        z, means, frames = self.head.forward_training(output.hidden_states, mask, _validated=True)
        return z, means, frames, valid


def install_runtime(model, chunk_layers=5):
    """Install the frame-returning interface while preserving every Parameter/RNG."""
    if isinstance(model, Detector):
        model.__class__ = RuntimeDetector
        model.head.__class__ = FusedMultiConvHead
    model = install_v32_runtime(model, chunk_layers)
    model.__class__ = Detector
    model.head.__class__ = FrameHead
    return model
