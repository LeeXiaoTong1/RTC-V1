"""Training-only fusion batching; checkpoint keys and evaluation are unchanged."""
import torch
from w2v_v3.model import Detector as ReferenceDetector, MultiConvHead, mask_frames


class Detector(ReferenceDetector):
    """Padded training with the native HF 4.38.2 attention and convolution masks.

    This frontend masks convolution input in every Conformer layer and masks
    invalid attention keys. The MultiConv blocks and pool also mask invalid
    positions. Therefore right padding preserves valid-frame mathematics; it is
    never treated as recorded silence. Evaluation retains exact-length batches.
    """
    supports_padded_training = True

    def forward(self, features, mask):
        if not self.training:
            return super().forward(features, mask)
        if features.ndim != 3 or mask.ndim != 2 or features.shape[:2] != mask.shape:
            raise ValueError('Expected [B,T,D] acoustic features and [B,T] prefix mask')
        if features.shape[1] < 1 or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('A binary prefix mask and at least one frame are required')
        valid = mask.bool()
        if not bool(valid.any(1).all()) or bool((~valid[:, :-1] & valid[:, 1:]).any()):
            raise ValueError('Each waveform requires one nonempty valid prefix')
        # Caller-provided padding content cannot leak through feature projection.
        # This is padding only; no valid frame is removed or re-normalized.
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        output = self.backbone(input_features=features, attention_mask=mask,
                               output_hidden_states=True, return_dict=True)
        return self.head(output.hidden_states, mask)


def microbatches(examples, size=4, frame_budget=1600, max_padding_ratio=1.5):
    """Group nearby native lengths without dropping or truncating any waveform.

    Sorting is local to the existing logical batch. Its source count, class
    weights, and single optimizer update are unchanged. A single recording
    longer than the frame budget is emitted intact, as in the reference path.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or frame_budget < 1:
        raise ValueError('Positive integer microbatch size and frame budget required')
    if not 1 <= max_padding_ratio < float('inf'):
        raise ValueError('max_padding_ratio must be finite and at least one')
    for ex in examples:
        f, m = ex['features'], ex['mask']
        if f.ndim != 3 or f.shape[0] != 1 or f.shape[:2] != m.shape:
            raise ValueError('Each example requires one [1,T,D] waveform and [1,T] mask')
        if f.shape[1] < 1 or not bool((m == 1).all()):
            raise ValueError('Input examples must contain only valid native frames')
    ordered = sorted(range(len(examples)), key=lambda i: examples[i]['features'].shape[1])

    def emit(indices):
        waves = [examples[i]['features'].squeeze(0) for i in indices]
        features = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True, padding_value=0.)
        lengths = torch.tensor([w.shape[0] for w in waves], device=features.device)
        mask = (torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]).long()
        return indices, features, mask

    selected = []
    for i in ordered:
        length = examples[i]['features'].shape[1]
        if selected:
            shortest = examples[selected[0]]['features'].shape[1]
            if (len(selected) >= size or length * (len(selected) + 1) > frame_budget
                    or length / shortest > max_padding_ratio):
                yield emit(selected)
                selected = []
        selected.append(i)
    if selected:
        yield emit(selected)


class FusedMultiConvHead(MultiConvHead):
    """Batch independent layer projections; preserve the original layer sum order.

    There is no dropout in the projection/SwiGLU stage. Grouping its matrix
    products therefore consumes no extra RNG and changes no intended objective.
    GEMM rounding can differ slightly, especially with BF16. Evaluation always
    uses the original implementation, including fixed-Dev checkpoint selection.
    """
    def forward(self, hidden_states, mask):
        if not self.training:
            return super().forward(hidden_states, mask)
        if not hidden_states or mask.ndim != 2 or not bool(mask.bool().any(1).all()):
            raise ValueError('Nonempty hidden states and valid frames are required')
        for h in hidden_states:
            if h.shape[:2] != mask.shape or h.shape[-1] != self.config.input_dim:
                raise ValueError('SSL hidden-state shape differs from MultiConv input')
        x = None
        chunk_layers = self.fusion_chunk_layers
        for start in range(0, len(hidden_states), chunk_layers):
            chunk = hidden_states[start:start + chunk_layers]
            joined = chunk[0] if len(chunk) == 1 else torch.cat(chunk, dim=0)
            projected = self.gating(self.projection(joined))
            # Do not reduce across layers with sum(): preserve the checkpoint's
            # sequential accumulation order even when projected in groups.
            for z in projected.split(mask.shape[0], dim=0):
                x = z if x is None else x + z
        x = mask_frames(x, mask)
        frames = []
        for block in self.blocks:
            x = block(x, mask)
            frames.append(x)
        stats, block_means = self.pool(frames, mask)
        return self.classifier(stats), block_means


def install_fast_fusion(model, chunk_layers=5):
    """Install before training; retain all Parameter objects, keys, and RNG state."""
    if isinstance(chunk_layers, bool) or not isinstance(chunk_layers, int) or chunk_layers < 1:
        raise ValueError('fusion chunk_layers must be a positive integer')
    if type(model.head) not in (MultiConvHead, FusedMultiConvHead):
        raise ValueError('Expected the checkpoint-compatible MultiConv head')
    # Both classes have the same layout. This avoids constructing/reinitializing
    # modules, preserves optimizer parameter identity, and consumes no RNG.
    model.head.__class__ = FusedMultiConvHead
    model.head.fusion_chunk_layers = chunk_layers
    return model


def install_runtime(model, chunk_layers=5):
    """Upgrade a loaded reference Detector without initialization or new weights."""
    if type(model) not in (ReferenceDetector, Detector):
        raise ValueError('Expected a checkpoint-compatible w2v-BERT MultiConv Detector')
    model.__class__ = Detector
    return install_fast_fusion(model, chunk_layers)

