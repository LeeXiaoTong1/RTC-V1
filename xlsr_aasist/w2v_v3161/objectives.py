"""TFCL paper objectives at the full SSL output, with variable-duration masks.

Reference: https://arxiv.org/html/2607.17761v2 and JunXue-tech/TFCL/code/model.py.
The published implementation also uses bidirectional eight-head attention and
channel CKA after a shared temporal projection. Necessary differences: retain
whole variable-length audio; pool only the structure arm; average per-source
CKA rather than flattening unrelated sources across a hardware-dependent batch.
The latter equals the published channel-CKA definition at batch size one.
"""
from contextlib import contextmanager
from w2v_v3151.objectives import TFCL as MaskedTFCL, channel_cka


class TFCL(MaskedTFCL):
    def __init__(self, channels=1024, heads=8, bins=201):
        super().__init__(channels, heads, bins)
        # As in the released nn.Linear, learn a shared temporal projection.
        self.time_projection.reset_parameters()


@contextmanager
def ssl_frames(model):
    captured = []
    # Hook the backbone output once, not a checkpointed encoder sublayer that
    # can be replayed during backward. No additional encoder forward is needed.
    handle = model.backbone.register_forward_hook(
        lambda _module, _args, output: captured.append(output.last_hidden_state))
    try:
        yield captured
    finally:
        handle.remove()
        captured.clear()
