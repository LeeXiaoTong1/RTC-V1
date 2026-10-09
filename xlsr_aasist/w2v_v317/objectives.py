"""Same bidirectional TFCL definitions; inputs are the final forensic frames."""
from w2v_v3161.objectives import TFCL


def clear_features(model):
    # Frames are returned per forward, never retained in module attributes.
    pass
