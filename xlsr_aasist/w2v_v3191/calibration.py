"""Fixed score protocol: no fitting on either Select or Audit."""
from pathlib import Path
from .common import digest, atomic_json


def write_identity(run, state, *, paused=False):
    checksum = digest(Path(run)/"last.pt")
    for kind in (("best",) if paused else ("best", "last")):
        tag = state[kind+"_tag"]
        if tag:
            atomic_json(Path(run)/f"calibration_{kind}.json", dict(
                checkpoint_sha256=checksum, selected=tag, scale=1., bias=0.,
                enabled=False, fitted=False, selected_candidate="fixed_identity",
                classification_threshold=.5,
                note="Raw model probabilities; no threshold/affine fitting on Select or Audit"))
    return checksum


def validate_identity(value):
    if (value.get("scale") != 1. or value.get("bias") != 0.
            or value.get("fitted") is not False
            or value.get("selected_candidate") != "fixed_identity"):
        raise ValueError("V3.19.1 exports only fixed raw probabilities; affine calibration is forbidden")
