"""Bound full speech and repeated short inputs before starting GPU fitting."""
from w2v_v3191.resources import audit_training as inherited_audit, _samples


def audit_training(rows, dev, cfg):
    maximum_short = 3 * 16000  # largest registered screening candidate
    representatives = {}
    for raw in rows:
        row = dict(raw)
        samples, _ = _samples(row)
        row["output_samples"] = max(samples, maximum_short)
        row["source_id"] = row["canonical_source"]
        key = row["canonical_source"], row["condition"]
        if key not in representatives or row["output_samples"] > representatives[key]["output_samples"]:
            representatives[key] = row
    evaluation = []
    for raw in dev:
        samples, _ = _samples(raw)
        evaluation.append(dict(raw, output_samples=max(samples, maximum_short)))
    result = inherited_audit(list(representatives.values()), evaluation, cfg)
    result["v5_notes"] = (
        "Bounds include 3-second input repetition and the slowest inherited speed augmentation. "
        "RTC context is trimmed before encoding. Canonical duplicates use the longest member. "
        "Dev official references are conservatively counted even though Dev PCL is disabled.")
    return result
