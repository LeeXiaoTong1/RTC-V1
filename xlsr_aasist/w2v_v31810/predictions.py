"""Small, pickle-free diagnostic outputs from already computed Dev logits."""
import hashlib
import json
from pathlib import Path
import numpy as np


def save_predictions(path, rows, logits, *, protocol_hash, protocol_content_hash,
                     checkpoint_tag, split):
    path = Path(path)
    if path.suffix != '.npz' or split not in ('select', 'audit') or not checkpoint_tag:
        raise ValueError('Prediction output needs an NPZ path, split and checkpoint tag')
    z = np.asarray(logits, dtype=np.float32)
    if z.shape != (len(rows), 2) or not np.isfinite(z).all():
        raise ValueError('Prediction logits must be finite FP32 rows in fake/real class order')
    identities = [(str(r['id']), str(r['canonical_source']), str(r['language']),
                   int(r['label']), str(r['validation_panel'])) for r in rows]
    if any(r[3] not in (0, 1) for r in identities):
        raise ValueError('Prediction labels must use fake=0, real=1')
    identity_hash = hashlib.sha256(json.dumps(identities, separators=(',', ':'),
                                             ensure_ascii=True).encode()).hexdigest()
    metadata = dict(schema='rtc_v31810_predictions_v1', split=split,
                    checkpoint_tag=str(checkpoint_tag), protocol_hash=protocol_hash,
                    protocol_content_sha256=protocol_content_hash, row_identity_sha256=identity_hash,
                    count=len(rows), class_order=['fake', 'real'], threshold=.5,
                    calibration='none', logits_dtype='float32')
    arrays = dict(logits=z, row_index=np.arange(len(rows), dtype=np.int64),
                  ids=np.asarray([r[0] for r in identities], dtype=np.str_),
                  canonical_source=np.asarray([r[1] for r in identities], dtype=np.str_),
                  language=np.asarray([r[2] for r in identities], dtype=np.str_),
                  label=np.asarray([r[3] for r in identities], dtype=np.int64),
                  panel=np.asarray([r[4] for r in identities], dtype=np.str_),
                  metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size, **metadata)


def ranking_metrics(labels, scores):
    """Fake-positive ROC; ties move together and EER linearly interpolates ROC."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Ranking metrics need finite scores matching the label vector')
    if not set(labels.tolist()) <= {0, 1}:
        raise ValueError('Ranking metrics expect fake=0, real=1')
    positives = int((labels == 0).sum()); negatives = int((labels == 1).sum())
    if not positives or not negatives:
        return dict(auc=None, eer=None)
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]; positive = labels[order] == 0
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered)-1]
    tpr = np.r_[0., np.cumsum(positive)[ends]/positives]
    fpr = np.r_[0., np.cumsum(~positive)[ends]/negatives]
    auc = float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])*.5))
    fnr = 1.-tpr; difference = fpr-fnr
    index = int(np.flatnonzero(difference >= 0)[0])
    if difference[index] == 0:
        eer = float(fpr[index])
    else:
        weight = -difference[index-1]/(difference[index]-difference[index-1])
        eer = float(fpr[index-1]+weight*(fpr[index]-fpr[index-1]))
    return dict(auc=auc, eer=eer)
