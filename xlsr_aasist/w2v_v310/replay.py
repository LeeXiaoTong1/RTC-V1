"""Compare zero-adapter and original paths under the same numerical execution.

Historical frozen vectors are not a same-runtime forward reference: older cache
producers did not pin cuDNN TF32 and used different batch boundaries. Their
checksums/provenance still matter, but their logits cannot establish whether a
new adapter changed the original function. Measure the original detector again
once; persist only its small scores and probe diagnostics for exact resumption.
"""
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from w2v_v3.model import Detector as ReferenceDetector, MultiConvHead
from w2v_v32.model import Detector as RuntimeDetector, FusedMultiConvHead
from w2v_v39.common import atomic_json, digest, read_json, verify_files, announce
from w2v_v39.metrics import measure
from .model import FeatureClassifier
from .probes import run_probes
from .state import identity

SCHEMA = 'rtc_v310_same_runtime_baseline_v1'
RTOL, ATOL = 3e-5, 1e-3  # Preserve the original zero-adapter tolerance.


@contextmanager
def fp32_inference(device):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


@contextmanager
def original_path(model):
    """Use the exact reference classes/Sequential without a second SSL copy."""
    classifier = model.head.classifier
    if (type(model) is not RuntimeDetector or type(model.head) is not FusedMultiConvHead
            or not isinstance(classifier, FeatureClassifier)):
        raise ValueError('Expected the original-compatible V3.10 detector')
    model_class, head_class, training = type(model), type(model.head), model.training
    model.eval()
    classifier.features = None
    original = nn.Sequential(*(classifier[i] for i in range(3))).eval()
    try:
        model.__class__ = ReferenceDetector
        model.head.__class__ = MultiConvHead
        model.head.classifier = original
        yield model
    finally:
        model.head.classifier = classifier
        model.head.__class__ = head_class
        model.__class__ = model_class
        classifier.features = None
        model.train(training)


def delta(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or not len(a):
        raise ValueError('Replay requires aligned nonempty binary logits')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError('Replay logits must be finite')
    difference = np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(rows=len(a), max_logit_delta=float(difference.max()), mean_logit_delta=float(difference.mean()),
                decision_changes=int(np.sum(a.argmax(1)!=b.argmax(1))),
                allclose=bool(np.allclose(a,b,rtol=RTOL,atol=ATOL)), rtol=RTOL, atol=ATOL)


def replay_indices(rows, limit=32):
    # Include both labels/languages and each condition, rather than only the first online rows.
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        groups[(row['condition'],row['language'],row['label'])].append(i)
    chosen = []
    depth = 0
    while len(chosen) < min(limit,len(rows)):
        for values in groups.values():
            if depth < len(values):
                chosen.append(values[depth])
                if len(chosen)==min(limit,len(rows)):
                    break
        depth += 1
    return sorted(chosen)


def check_initial(model, dev, cfg, run, infer):
    path = Path(run)/'startup_replay.json'
    output = model.head.classifier.adapter[-1]
    zero = bool(torch.count_nonzero(output.weight)==0 and torch.count_nonzero(output.bias)==0)
    if not zero:
        atomic_json(path,dict(status='failed',reason='adapter_not_zero_initialized'))
        raise ValueError('Adapter is not zero-initialized; inspect startup_replay.json')
    indices = replay_indices(dev['rows'])
    selected = [dev['rows'][i] for i in indices]
    # Identical row order, feature extraction, microbatches and FP32 policy in both calls.
    with original_path(model):
        reference,_ = infer(model,selected,cfg,'V3.10 original-path startup replay')
    candidate,_ = infer(model,selected,cfg,'V3.10 zero-adapter startup replay')
    comparison = delta(reference,candidate)
    old = delta(np.asarray(dev['logits'])[indices],reference)
    report = dict(status='passed' if comparison['allclose'] and comparison['decision_changes']==0 else 'failed',
        adapter_zero_initialized=zero, original_vs_adapter=comparison, historical_cache_vs_original=old,
        sample_indices=indices, samples=[{k:r[k] for k in ('id','condition','language','label')} for r in selected],
        evaluation=dict(dtype='float32',autocast=False,matmul_tf32=False,cudnn_tf32=False,
                        feature_batch=cfg['feature_batch'],microbatch=cfg['microbatch'],frame_budget=cfg['frame_budget']),
        historical_cache_role='provenance and drift diagnostic, not an adapter-equivalence assertion')
    atomic_json(path,report)
    print(f'V310_SAME_RUNTIME_REPLAY max_logit_delta={comparison["max_logit_delta"]:.8g} '
          f'decision_changes={comparison["decision_changes"]}; '
          f'historical_cache_delta={old["max_logit_delta"]:.8g}',flush=True)
    if report['status'] != 'passed':
        raise ValueError('Zero-adapter differs from original under identical execution: '
                         f'max_logit_delta={comparison["max_logit_delta"]:.8g}, '
                         f'decision_changes={comparison["decision_changes"]}; see startup_replay.json')
    return report


def _rows_identity(dev_rows, probe_rows, split):
    content = json.dumps(dict(dev=dev_rows,probe=probe_rows,split=split),sort_keys=True,allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def baseline(model, dev, probe_rows, split, cfg, run, infer):
    run = Path(run)
    marker = run/'baseline_complete.json'
    signature = _rows_identity(dev['rows'],probe_rows,split)
    if marker.is_file():
        saved = read_json(marker)
        if saved.get('schema')!=SCHEMA or saved.get('identity')!=identity(cfg) or saved.get('rows_identity')!=signature:
            raise ValueError('Committed baseline belongs to another run, execution policy or data inventory')
        verify_files({str(run/name):sha for name,sha in saved['files'].items()})
        with np.load(run/'dev_scores_baseline.npz',allow_pickle=False) as data:
            logits = data['logits'].copy()
        delta(logits,logits)  # Shape/finite checks, in addition to the committed file hash.
        if len(logits)!=len(dev['rows']):
            raise ValueError('Committed baseline row count changed')
        print('V310_BASELINE_REUSE=same-runtime original; no repeated full baseline inference',flush=True)
        return logits,read_json(run/'baseline_metrics.json'),read_json(run/'baseline_probes.json')
    if (run/'last.pt').is_file():
        raise ValueError('Cannot resume trained state without its committed same-runtime baseline')
    check_initial(model,dev,cfg,run,infer)
    announce('V3.10 measuring original best once under the same FP32 validation policy')
    with original_path(model):
        logits,_ = infer(model,dev['rows'],cfg,'V3.10 original baseline Dev')
        _,features = infer(model,probe_rows,cfg,'V3.10 original baseline real probes',capture=True)
    metrics = measure(dev['rows'],logits,target=cfg['matched_fake_recall'])
    probes = run_probes(features,probe_rows,split,cfg)
    historical = measure(dev['rows'],np.asarray(dev['logits']),target=cfg['matched_fake_recall'])
    drift = dict(logits=delta(np.asarray(dev['logits']),logits),historical_metrics=historical,current_metrics=metrics,
                 note='Same checkpoint measured under current policy. Numerical drift is not training improvement.')
    atomic_json(run/'baseline_metrics.json',metrics)
    atomic_json(run/'baseline_probes.json',probes)
    atomic_json(run/'baseline_cache_drift.json',drift)
    np.savez_compressed(run/'dev_scores_baseline.npz',logits=logits)
    names = ('baseline_metrics.json','baseline_probes.json','baseline_cache_drift.json',
             'dev_scores_baseline.npz','startup_replay.json')
    atomic_json(marker,dict(schema=SCHEMA,identity=identity(cfg),rows_identity=signature,
                           files={name:digest(run/name) for name in names}))
    print(f'V310_BASELINE historical_weighted={100*historical["weighted_f1"]:.3f} '
          f'current_weighted={100*metrics["weighted_f1"]:.3f}; comparison uses current baseline',flush=True)
    return logits,metrics,probes
