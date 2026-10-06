"""Bound confidence/representation scale without a second encoder or audio cache."""
import math
from collections import defaultdict

import numpy as np
import torch


class StabilityStop(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('Representation stability limit exceeded; previous committed state preserved')


def scale_loss(features, logits, rows, weights, cfg):
    h, z = features.float(), logits.float()
    if not bool(torch.isfinite(h).all() and torch.isfinite(z).all()):
        raise StabilityStop(dict(stage='training', reason='nonfinite_features_or_logits'))
    reference = h.new_tensor([r['reference_rms'] for r in rows])
    ceiling = h.new_tensor([r['margin_ceiling'] for r in rows])
    if not bool(torch.isfinite(reference).all() and torch.isfinite(ceiling).all()
                and (reference > 0).all() and (ceiling > 0).all()):
        raise ValueError('Scale references must be positive')
    rms = torch.linalg.vector_norm(h, dim=-1)/math.sqrt(h.shape[-1])
    ratio = rms/reference
    margin_ratio = (z[:, 0]-z[:, 1]).abs()/ceiling
    if not bool(torch.isfinite(ratio).all() and torch.isfinite(margin_ratio).all()):
        raise StabilityStop(dict(stage='training',reason='nonfinite_scale_arithmetic'))
    norms = (ratio-cfg['feature_ratio_high']).clamp_min(0).square()
    norms = norms+(cfg['feature_ratio_low']-ratio).clamp_min(0).square()
    scores = (margin_ratio-1).clamp_min(0).square()
    # This check happens BEFORE the optimizer update, including on microbatches.
    bad = (ratio > cfg['hard_feature_ratio']) | (margin_ratio > cfg['hard_margin_ratio'])
    if bool(bad.any()):
        indices = bad.nonzero().flatten().tolist()
        raise StabilityStop(dict(stage='training', reason='feature_or_margin_scale',
            rows=[dict(id=rows[i]['id'], condition=rows[i]['condition'], language=rows[i]['language'],
                       label=rows[i]['label'], feature_ratio=float(ratio[i].detach()),
                       margin_ratio=float(margin_ratio[i].detach())) for i in indices]))
    return (weights*(norms+scores)).sum(), dict(
        maximum_feature_ratio=float(ratio.detach().max()),
        maximum_margin_ratio=float(margin_ratio.detach().max()))


def summarize(rows, feature_rms, stats_rms, logits, adapter_ratio):
    arrays = dict(feature_rms=np.asarray(feature_rms,dtype=np.float64),
        stats_rms=np.asarray(stats_rms,dtype=np.float64),
        abs_margin=np.abs(np.asarray(logits,dtype=np.float64)[:, 0]-np.asarray(logits,dtype=np.float64)[:, 1]),
        adapter_ratio=np.asarray(adapter_ratio,dtype=np.float64))
    if any(a.shape != (len(rows),) or not np.isfinite(a).all() for a in arrays.values()):
        raise StabilityStop(dict(stage='validation',reason='nonfinite_or_unaligned_diagnostics'))
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[f'{r["condition"]}/{r["language"]}/{r["label"]}'].append(i)
    output = {}
    for key, indices in groups.items():
        output[key] = {name:dict(zip(('p50','p95','p99','max'),
            np.quantile(values[indices],[.5,.95,.99,1]).tolist())) for name,values in arrays.items()}
    return dict(rows=len(rows),groups=output)


def compare_diagnostics(baseline, current, cfg):
    if baseline['rows'] != current['rows'] or baseline['groups'].keys() != current['groups'].keys():
        raise ValueError('Stability diagnostics must use the identical validation inventory')
    reasons, ratios = [], {}
    for group, original in baseline['groups'].items():
        for name in ('feature_rms','stats_rms','abs_margin'):
            floor = cfg['margin_soft_floor'] if name == 'abs_margin' else cfg['feature_rms_floor']
            limit = cfg['validation_margin_ratio'] if name == 'abs_margin' else cfg['validation_norm_ratio']
            for quantile in ('p50','p95','p99','max'):
                key = f'{group}/{name}/{quantile}'
                ratio = current['groups'][group][name][quantile]/max(floor,original[name][quantile])
                ratios[key] = ratio
                if not math.isfinite(ratio) or ratio > limit:
                    reasons.append(key+'_scale_exceeded')
    return dict(stable=not reasons,reasons=reasons,ratios=ratios)
