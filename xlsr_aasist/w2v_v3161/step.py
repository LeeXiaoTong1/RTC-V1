"""All supervised views; two-sided soft TFCL without confidence/saliency gates."""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from w2v_v316_tfcl.data import validate_batch
from w2v_v316_tfcl.step import source_batches
from w2v_v315.step import clear_features
from .objectives import ssl_frames


def train_step(model, auxiliary, optimizer, examples, cfg, warm, audit=False):
    model.train(); auxiliary.train(); optimizer.zero_grad(set_to_none=True)
    groups = list(validate_batch(examples, cfg['source_batch'], cfg).values())
    stats = dict(classification_loss=0., time_loss=0., structure_loss=0.,
        weighted_time_loss=0., weighted_structure_loss=0., total_loss=0., maximum_example_ce=0.,
        bridge_pairs=0, matched_pairs=0, attempted_pairs=0, physical_forwards=0,
        groups={}, feature_gradient={})
    try:
        for rows, x, mask in source_batches(groups, cfg['microbatch'], cfg['frame_budget']):
            context = torch.autocast('cuda', dtype=torch.bfloat16) if cfg['amp']=='bf16' else nullcontext()
            with ssl_frames(model) as captured, context:
                logits, _ = model(x.to(cfg['device']), mask.to(cfg['device']))
                if len(captured) != 1:
                    raise ValueError('Expected exactly one SSL output per detector forward')
                features = captured[0]
                labels = torch.tensor([r['label'] for r in rows], device=logits.device)
                ce = F.cross_entropy(logits.float(), labels, reduction='none')
                weights = torch.tensor([r['ce_weight'] for r in rows], device=logits.device)
                ce_loss = (ce*weights).sum()
                aux_loss = features[:0].float().sum()
                stats['classification_loss'] += float(ce_loss.detach())
                stats['maximum_example_ce'] = max(stats['maximum_example_ce'], float(ce.detach().max()))
                edges, left, right, lm, rm = [], [], [], [], []
                if warm > 0:
                    for i, r in enumerate(rows):
                        if r['role'] != 'offline': continue
                        for j, target in enumerate(rows):
                            if target['pair_occurrence'] != r['pair_occurrence'] or target['role']=='offline': continue
                            # Pair identity is verified by the loader. Do not reject
                            # hard examples because the current detector is wrong.
                            a, b = len(r['aux_valid']), len(target['aux_valid'])
                            if a>features.shape[1] or b>features.shape[1]:
                                raise ValueError('SSL and native feature lengths differ')
                            left.append(features[i,:a]); right.append(features[j,:b])
                            lm.append(torch.as_tensor(r['aux_valid'])); rm.append(torch.as_tensor(target['aux_valid']))
                            edges.append((r['language'], r['label'], target['role']))
                if edges:
                    t, s, eligible = auxiliary.forward_batch(left, right, lm, rm)
                    # Fixed budgets per source and edge; missing official Online
                    # does not increase the remaining noisy edge's influence.
                    scale = .5/cfg['source_batch']
                    t, s = t.sum()*scale, s.sum()*scale
                    wt, ws = warm*cfg['tfcl_time_weight']*t, warm*cfg['tfcl_structure_weight']*s
                    aux_loss = wt+ws
                    for key, value in (('time_loss',t),('structure_loss',s),
                                       ('weighted_time_loss',wt),('weighted_structure_loss',ws)):
                        stats[key] += float(value.detach())
                    for (language, label, role), ok in zip(edges, eligible.tolist()):
                        key = f'{language}/{"fake" if label==0 else "real"}/{role}'
                        cell = stats['groups'].setdefault(key, dict(attempted=0, valid=0))
                        cell['attempted'] += 1; cell['valid'] += int(ok)
                        stats['attempted_pairs'] += 1
                        stats['bridge_pairs' if role=='online' else 'matched_pairs'] += int(ok)
                loss = ce_loss+aux_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite TFCL/CE: no optimizer update applied')
            if audit and not stats['feature_gradient']:
                cg = torch.autograd.grad(ce_loss, features, retain_graph=True, allow_unused=True)[0]
                ag = torch.autograd.grad(aux_loss, features, retain_graph=True, allow_unused=True)[0]
                stats['feature_gradient'] = dict(
                    ce_norm=float(cg.float().norm()) if cg is not None else 0.,
                    tfcl_norm=float(ag.float().norm()) if ag is not None else 0.)
            loss.backward(); clear_features(model)
            stats['total_loss'] += float(loss.detach()); stats['physical_forwards'] += 1
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'])
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite gradient: no optimizer update applied')
        stats['gradient_norm'] = float(norm)
        optimizer.step()
        return stats
    finally:
        clear_features(model)
