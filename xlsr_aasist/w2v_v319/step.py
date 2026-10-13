"""One shared full-wave forward per view; fixed group and edge loss budgets."""
from collections import defaultdict
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from w2v_v315.step import clear_features, _grad_norm
from w2v_v3151.step import triplet_batches
from .data import validate_batch
from .objectives import fusion_frames


def validate_examples(rows, cfg):
    # Share the authoritative data contract: unique sources within the logical
    # batch, exact quotas, no raw Offline classification, no invented bridge.
    if 'group_quotas' in cfg:
        return validate_batch(rows, cfg['source_batch'], cfg)
    occurrences = defaultdict(list)
    masses = defaultdict(float)
    for row in rows:
        weight, source_weight = row['ce_weight'], row['source_weight']
        if (row.get('split') != 'train' or not math.isfinite(weight) or weight < 0
                or not math.isfinite(source_weight) or source_weight <= 0):
            raise ValueError('Only Train with finite positive source budgets is allowed')
        masses[(row['language'], row['label'])] += weight
        occurrences[row['pair_occurrence']].append(row)
    if (len(occurrences) != cfg['source_batch']
            or set(masses) != {('en', 0), ('en', 1), ('zh', 0), ('zh', 1)}
            or any(abs(mass-.25) > 1e-7 for mass in masses.values())):
        raise ValueError('Logical source/group CE budget differs')
    identity_keys = ('source_id', 'group_id', 'source_sha256', 'language', 'label')
    role_masses = dict(online=cfg.get('online_ce_mass', .3),
                       reference=cfg.get('reference_ce_mass', .1), noisy=cfg.get('main_noisy_ce_mass', .6))
    for group in occurrences.values():
        if len(group) != 3 or {r['role'] for r in group} != set(role_masses):
            raise ValueError('Every source needs online/reference/noisy views')
        if len({tuple(r.get(k) for k in identity_keys) for r in group}) != 1:
            raise ValueError('Paired views have different source identities')
        source = group[0]['source_weight']
        for row in group:
            if (abs(row['source_weight']-source) > 1e-10
                    or abs(row['ce_weight']-source*role_masses[row['role']]) > 1e-8):
                raise ValueError('Source/role CE mass differs')
    return occurrences


def _gradient_comparison(ce, auxiliary, model):
    targets = [('fusion', model.head.projection.weight)]
    encoder = [p for p in model.backbone.encoder.layers[-1].parameters() if p.requires_grad]
    if encoder:
        targets.append(('last_encoder_layer', encoder[0]))
    parameters = [p for _, p in targets]
    cg = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
    ag = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
    result = {}
    for (name, _), a, b in zip(targets, cg, ag):
        cn = float(a.float().norm()) if a is not None else 0.
        an = float(b.float().norm()) if b is not None else 0.
        cosine = None
        if a is not None and b is not None and cn > 0 and an > 0:
            cosine = float((a.float().reshape(-1)@b.float().reshape(-1))/(cn*an))
        result[name] = dict(parameter_scope='projection.weight' if name=='fusion' else 'first trainable parameter of last encoder layer',
                            ce_norm=cn, auxiliary_norm=an,
                            auxiliary_to_ce=an/cn if cn > 0 else None, cosine=cosine)
        if not all(math.isfinite(x) for x in (cn, an)) or (cosine is not None and not math.isfinite(cosine)):
            raise FloatingPointError('Nonfinite auxiliary gradient audit')
    return result


def train_step(model, auxiliary, optimizer, examples, cfg, warm=1., audit=False):
    if not math.isfinite(warm) or not 0 <= warm <= 1:
        raise ValueError('Consistency multiplier must be finite in [0, 1]')
    model.train()
    auxiliary.train()
    occurrences = validate_examples(examples, cfg)
    order = {'online': 0, 'reference': 1, 'noisy': 2}
    groups = [sorted(rows, key=lambda r: order[r['role']]) for rows in occurrences.values()]
    optimizer.zero_grad(set_to_none=True)
    stats = dict(classification_loss=0., time_loss=0., structure_loss=0.,
        weighted_time_loss=0., weighted_structure_loss=0., total_loss=0.,
        bridge_time_loss=0., bridge_structure_loss=0., matched_time_loss=0., matched_structure_loss=0.,
        bridge_pairs=0, matched_pairs=0, eligible_pairs=0, maximum_example_ce=0.,
        maximum_padded_frames=0, physical_forwards=0, bridge_valid_frame_fraction=0.,
        matched_valid_frame_fraction=0., bridge_source_mass=0., matched_source_mass=0.,
        bridge_skipped_missing_online=0, bridge_skipped_time_incompatible=0)
    active = warm > 0 and (cfg['tfcl_time_weight'] > 0 or cfg['tfcl_structure_weight'] > 0)
    context = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if cfg['amp'] == 'bf16' else nullcontext()
    checked = False
    try:
        for rows, x, mask in triplet_batches(groups, cfg['microbatch'], cfg['frame_budget']):
            with fusion_frames(model) as capture, context():
                z, _ = model(x.to(cfg['device']), mask.to(cfg['device']))
                if len(capture) != 1:
                    raise ValueError('Expected one pre-MultiConv fusion feature tensor')
                features = capture[0]
                labels = torch.tensor([r['label'] for r in rows], device=z.device)
                mass = torch.tensor([r['ce_weight'] for r in rows], device=z.device)
                ce = F.cross_entropy(z.float(), labels, reduction='none')
                ce_loss = (ce*mass).sum()
                stats['classification_loss'] += float(ce_loss.detach())
                stats['maximum_example_ce'] = max(stats['maximum_example_ce'], float(ce.detach().max()))
                aux_loss = features[:0].float().sum()
                if active:
                    left, right, lm, rm, kinds, source_weights = [], [], [], [], [], []
                    for j in range(0, len(rows), 3):
                        if cfg['tfcl_bridge_mass'] > 0 and not rows[j].get('online_pair_eligible', False):
                            missing = not rows[j].get('official_online', rows[j].get('condition') == 'online')
                            stats['bridge_skipped_missing_online' if missing else 'bridge_skipped_time_incompatible'] += 1
                        for kind, a, b, eligible in (
                            ('bridge', j, j+1, rows[j].get('online_pair_eligible', False)),
                            ('matched', j+1, j+2, rows[j+2].get('noise_pair_eligible', rows[j+2].get('pair_eligible', False)))):
                            edge_budget = cfg['tfcl_bridge_mass' if kind == 'bridge' else 'tfcl_matched_mass']
                            if not eligible or edge_budget <= 0:
                                continue
                            am = torch.as_tensor(rows[a]['aux_valid'], device=features.device, dtype=torch.bool)
                            bm = torch.as_tensor(rows[b]['aux_valid'], device=features.device, dtype=torch.bool)
                            left.append(features[a, :len(am)]); right.append(features[b, :len(bm)])
                            lm.append(am); rm.append(bm); kinds.append(kind)
                            source_weights.append(rows[a]['source_weight'])
                    if left:
                        temporal, structure, valid, details = auxiliary.forward_batch(
                            left, right, lm, rm, kinds=kinds, return_diagnostics=True,
                            compute_time=cfg['tfcl_time_weight'] > 0,
                            compute_structure=cfg['tfcl_structure_weight'] > 0)
                        weights = torch.tensor(source_weights, device=features.device)
                        edge_mass = torch.tensor([cfg['tfcl_bridge_mass' if k == 'bridge' else 'tfcl_matched_mass']
                                                  for k in kinds], device=features.device)
                        ta, sd = (temporal*weights*edge_mass).sum(), (structure*weights*edge_mass).sum()
                        wt, ws = warm*cfg['tfcl_time_weight']*ta, warm*cfg['tfcl_structure_weight']*sd
                        aux_loss = wt+ws
                        stats['time_loss'] += float(ta.detach()); stats['structure_loss'] += float(sd.detach())
                        stats['weighted_time_loss'] += float(wt.detach()); stats['weighted_structure_loss'] += float(ws.detach())
                        for kind in ('bridge', 'matched'):
                            chosen = torch.tensor([k == kind for k in kinds], device=features.device)
                            stats[kind+'_time_loss'] += float((temporal[chosen]*weights[chosen]).sum().detach())
                            stats[kind+'_structure_loss'] += float((structure[chosen]*weights[chosen]).sum().detach())
                            stats[kind+'_pairs'] += int((valid & chosen).sum())
                            for detail, source_weight in zip(details, source_weights):
                                if detail['kind'] == kind and detail['eligible']:
                                    stats[kind+'_source_mass'] += source_weight
                                    stats[kind+'_valid_frame_fraction'] += source_weight*detail['valid_frame_fraction']
                        if audit and not checked and any(d['eligible'] for d in details):
                            stats['gradient_alignment'] = _gradient_comparison(ce_loss, aux_loss, model)
                            stats['gradient_alignment_scope'] = 'first_eligible_physical_batch'
                            stats['aux_gradient_audit'] = {k: v['auxiliary_norm'] for k, v in stats['gradient_alignment'].items()}
                            checked = True
                loss = ce_loss+aux_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite CE/consistency loss; update was not applied')
            stats['total_loss'] += float(loss.detach())
            loss.backward()
            clear_features(model)
            stats['maximum_padded_frames'] = max(stats['maximum_padded_frames'], x.shape[0]*x.shape[1])
            stats['physical_forwards'] += 1
            del features, z, loss, aux_loss
        stats['eligible_pairs'] = stats['matched_pairs']
        stats['aux_parameter_grad_norm'] = _grad_norm(auxiliary.parameters())
        if audit:
            stats['module_gradient_norms'] = {g.get('name', str(i)): _grad_norm(g['params'])
                                              for i, g in enumerate(optimizer.param_groups)}
        params = [p for g in optimizer.param_groups for p in g['params']]
        norm = torch.nn.utils.clip_grad_norm_(params, cfg['max_grad_norm'])
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite gradient; update was not applied')
        stats['gradient_norm'] = float(norm)
        stats['gradient_clip_scale'] = min(1., cfg['max_grad_norm']/max(float(norm), 1e-12))
        optimizer.step()
        return stats
    finally:
        clear_features(model)
