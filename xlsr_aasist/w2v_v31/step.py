"""Source-normalized full/short CE and full-view-only global CKA."""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from w2v_aasist.runtime import amp_context
from w2v_v3.model import diversity_cka, microbatches
from w2v_v3.step import ActivationOffload, _components, predict


def view_components(examples, weights, noisy_class_weights, device, noisy_weight):
    """Map one loss budget per original row to its one or two expanded views."""
    if not examples:
        raise ValueError('A nonempty logical batch is required')
    groups = {}
    for i, ex in enumerate(examples):
        if not isinstance(ex.get('source_group'), str) or not ex['source_group']:
            raise ValueError('Each view must identify its original source_group')
        if ex.get('view') not in ('full', 'short'):
            raise ValueError('Each view must be full or short')
        weight = float(ex.get('view_weight', float('nan')))
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError('Each view requires a finite weight in (0,1]')
        indices = groups.setdefault(ex['source_group'], [])
        indices.append(i)
    full_indices, group_numbers = [], {}
    for group, indices in groups.items():
        rows = [examples[i] for i in indices]
        full = [i for i in indices if examples[i]['view'] == 'full']
        if len(full) != 1 or len(rows) > 2 or len({r['view'] for r in rows}) != len(rows):
            raise ValueError('Each source_group requires exactly one full and at most one short view')
        # Separate versions/exposures remain separate groups, not extra positive
        # class weight or repeated CKA samples in the original group's budget.
        if any(r['label'] != rows[0]['label'] or bool(r['noisy']) != bool(rows[0]['noisy'])
               or r.get('id') != rows[0].get('id') for r in rows):
            raise ValueError('Views of a source_group disagree on source, label or supervision component')
        if not math.isclose(sum(float(r['view_weight']) for r in rows), 1., rel_tol=0, abs_tol=1e-7):
            raise ValueError('View weights must sum to one for each original row')
        group_numbers[group] = len(full_indices)
        full_indices.append(full[0])
    source_rows = [examples[i] for i in full_indices]
    _, _, ow, nw, source_coefficients, n, s = _components(
        source_rows, weights, noisy_class_weights, device, noisy_weight)
    numbers = torch.tensor([group_numbers[e['source_group']] for e in examples], device=device)
    view_weights = torch.tensor([float(e['view_weight']) for e in examples], dtype=torch.float32, device=device)
    coefficients = source_coefficients[numbers] * view_weights
    labels = torch.tensor([e['label'] for e in examples], dtype=torch.long, device=device)
    noisy = torch.tensor([e['noisy'] for e in examples], dtype=torch.bool, device=device)
    full_mask = torch.tensor([e['view'] == 'full' for e in examples], dtype=torch.bool, device=device)
    return labels, noisy, ow, nw, coefficients, n, s, full_mask, view_weights


def loss_function(logits, blocks, examples, weights, noisy_weight=.5, cka_weight=.01,
                  noisy_class_weights=None):
    """Reference objective; source counts remain fixed when view counts differ."""
    labels, noisy, ow, nw, coefficients, n, s, full, vw = view_components(
        examples, weights, noisy_class_weights, logits.device, noisy_weight)
    ce = F.cross_entropy(logits.float(), labels, reduction='none')
    classification = (ce*coefficients).sum()
    ordinary = (ce[~noisy]*ow[labels[~noisy]]*vw[~noisy]).sum()/n
    simulated = (ce[noisy]*nw[labels[noisy]]*vw[noisy]).sum()/s if s else ce.sum()*0
    cka = diversity_cka(blocks[full]) if cka_weight else classification*0
    return classification+cka_weight*cka, {'ce':classification, 'ordinary_ce':ordinary,
                                        'noisy_ce':simulated, 'cka':cka}


def supervised_step(model, examples, optimizer, weights, device, amp='bf16',
                    noisy_weight=.5, cka_weight=.01, grad_clip=1., microbatch=4,
                    frame_budget=1600, noisy_class_weights=None,
                    offload_activations=True, activation_budget_gib=0.):
    """One forward per view, bounded activation storage, one optimizer update."""
    if not math.isfinite(cka_weight) or cka_weight < 0 or not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError('Finite nonnegative CKA weight and positive gradient clip required')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Exact variable-length microbatching requires no BatchNorm')
    device = torch.device(device)
    labels, noisy, ow, nw, coefficients, n, s, full, vw = view_components(
        examples, weights, noisy_class_weights, device, noisy_weight)
    optimizer.zero_grad(set_to_none=True)
    offload = ActivationOffload(model, activation_budget_gib) if offload_activations and device.type == 'cuda' else None
    logits, full_blocks, order = [], [], []
    stats = {'ordinary_ce':0., 'noisy_ce':0., 'full_ce_contribution':0., 'short_ce_contribution':0.}
    classification = torch.zeros((), device=device)
    with offload if offload is not None else nullcontext():
        for indices, features, mask in microbatches(examples, microbatch, frame_budget):
            with amp_context(device, amp):
                z, h = model(features.to(device), mask.to(device))
            if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(h).all()):
                raise FloatingPointError('Non-finite model output; optimizer has not advanced')
            ce = F.cross_entropy(z.float(), labels[indices], reduction='none')
            contribution = ce*coefficients[indices]
            local_loss = contribution.sum()
            if not bool(torch.isfinite(local_loss)):
                raise FloatingPointError('Non-finite classification loss; optimizer has not advanced')
            ln, ly, lf, lvw = noisy[indices], labels[indices], full[indices], vw[indices]
            stats['ordinary_ce'] += float((ce.detach()[~ln]*ow[ly[~ln]]*lvw[~ln]).sum())/n
            if s:
                stats['noisy_ce'] += float((ce.detach()[ln]*nw[ly[ln]]*lvw[ln]).sum())/s
            stats['full_ce_contribution'] += float(contribution.detach()[lf].sum())
            stats['short_ce_contribution'] += float(contribution.detach()[~lf].sum())
            logits.append(z.detach().float()); order.extend(indices)
            if cka_weight:
                classification = classification+local_loss
                if bool(lf.any()):
                    full_blocks.append(h[lf].float())
            else:
                local_loss.backward()
                classification += local_loss.detach()
        cka = diversity_cka(torch.cat(full_blocks)) if cka_weight else classification.new_zeros(())
        loss = classification+cka_weight*cka
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite logical loss; optimizer has not advanced')
        if cka_weight:
            loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    scores = torch.cat(logits)[torch.argsort(torch.tensor(order, device=device))]
    return {**stats, 'ce':float(classification.detach()), 'cka':float(cka.detach()),
            'loss':float(loss.detach()), 'grad_norm':float(norm),
            'ordinary_sources':n, 'noisy_sources':s, 'full_views':n+s,
            'short_views':len(examples)-n-s, 'cka_full_views':n+s if cka_weight else 0,
            'activation_offload_gib':offload.bytes/1024**3 if offload else 0.,
            'activation_budget_gib':offload.limit/1024**3 if offload else 0.,
            'encoder_forward_microbatches':len(logits)}, scores
