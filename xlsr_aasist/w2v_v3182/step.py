"""Exact logical-batch gradients through RNG-preserving feature/logit replay."""
from collections import Counter, defaultdict
import torch
from w2v_v318.step import schedule as base_schedule
from .common import capture_rng, restore_rng, require_finite
from .data import microbatches
from .sampling import validate_units
from .objective import classification, contrastive


def schedule(optimizer, cfg, epoch, step, steps):
    base_schedule(optimizer, cfg, epoch, step, steps)
    if epoch < cfg['warm_epochs']:
        return 0.
    progress = ((epoch-cfg['warm_epochs'])*steps+step+1)/steps
    return cfg['contrastive_weight']*min(1., progress/cfg['contrastive_ramp_epochs'])


def backward_objective(model, units, cfg, coefficient, replay=True):
    """Accumulate gradients; caller owns zero_grad, clipping and optimizer.step.

    Replay is only enabled with frozen BN statistics. The first pass keeps
    no encoder graphs. Gradients w.r.t. the small [views,2] logits and [views,160]
    features are cached, then injected into identical microbatch replays.
    No stale queue, detached teacher, or microbatch-local negative pool is used.
    """
    micros = [[r for u in m for r in u] for m in microbatches(units, cfg['microbatch'], cfg['frame_budget'])]
    rows = [r for m in micros for r in m]
    stats, raw_values, logits_values = {}, [], []
    if coefficient == 0:
        ce_value = 0.
        for micro in micros:
            logits = model([r['wave'] for r in micro])
            loss, raw, _ = classification(logits, micro, cfg['label_smoothing'])
            require_finite(loss, 'weighted CE'); loss.backward()
            ce_value += float(loss.detach())
            raw_values.append(raw.detach()); logits_values.append(logits.detach())
        stats.update(loss=ce_value, ce_objective=ce_value, pair_loss=0., pair_coefficient=0.,
                     weighted_pair_loss=0., replay_max_abs_error=0., encoder_forward_passes=len(micros))
    elif not replay:
        outputs = [model([r['wave'] for r in m], return_features=True) for m in micros]
        logits, h = (torch.cat([o[k] for o in outputs]) for k in (0,1))
        ce, raw, _ = classification(logits, rows, cfg['label_smoothing'])
        pair, detail = contrastive(h, units, cfg['contrastive_temperature'], cfg['negative_mass'])
        loss = ce+coefficient*pair; loss.backward()
        raw_values.append(raw.detach()); logits_values.append(logits.detach())
        stats.update(detail, loss=float(loss.detach()), ce_objective=float(ce.detach()), pair_loss=float(pair.detach()),
            pair_coefficient=coefficient, weighted_pair_loss=float((coefficient*pair).detach()), replay_max_abs_error=0.,
            encoder_forward_passes=len(micros))
    else:
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()):
            raise ValueError('Gradient replay requires frozen BN running statistics')
        rngs, outputs = [], []
        with torch.no_grad():
            for micro in micros:
                rngs.append(capture_rng())
                outputs.append(model([r['wave'] for r in micro], return_features=True))
        end_rng = capture_rng()
        logits = torch.cat([o[0] for o in outputs]).detach().requires_grad_()
        h = torch.cat([o[1] for o in outputs]).detach().requires_grad_()
        ce, raw, _ = classification(logits, rows, cfg['label_smoothing'])
        pair, detail = contrastive(h, units, cfg['contrastive_temperature'], cfg['negative_mass'])
        loss = ce+coefficient*pair
        require_finite(loss, 'combined objective')
        dz, dh = torch.autograd.grad(loss, (logits, h))
        max_error, offset = 0., 0
        try:
            for micro, rng, cached in zip(micros, rngs, outputs):
                restore_rng(rng)
                z, f = model([r['wave'] for r in micro], return_features=True)
                for actual, expected in ((z, cached[0]), (f, cached[1])):
                    max_error = max(max_error, float((actual.detach()-expected).abs().max()))
                    if not torch.allclose(actual.detach(), expected, rtol=.005, atol=.0005):
                        raise RuntimeError('Gradient replay differs from cached forward; check stochastic/stateful layers')
                stop = offset+len(micro)
                torch.autograd.backward((z, f), (dz[offset:stop], dh[offset:stop]))
                offset = stop
        finally:
            restore_rng(end_rng)
        raw_values.append(raw.detach()); logits_values.append(logits.detach())
        stats.update(detail, loss=float(loss.detach()), ce_objective=float(ce.detach()), pair_loss=float(pair.detach()),
            pair_coefficient=coefficient, weighted_pair_loss=float((coefficient*pair).detach()),
            replay_max_abs_error=max_error, encoder_forward_passes=2*len(micros))
    raw = torch.cat(raw_values); logits = torch.cat(logits_values)
    # Telemetry needs no graph. Transfer the small vectors once rather than
    # forcing a device synchronization for every scalar and repeated argmax.
    raw_cpu = raw.detach().cpu()
    raw_items = raw_cpu.tolist()
    predicted_labels = logits.argmax(dim=1).detach().cpu().tolist()
    cells = defaultdict(lambda: defaultdict(float)); processing = Counter()
    predicted = {'fake':[0,0], 'real':[0,0]}
    for i, row in enumerate(rows):
        condition = 'noisy' if row['role'].startswith('noisy') else row['role']
        cell = cells[f'{condition}/{row["language"]}/{row["label"]}']
        cell['raw_ce_sum'] += raw_items[i]; cell['views'] += 1
        cell['correct'] += int(predicted_labels[i] == row['label'])
        cell['ce_mass'] += row['ce_weight']
        p = predicted['fake' if row['label'] == 0 else 'real']
        p[0] += int(predicted_labels[i] == 0); p[1] += 1
        rec = row.get('augmentation', {}).get('recipe')
        if rec:
            processing[f'{row["language"]}/{row["label"]}/{rec["family"]}/{rec["codec"]}/{rec["noise_type"]}'] += 1
    stats.update(raw_ce_sum=float(raw_cpu.sum()), views=len(rows), max_example_ce=float(raw_cpu.max()),
        cells={k:dict(v) for k,v in cells.items()}, processing_counts=dict(processing),
        predicted_fake_by_true_class=predicted, unique_original_sources=len(units))
    return stats


def train_step(model, optimizer, units, cfg, coefficient):
    validate_units(units, cfg['stream_sources'])
    model.train(); optimizer.zero_grad(set_to_none=True)
    stats = backward_objective(model, units, cfg, coefficient)
    for family, accept in (('lora',lambda n:n.endswith(('.lora_a','.lora_b'))),
                           ('head',lambda n:n.startswith('head.'))):
        norms=[p.grad.detach().float().norm() for n,p in model.named_parameters()
               if accept(n) and p.grad is not None]
        stats[family+'_grad_norm']=float(torch.stack(norms).norm()) if norms else 0.
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, cfg['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return dict(stats, grad_norm=float(norm), lrs={g['name']:g['lr'] for g in optimizer.param_groups})
