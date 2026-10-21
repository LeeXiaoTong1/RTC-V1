"""Train-only fixed-mixture ablation for the explicit source-isolated control."""
from itertools import islice
import torch
from .common import read_json, atomic_json, fingerprint, capture_rng, restore_rng, seed_all
from .parent import is_final_parent
from .train import make_plan
from w2v_v5.data import Waves, TicketSampler, loader, close, microbatches


def fixed_train_weights(model, cfg, root, checkpoint_identity):
    if is_final_parent(cfg):
        raise ValueError('Submitted parent has trained Dev; no Dev ablation is permitted')
    rows = read_json(root / 'screen' / 'train_rows.json')
    binding = fingerprint(dict(checkpoint=checkpoint_identity, cfg=fingerprint(cfg),
                               train_ids=[r['id'] for r in rows], plan_batches=1))
    path = root / 'fixed_fusion_lock.json'
    if path.exists():
        saved = read_json(path)
        if saved['identity'] != binding:
            raise ValueError('Fixed-fusion calibration inputs changed')
        return saved['weights']
    rng, was_training, batches = capture_rng(), model.training, None
    try:
        seed_all(cfg['sampling_seed'])
        model.eval()
        model.fusion.reset_observations(enabled=True)
        plan = make_plan(rows, cfg)
        sampler = TicketSampler()
        sampler.batches = list(islice(plan.batches(0), 1))
        batches = loader(Waves(rows, cfg), cfg, batch_sampler=sampler)
        with torch.inference_mode():
            for units in batches:
                # Measure classification views only; Train references and Dev
                # never choose the fixed mixture. Reuse the real V5.1 wave path.
                views = []
                for unit in units:
                    row = unit[0]
                    views.append([dict(row, pcl_refs={}, pcl_pairs={})])
                    if row.get('role') == 'noisy':
                        ref = row.get('pcl_refs', {}).get('clean')
                        if ref is not None:
                            views.append([dict(ref, pcl_refs={}, pcl_pairs={})])
                for group in microbatches(views, cfg['microbatch'], cfg['frame_budget']):
                    model.encode([u[0]['wave'] for u in group])
        stats = model.fusion.observations()
        if not stats['encoded_views']:
            raise ValueError('No Train classification views available for fixed-fusion ablation')
        atomic_json(path, dict(identity=binding, weights=stats['mean_weights'],
            training_views=stats['encoded_views'], source='first balanced Train plan batch',
            checkpoint=checkpoint_identity, purpose='diagnostic only; never selects final recipe'))
        return stats['mean_weights']
    finally:
        if batches is not None:
            close(batches)
        model.fusion.reset_observations()
        model.train(was_training)
        restore_rng(rng)
