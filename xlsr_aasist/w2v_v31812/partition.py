"""All conditions of a PCM-registered source stay on the same residual side."""
from collections import defaultdict
from w2v_v31810.common import fingerprint, seed_for
from w2v_v31810.identity import registered_sources


def split_sources(rows, cfg, fraction=.1, seed=3181200):
    registered_sources(rows, cfg)
    if not 0 < fraction < .5:
        raise ValueError('Train-internal validation fraction must be between 0 and .5')
    groups = defaultdict(dict)
    for r in rows:
        groups[r['language'], r['label']].setdefault(r['canonical_source'], set()).add(r['condition'])
    hold = set()
    for group, members in groups.items():
        # Stratify paired/Offline-only too, so all eight validation cells exist.
        strata = defaultdict(list)
        for source, conditions in members.items():
            strata['online' in conditions].append(source)
        for paired, sources in strata.items():
            if paired and len(sources) < 10:
                raise ValueError('Insufficient independent paired sources for residual validation')
            ordered = sorted(sources, key=lambda s: (seed_for(seed, group, paired, s), s))
            count = min(len(ordered)-1, max(1, round(len(ordered)*fraction))) if len(ordered)>1 else 0
            hold.update(ordered[:count])
    all_sources = {r['canonical_source'] for r in rows}
    result = dict(seed=seed, fraction=fraction, fit=sorted(all_sources-hold), validation=sorted(hold),
        interpretation='Validation sources are unseen by the residual/probes only; the parent was trained on old Train.',
        external_validation='Unchanged parent Dev Select, then one selected candidate on parent Audit.')
    result['hash'] = fingerprint(result)
    return result


def full_sweep(rows, plan, view, *, original=False):
    """Every canonical source/stream once, using parent augmentation functions.

    A coverage sweep changes extraction order, not DSP controls. Training later
    retains the parent's four source exposure budgets for each of two streams.
    """
    unique = {}
    for i, r in sorted(enumerate(rows), key=lambda ir: ir[1]['id']):
        unique.setdefault((r['canonical_source'], r['condition']), i)
    result = []
    for rank, ((source, kind), i) in enumerate(sorted(unique.items())):
        spec = plan.view_spec(i, kind, view)
        if original:
            spec = dict(base_variant=0, base_family='original', base_generation=0,
                        base_dynamic=False, base_view_id=rows[i]['id']+'#original')
        number = seed_for(plan.seed, 'v31812-condition', source, view) % (18*3*100)
        domain, nuisance, heldout = plan._conditions(number, 'X' if rank%2 else 'Y', rank%3)
        result.append(dict(index=i, kind=kind, **spec, canonical_source=source,
            occurrence=f'v31812-full:{view}:{source}:{kind}', epoch=max(1, view+1), step=number,
            source_cycle=view, noise_recipe_view=view%2, domain_override=domain if kind=='offline' else None,
            nuisance_seed=nuisance, heldout_condition=heldout, condition_rank=rank%3))
    return result
