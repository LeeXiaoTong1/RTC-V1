"""Six English base versions: original plus five mild, label-blind transforms.

These are reproducible views of an existing recording, not new identities.
Every group uses the same one-in-six unmodified exposure schedule. ZH-fake
receives ordinary dynamic augmentation, rather than a six-slot expansion.
"""
import math
import numpy as np
from w2v_v3185.views import recipe as previous_recipe, augment as previous_augment
from w2v_v315.augment import seed_for

FAMILIES = ('original', 'speed_slow', 'speed_fast', 'frequency_response',
            'temporal_context', 'gentle_envelope')
DEFAULT_VIEW_FACTORS = (6, 6, 1, 3)


def recipe(seed, source, kind, variant, *, family=None, generation=0, dynamic=False):
    if kind not in ('online', 'offline') or type(variant) is not int or not 0 <= variant < 6:
        raise ValueError('Expected Online/Offline and one of six base-version slots')
    family = FAMILIES[variant] if family is None else family
    if family not in FAMILIES or type(generation) is not int or generation < 0 or type(dynamic) is not bool:
        raise ValueError('Invalid base augmentation family or generation')
    if family == 'original' and (dynamic or generation):
        raise ValueError('An original view cannot be dynamic or refreshed')
    if family != 'gentle_envelope':
        # Reuse tested V3.18.5 numerical transforms. Slot identity is independent
        # of family for ZH-real, and dynamic common augmentation uses slot zero.
        result = previous_recipe(seed, source, kind, min(variant, 4), family=family,
                                 generation=generation, dynamic=dynamic)
        # A verified official Online and its Offline source share temporal
        # geometry. Their EQ/envelope details may differ, but a speed/crop
        # choice must not create mismatched content for the auxiliary bridge.
        geometry = np.random.default_rng(seed_for(seed, 'v319-shared-geometry', source,
                                                   variant, family, generation, dynamic))
        if family == 'speed_slow':
            result['rate'] = int(geometry.integers(970, 991))
        elif family == 'speed_fast':
            result['rate'] = int(geometry.integers(1010, 1031))
        elif family == 'temporal_context':
            result.update(keep=float(geometry.uniform(.85, .95)), start=float(geometry.random()))
        result.update(variant=variant, version=3)
        return result
    key = seed_for(seed, 'v319-base-envelope-v1', source, kind, variant, generation, dynamic)
    rng = np.random.default_rng(key)
    return dict(version=3, seed=key, variant=variant, family=family,
                generation=generation, dynamic=dynamic,
                envelope_db=float(rng.uniform(.5, 1.5)),
                cycles=float(rng.uniform(.5, 1.5)), phase=float(rng.uniform(0, 2*np.pi)))


def augment(wave, specification):
    x = np.asarray(wave, np.float32)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Base augmentation requires a finite mono recording')
    if specification['family'] != 'gentle_envelope':
        y, meta = previous_augment(x, specification)
    else:
        phase = np.linspace(0, 2*np.pi*specification['cycles'], len(x)) + specification['phase']
        y = (x*np.power(10., specification['envelope_db']*np.sin(phase)/20.)).astype(np.float32)
        peak = float(np.abs(y).max())
        if peak > .98:
            y *= .98/peak
        meta = dict(recipe=specification, input_samples=len(x), output_samples=len(y), fallback=None,
                    source_identity_preserved=True, independent_new_recording=False)
    cropped = specification['family'] == 'temporal_context' and meta['fallback'] is None
    start = 0
    if cropped:
        count = len(y)
        start = min(len(x)-count, int(specification['start']*(len(x)-count+1)))
    meta.update(retains_full_source=not cropped,
                input_interval=[start, start+len(y)] if cropped else [0, len(x)],
                normalized_time_bridge_compatible=not cropped)
    return y, meta


class ViewSchedule:
    """Source-cycle scheduling independent of epoch boundaries and worker order."""
    def __init__(self, rows, inventory, pools, seed, factors=DEFAULT_VIEW_FACTORS):
        from w2v_v313.data import GROUPS
        if tuple(factors) != DEFAULT_VIEW_FACTORS:
            raise ValueError('V3.19 fixed base-version budgets must be 6 6 1 3')
        self.rows, self.inventory, self.seed = rows, inventory, seed
        self.factors = dict(zip(GROUPS, factors))
        self.definitions = {}
        # All ordered unequal pairs distribute the five shared transform
        # families evenly across ZH-real's two fixed augmented slots.
        pairs = [(a, b) for a in range(1, 6) for b in range(1, 6) if a != b]
        for kind in ('offline', 'online'):
            for group, sources in pools.items():
                ordered = sorted(sources, key=lambda s: (seed_for(seed, 'v319-definition', s), s))
                for rank, source in enumerate(ordered):
                    factor = self.factors[group]
                    families = tuple(range(1, 6)) if factor == 6 else pairs[rank % len(pairs)] if factor == 3 else ()
                    self.definitions[kind, source] = dict(factor=factor, phase=rank % 6,
                        family_phase=rank % 5, families=families)

    def description(self, source, kind, cycle):
        if type(cycle) is not int or cycle < 0:
            raise ValueError('Source cycle must be nonnegative')
        d = self.definitions[kind, source]
        original = (cycle+d['phase']) % 6 == 0
        originals_before = (cycle+d['phase']+5)//6 - (d['phase']+5)//6
        augmented_before = cycle-originals_before
        variant = generation = family = 0
        dynamic = False
        if not original:
            if d['factor'] == 1:
                dynamic = True
                family = 1+(augmented_before+d['family_phase']) % 5
                generation = augmented_before//5
            else:
                count = d['factor']-1
                rotation = d['family_phase'] if count == 5 else 0
                variant = 1+(augmented_before+rotation) % count
                family = d['families'][variant-1]
        mode = 'dynamic' if dynamic else 'fixed'
        identity = f'{source}#v319:{kind}:{mode}:{variant}:g{generation}:{FAMILIES[family]}'
        return dict(variant=variant, family=FAMILIES[family], generation=generation,
                    view_id=identity, fixed_factor=d['factor'], source_cycle=cycle,
                    dynamic=dynamic, original=original)

    def specification(self, source, kind, cycle):
        description = self.description(source, kind, cycle)
        r = recipe(self.seed, source, kind, description['variant'], family=description['family'],
                   generation=description['generation'], dynamic=description['dynamic'])
        return dict(description, recipe=r)

    def cumulative(self, source, kind, visits):
        d = self.definitions[kind, source]
        originals = (visits+d['phase']+5)//6 - (d['phase']+5)//6
        augmented = visits-originals
        fixed = int(originals > 0)+min(d['factor']-1, augmented)
        return dict(fixed=fixed, dynamic=augmented if d['factor'] == 1 else 0,
                    realized=int(originals > 0)+(augmented if d['factor'] == 1 else min(d['factor']-1, augmented)))
