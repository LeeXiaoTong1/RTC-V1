"""Two full views per source, with exactly half the classification mass noisy."""
from collections import Counter
import math

import numpy as np

from w2v_v313.data import SourcePlan as PreviousPlan, PairIndex, loader as previous_loader, GROUPS
from w2v_v312.data import bundles


class SourcePlan(PreviousPlan):
    def __init__(self, rows, batch, seed):
        super().__init__(rows, batch, seed, micro_sources=4)

    def batches(self, epoch, start=0):
        if type(epoch) is not int or epoch < 0 or type(start) is not int or not 0 <= start <= self.steps:
            raise ValueError('Invalid deterministic epoch/resume cursor')
        streams, output = self._streams(epoch), []
        rounds = self.batch // 4
        for step in range(start, self.steps):
            batch = []
            for j in range(rounds):
                position = step * rounds + j
                # Both axes rotate independently across steps AND epochs.
                phase = self.seed + epoch * (self.steps * rounds + 1) + position
                ordinary = 'online' if phase % 2 else 'offline'
                noisy = 'noisy_b' if (phase // 2) % 2 else 'noisy_a'
                for group_index, key in enumerate(GROUPS):
                    item = self.inventory[streams[key][position]]
                    first = ordinary if ordinary in item['indices'] else 'offline'
                    occurrence = f'{epoch}:{step}:{4*j+group_index}'
                    for side, condition in enumerate((first, noisy)):
                        batch.append(PairIndex(item['indices'][condition], occurrence, side,
                            'ordinary_noisy', 'ordinary_noisy', j, 1./(2*self.batch), 0.))
            output.append(batch)
        return output

    def coverage(self, epoch=0):
        batches = self.batches(epoch)
        draws = Counter(self.rows[t.index]['source_id'] for b in batches for t in b if t.pair_position == 0)
        conditions = Counter(self.rows[t.index]['condition'] for b in batches for t in b)
        mass = Counter()
        for b in batches:
            for ticket in b:
                r = self.rows[ticket.index]
                mass['noisy' if r['condition'].startswith('noisy_') else 'ordinary'] += ticket.ce_weight / self.steps
        return dict(epoch=epoch, steps=self.steps, source_draws=sum(draws.values()), audio_views=sum(conditions.values()),
            available_sources=len(self.inventory), unique_sources=len(draws), view_conditions=dict(conditions),
            ce_mass=dict(mass), source_groups={f'{g[0]}/{g[1]}': dict(available=len(pool),
                draws=sum(draws[s] for s in pool), unique=sum(draws[s]>0 for s in pool),
                maximum_repeats=max(draws[s] for s in pool)) for g, pool in self.pools.items()},
            new_audio_bytes=0, new_frame_cache_bytes=0)


def loader(plan, epoch, start, cfg):
    # Reuse verified numpy worker transport, without any paired gradient graph.
    return previous_loader(plan, epoch, start, dict(cfg, pair_micro_sources=4))


def loss_weights(rows):
    if not rows:
        raise ValueError('Empty logical batch')
    expected = 1. / len(rows)
    if any(not math.isclose(r['ce_weight'], expected, abs_tol=1e-12) for r in rows):
        raise ValueError('Weights must be normalized over the logical batch before microbatching')
    cells, noisy = Counter(), 0.
    for r in rows:
        cells[(r['language'], r['label'])] += r['ce_weight']
        if r['condition'].startswith('noisy_'):
            noisy += r['ce_weight']
    if set(cells) != set(GROUPS) or any(not math.isclose(v, .25) for v in cells.values()) or not math.isclose(noisy, .5):
        raise ValueError('Need four equal groups and exactly 50% Noisy CE mass')
    return np.asarray([r['ce_weight'] for r in rows], dtype=np.float32)


def head_weights(rows):
    """Exact population counterpart to balanced Stage B source sampling."""
    plan = SourcePlan(rows, 16, 0)
    result = np.zeros(len(rows), dtype=np.float64)
    for item in plan.inventory.values():
        mass = .25 / len(plan.pools[(item['language'], item['label'])])
        normal = [c for c in ('offline', 'online') if c in item['indices']]
        for condition in normal:
            result[item['indices'][condition]] = .5 * mass / len(normal)
        for condition in ('noisy_a', 'noisy_b'):
            result[item['indices'][condition]] = .25 * mass
    if not np.isclose(result.sum(), 1.):
        raise ValueError('Frozen-head source budget is incomplete')
    return result
