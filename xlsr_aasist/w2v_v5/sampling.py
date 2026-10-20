"""Source-isolated paired CE; inherited PCL only where real phone assets exist."""
import math
import numpy as np
from w2v_v31810.sampling import Plan as MetaPlan
from w2v_v3191.sampling import Plan as PairedPlan
from w2v_v3186.sampling import canonical_pairs
from w2v_v3185.common import seed_for


class Plan(PairedPlan):
    def __init__(self, rows, stream_sources=24, seed=50001, quotas=(6, 6, 6, 6),
                 view_factors=(5, 5, 1, 3), refresh_base_variants=True, *, cfg=None):
        MetaPlan.__init__(self, rows, stream_sources, seed, quotas,
                          view_factors, refresh_base_variants, cfg=cfg)
        self.continuation_cfg = dict(cfg or {})
        eligible = [i for i, r in enumerate(rows) if r.get("v5_pcl_eligible", False)]
        subset = [rows[i] for i in eligible]
        pairs = canonical_pairs(subset) if subset and cfg.get("consistency") == "dual" else {}
        self.reference_pairs = {eligible[i]: eligible[j] for i, j in pairs.items()}

    def view_stream(self, kind, group, epoch):
        if self.continuation_cfg.get("v5_manifest_phase") != "final":
            return super().view_stream(kind, group, epoch)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        if self._view_epoch != epoch:
            self._view_epoch, self._view_streams = epoch, {}
        key = kind, group
        if key in self._view_streams:
            return self._view_streams[key]
        pool, count = self.pools[kind][group], self.budgets[group]
        if count < len(pool):
            raise ValueError("Final budget cannot omit canonical sources")
        order = np.random.default_rng(seed_for(self.seed, "v5-full-epoch", kind, group, epoch)).permutation(len(pool))
        # Start at a permutation boundary, not the middle of the parent's
        # continuous cycle. N draws crossing two independently shuffled cycles
        # would otherwise repeat some sources and miss others.
        # Repeat the same permutation within this epoch: when a cycle wraps,
        # a logical batch still cannot repeat a source (N >= its quota).
        cycle_offset = (int(self.cfg.get("source_epoch_offset", 0))+epoch)*math.ceil(count/len(pool))
        values = []
        for draw in range(count):
            cycle, position = divmod(draw, len(pool))
            index = pool[int(order[position])]
            variant_cycle = cycle_offset+cycle
            values.append((index, self.view_spec(index, kind, variant_cycle), variant_cycle))
        self._view_streams[key] = values
        return values

    def _references(self, tickets):
        tickets = super()._references(tickets)
        for ticket in tickets:
            if not self.rows[ticket["index"]].get("v5_pcl_eligible", False):
                ticket.update(pcl_enabled=False, pcl_noise=False, pcl_platform_budget=0)
                ticket.pop("pcl_online_index", None)
        return tickets
