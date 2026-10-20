"""PCM-isolated rotating tasks; inherited paired losses on A only."""
from collections import Counter
from w2v_v31810.sampling import Plan as MetaPlan, quota_schedule, TASKS
from w2v_v3189.sampling import validate_units, probe_tickets
from w2v_v3186.sampling import canonical_pairs
from w2v_v3185.common import GROUPS, seed_for


class Plan(MetaPlan):
    def __init__(self, rows, stream_sources=24, seed=31801, quotas=(6, 6, 6, 6),
                 view_factors=(5, 5, 1, 3), refresh_base_variants=True, *, cfg=None):
        super().__init__(rows, stream_sources, seed, quotas, view_factors, refresh_base_variants, cfg=cfg)
        self.continuation_cfg = dict(cfg or {})
        self.reference_pairs = canonical_pairs(rows) if self.continuation_cfg.get("consistency") == "dual" else {}

    def _references(self, tickets):
        cfg = self.continuation_cfg
        for side in ("A", "B"):
            members = [t for t in tickets if t["meta_side"] == side]
            quotas = tuple(sum(t["kind"] == "offline" and
                              (self.rows[t["index"]]["language"], self.rows[t["index"]]["label"]) == group
                              for t in members) for group in GROUPS)
            for ticket in members:
                ticket["batch_group_quotas"] = quotas
                ticket["pcl_enabled"] = False
            if side == "B" or cfg.get("consistency") != "dual":
                continue
            for group, quota in zip(GROUPS, quotas):
                offline = [t for t in members if t["kind"] == "offline" and
                           (self.rows[t["index"]]["language"], self.rows[t["index"]]["label"]) == group]
                target = min(quota, cfg.get("platform_sources_per_group", 3))
                candidates = sorted((t for t in offline if t["index"] in self.reference_pairs),
                                    key=lambda t: seed_for(self.seed, "v319-platform", t["occurrence"]))
                selected = {id(t) for t in candidates[:target]} if cfg.get("pcl_platform_weight", 0) > 0 else set()
                for ticket in offline:
                    ticket.update(pcl_noise=cfg.get("pcl_noise_weight", 0) > 0,
                                  pcl_platform_budget=target)
                    if id(ticket) in selected:
                        other = self.reference_pairs[ticket["index"]]
                        if self.canonical[other] != ticket["canonical_source"]:
                            raise ValueError("Auxiliary reference is outside its canonical source")
                        ticket["pcl_online_index"] = other
                    ticket["pcl_enabled"] = ticket["pcl_noise"] or id(ticket) in selected
        return tickets

    def batches(self, epoch):
        for tickets in super().batches(epoch):
            yield self._references(tickets)

    def calibration_batches(self):
        # The base implementation intentionally constructs its own MetaPlan.
        # Attach exactly the same A-only auxiliary contract to its fixed tasks.
        return [self._references(tickets) for tickets in super().calibration_batches()]

    def coverage(self, epoch):
        value = super().coverage(epoch)
        value["pcl"] = dict(enabled=self.continuation_cfg.get("consistency") == "dual",
                            side="A", reference_pairs=len(self.reference_pairs),
                            B_classification_only=True)
        return value

    def audit_budget(self, epochs=1):
        value = super().audit_budget(epochs)
        value.update(no_auxiliary_views=self.continuation_cfg.get("consistency") != "dual",
                     auxiliary_policy="Parent dual PCL on A; matched Clean-RTC enters paired CE on both sides")
        return value


def describe(rows, seed, stream_sources, quotas, view_factors=(5, 5, 1, 3),
             refresh_base_variants=True, *, cfg=None):
    plan = Plan(rows, stream_sources, seed, quotas, view_factors, refresh_base_variants, cfg=cfg)
    return dict(plan.coverage(0), full_budget=plan.audit_budget(cfg["epochs"]))
