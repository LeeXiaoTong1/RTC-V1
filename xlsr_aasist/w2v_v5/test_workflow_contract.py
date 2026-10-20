"""Production sampler contracts with nonaligned inherited source cycles."""
import copy
import hashlib
import unittest

from w2v_v3185.common import GROUPS
from w2v_v31810.identity import build_registry, registry_config
from w2v_v3191.sampling import Plan as ParentPlan
from .sampling import Plan


def sampler_fixture(stage):
    """23 genuine identities per cell, 24 draws, inherited cycle offset 6.

    The metadata-only hashes stand for verified PCM identities.  No waveform
    decoder is needed to exercise the real production scheduling algorithm.
    """
    rows = []
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    for group in GROUPS:
        for index in range(23):
            source = f"train/offline/{group[0]}/{group[1]}-{index}.wav"
            for kind in ("offline", "online"):
                name = f"train/{kind}/{group[0]}/{group[1]}-{index}.wav"
                rows.append(dict(id=name, audio=name, condition=kind, domain=kind,
                    language=group[0], label=group[1], split="train", source_id=source,
                    group_id=digest(source), source_sha256=digest(source),
                    audio_sha256=digest(name + "-file"), decoded_pcm_sha256=digest(name + "-pcm"),
                    canonical_source=digest(source + "-canonical"),
                    v5_origin_split="train", v5_pcl_eligible=False))
    registry = build_registry(rows, [], [])
    cfg = dict(consistency="none", source_epoch_offset=6, source_draws_per_stream=[24]*4,
               experiment="E0", v5_stage=stage, v5_manifest_phase=stage,
               source_registry=registry_config(registry))
    return rows, cfg


class WorkflowContractTests(unittest.TestCase):
    def test_final_visits_every_source_in_each_epoch_with_nonzero_parent_offset(self):
        rows, cfg = sampler_fixture("final")
        plan = Plan(rows, stream_sources=24, seed=50001, cfg=cfg)
        for epoch in range(2):
            seen = {(kind, group): set() for kind in ("online", "offline") for group in GROUPS}
            draws = {key: 0 for key in seen}
            for tickets in plan.batches(epoch):
                # Source repair must keep its original isolation guarantee.
                self.assertEqual(len({t["canonical_source"] for t in tickets}), len(tickets))
                for ticket in tickets:
                    row = rows[ticket["index"]]
                    key = (ticket["kind"], (row["language"], row["label"]))
                    seen[key].add(ticket["canonical_source"])
                    draws[key] += 1
            for key in seen:
                self.assertEqual(len(seen[key]), 23, (epoch, key))
                self.assertEqual(draws[key], 24, (epoch, key))
        # Additional fitting remains deterministic across independent resumes.
        resumed = Plan(copy.deepcopy(rows), stream_sources=24, seed=50001, cfg=copy.deepcopy(cfg))
        self.assertEqual(list(plan.batches(1)), list(resumed.batches(1)))

    def test_screen_keeps_historical_continuous_cycle_sampling(self):
        rows, cfg = sampler_fixture("screen")
        current = Plan(rows, stream_sources=24, seed=50001, cfg=cfg)
        previous = ParentPlan(rows, stream_sources=24, seed=50001, cfg=cfg)
        for epoch in range(2):
            for kind in ("offline", "online"):
                for group in GROUPS:
                    self.assertEqual(current.view_stream(kind, group, epoch),
                                     previous.view_stream(kind, group, epoch))


if __name__ == "__main__":
    unittest.main()
