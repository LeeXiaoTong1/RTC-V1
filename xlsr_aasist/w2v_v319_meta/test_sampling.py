import unittest
from collections import Counter
from w2v_v3186.test_sampling import rows_for
from w2v_v3185.common import GROUPS
from w2v_v31810.identity import canonical_sources, build_registry, registry_config
from .sampling import Plan


class SamplingTests(unittest.TestCase):
    def settings(self):
        return dict(source_draws_per_stream=[40, 40, 40, 40], consistency="dual",
                    platform_sources_per_group=3, pcl_noise_weight=.15, pcl_platform_weight=.045,
                    warm_epochs=0, source_epoch_offset=12, experiment="E2")

    def test_exact_budgets_distinct_sources_A_only_references_and_resume(self):
        rows = rows_for((20, 20, 20, 20))
        cfg = self.settings()
        plan = Plan(rows, cfg=cfg)
        batches = list(plan.batches(0))
        counts = Counter()
        for batch in batches:
            sources = [t["canonical_source"] for t in batch]
            self.assertEqual(len(sources), len(set(sources)))
            for t in batch:
                r = rows[t["index"]]
                counts[t["kind"], r["language"], r["label"]] += 1
                if t["meta_side"] == "B" or t["kind"] == "online":
                    self.assertFalse(t["pcl_enabled"])
                    self.assertNotIn("pcl_online_index", t)
                else:
                    self.assertTrue(t["pcl_enabled"])
                    if "pcl_online_index" in t:
                        self.assertEqual(plan.canonical[t["pcl_online_index"]], t["canonical_source"])
                quota = t["batch_group_quotas"][GROUPS.index((r["language"], r["label"]))]
                actual = sum(x["meta_side"] == t["meta_side"] and x["kind"] == t["kind"] and
                             (rows[x["index"]]["language"], rows[x["index"]]["label"]) == (r["language"], r["label"])
                             for x in batch)
                self.assertEqual(quota, actual)
        self.assertEqual(set(counts.values()), {40})
        self.assertEqual(batches, list(Plan(rows, cfg=cfg).batches(0)))
        self.assertEqual(batches, list(Plan(rows, cfg=dict(cfg, experiment="E0")).batches(0)))

    def test_calibration_keeps_all_three_tasks_and_A_pcl(self):
        rows = rows_for((20, 20, 20, 20))
        plan = Plan(rows, cfg=self.settings())
        batches = plan.calibration_batches()
        self.assertEqual(Counter(b[0]["task_kind"] for b in batches), dict(noise=4, rtc=4, joint=4))
        for batch in batches:
            for t in batch:
                self.assertTrue(t["calibration"])
                if t["kind"] == "offline" and t["meta_side"] == "A":
                    self.assertTrue(t["pcl_enabled"])
                if t["meta_side"] == "B":
                    self.assertFalse(t["pcl_enabled"])

    def test_production_sampler_consumes_pcm_registry_and_rejects_tampering(self):
        rows = rows_for((20, 20, 20, 20))
        keys, _ = canonical_sources(rows)
        import hashlib
        for row, key in zip(rows, keys):
            row.update(canonical_source=hashlib.sha256(key.encode()).hexdigest(),
                       audio_sha256=hashlib.sha256(row["id"].encode()).hexdigest(),
                       decoded_pcm_sha256=hashlib.sha256(("pcm:"+row["id"]).encode()).hexdigest())
        registry = build_registry(rows, [], {})
        cfg = dict(self.settings(), source_registry=registry_config(registry))
        plan = Plan(rows, cfg=cfg)
        self.assertEqual(plan.canonical, [r["canonical_source"] for r in rows])
        rows[0]["decoded_pcm_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "identity/order"):
            Plan(rows, cfg=cfg)


if __name__ == "__main__":
    unittest.main()
