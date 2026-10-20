import unittest
from unittest.mock import patch
import numpy as np

from .data import Waves
from .side import _layout


class DataTests(unittest.TestCase):
    def test_domain_ticket_reaches_synthesizer_and_only_A_builds_pcl(self):
        wave = np.random.default_rng(10).normal(0, .05, 16000).astype(np.float32)
        off = dict(id="off", source_id="off", group_id="g", language="en", label=1, split="train")
        on = dict(off, id="on")
        cfg = dict(seed=31801, warm_epochs=0, consistency="dual", rtc_activity_policy="mixed")
        dataset = Waves([off, on], cfg)
        dataset.original = lambda _: wave.copy()
        dataset.bank = object()
        dataset.engines = object()
        dataset._phones = lambda _: [dict(instance=0, token="a", start_sample=1000,
                                         end_sample=5000, posterior=.9, valid=True)]
        ticket = dict(index=0, kind="offline", base_variant=0, base_family="original",
                      base_generation=0, base_dynamic=False, base_view_id="view", occurrence="source-visit",
                      epoch=0, meta_side="A", canonical_source="canonical", task_kind="noise",
                      domain_override=dict(scene="rain", family="webrtc"), nuisance_seed=1234,
                      pcl_enabled=True, pcl_noise=True, pcl_online_index=1)
        captured = []

        def matched(source, recipe, *_):
            captured.append(recipe)
            return source.copy(), source.copy(), dict(recipe=recipe, pair_eligible=True)

        def single(source, recipe, *_):
            captured.append(recipe)
            return source.copy(), dict(recipe=recipe)

        with patch("w2v_v319_meta.data.generate_matched", side_effect=matched) as pair, \
             patch("w2v_v319_meta.data.generate", side_effect=single) as noisy, \
             patch("w2v_v319_meta.data.estimate_delay", return_value=dict(valid=True, offset=0.)), \
             patch("w2v_v319_meta.data.reliable_spans", return_value=[True]):
            a = dataset[ticket]
            b = dataset[dict(ticket, meta_side="B", pcl_enabled=False)]
        self.assertEqual((pair.call_count, noisy.call_count), (1, 1))
        self.assertEqual(captured[0]["domain_contract"]["scene"], "rain")
        self.assertEqual(captured[0]["family"], "webrtc")
        self.assertEqual(captured[0], captured[1])
        self.assertEqual(a[0]["canonical_source"], "canonical")
        self.assertEqual(a[0]["meta_side"], "A")
        self.assertTrue(a[0]["pcl_pairs"]["noise"])
        self.assertTrue(a[0]["pcl_pairs"]["platform"])
        self.assertEqual(len(_layout(a, dict(noise=.15, platform=.045))[0]), 3)
        self.assertNotIn("pcl_refs", b[0])
        np.testing.assert_array_equal(a[0]["wave"], b[0]["wave"])


if __name__ == "__main__":
    unittest.main()
