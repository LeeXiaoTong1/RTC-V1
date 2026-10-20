import copy
import json
import unittest
import torch

from w2v_v3189.test_step import CountingDetector
from w2v_v31810.test_meta import assert_state_equal
from .test_continuation import tasks, sgd
from .meta import rng_state
from .calibration import calibrate, CalibrationError, CANDIDATES, choose_alpha


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_dual_pcl_calibration_restores_every_state_and_never_encodes_B(self):
        torch.manual_seed(19)
        cfg, _ = tasks()
        model = CountingDetector()
        optimizer = sgd(model)
        before = copy.deepcopy(model.state_dict())
        moments = copy.deepcopy(optimizer.state_dict())
        rng = rng_state()
        batches = []
        for kind in ("noise", "rtc", "joint"):
            for _ in range(4):
                _, units = tasks()
                for unit in units:
                    unit[0]["task_kind"] = kind
                batches.append(units)
        b_waves = {id(unit[0]["wave"]) for units in batches for unit in units if unit[0]["meta_side"] == "B"}
        try:
            report = calibrate(model, optimizer, batches, cfg)
        except CalibrationError as exc:
            report = exc.report
        assert_state_equal(self, model.state_dict(), before)
        assert_state_equal(self, optimizer.state_dict(), moments)
        assert_state_equal(self, rng_state(), rng)
        self.assertFalse(set(model.encoded) & b_waves)
        self.assertEqual(report["task_counts"], dict(noise=4, rtc=4, joint=4))
        self.assertEqual(len(report["records"]), 12 * len(CANDIDATES))
        self.assertEqual(report["encoded_views"], len(model.encoded))
        self.assertEqual(report["encoded_views"], 12 * 16 * (1 + len(CANDIDATES)))
        self.assertTrue(any(r["A_logit_change"]["max_abs"] > 0 for r in report["records"]))
        json.dumps(report, allow_nan=False)

    def test_small_step_candidates_and_effect_size_gates(self):
        records = []
        for i, alpha in enumerate(CANDIDATES):
            records.extend(dict(alpha=alpha, relative_drop=.01 if i == 2 else .09, finite=True)
                           for _ in range(12))
        alpha, _ = choose_alpha(records)
        self.assertEqual(alpha, CANDIDATES[2])
        records[2 * 12]["relative_drop"] = -.21
        self.assertIsNone(choose_alpha(records)[0])


if __name__ == "__main__":
    unittest.main()
