"""Averaging changes one actual detector, never a collection of scores."""
import copy
import unittest

import numpy as np
import torch
from torch import nn

from .weights import average_states, invariant_keys


class WeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_lora_factor_average_is_not_effective_update_average(self):
        first = {"encoder.lora_a": torch.tensor([[1.]]),
                 "encoder.lora_b": torch.tensor([[1.]])}
        last = {"encoder.lora_a": torch.tensor([[3.]]),
                "encoder.lora_b": torch.tensor([[5.]])}
        result = average_states([first, last], [1., 1.])
        actual = result["encoder.lora_b"] @ result["encoder.lora_a"]
        effective_average = .5 * (first["encoder.lora_b"] @ first["encoder.lora_a"]
                                  + last["encoder.lora_b"] @ last["encoder.lora_a"])
        self.assertEqual(float(actual), 6.)
        self.assertEqual(float(effective_average), 8.)
        self.assertNotEqual(float(actual), float(effective_average))

    def test_endpoints_load_exact_predictions_in_real_partial_detector(self):
        from w2v_v3189.test_model import tiny_cfg, tiny_model
        from w2v_v3189.common import partial_state, apply_partial

        torch.manual_seed(510)
        model = tiny_model(tiny_cfg())
        model.eval()
        wave = np.random.default_rng(510).normal(size=5200).astype(np.float32)
        first = partial_state(model)
        with torch.no_grad():
            expected_first = model([wave]).clone()
            for name, parameter in model.named_parameters():
                if name.endswith(".lora_b") or name.startswith("head."):
                    parameter.add_(torch.randn_like(parameter) * .001)
        last = partial_state(model)
        with torch.no_grad():
            expected_last = model([wave]).clone()
        for coefficients, expected in (([1., 0.], expected_first), ([0., 1.], expected_last)):
            state = average_states([first, last], coefficients,
                                   constant_keys=invariant_keys(model, first))
            apply_partial(model, state)
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model([wave]), expected, rtol=0, atol=0)

    def test_average_preserves_shapes_dtypes_and_inputs(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            first = {"head.weight": torch.tensor([[1., 3.]], dtype=dtype),
                     "pcl_scale.frozen": torch.tensor(True)}
            last = {"head.weight": torch.tensor([[5., 7.]], dtype=dtype),
                    "pcl_scale.frozen": torch.tensor(True)}
            original = copy.deepcopy(first)
            result = average_states([first, last], [3., 1.])
            self.assertEqual(result["head.weight"].dtype, dtype)
            self.assertEqual(result["head.weight"].shape, first["head.weight"].shape)
            torch.testing.assert_close(result["head.weight"],
                                       torch.tensor([[2., 4.]], dtype=dtype), rtol=0, atol=0)
            result["head.weight"].zero_()
            for key in original:
                torch.testing.assert_close(first[key], original[key], rtol=0, atol=0)
        state = {"head.weight": torch.tensor([-0., 2.])}
        endpoint = average_states([state], [9.])
        self.assertTrue(bool(torch.signbit(endpoint["head.weight"])[0]))
        self.assertNotEqual(endpoint["head.weight"].data_ptr(), state["head.weight"].data_ptr())

    def test_all_registered_buffers_and_frozen_parameters_are_invariant(self):
        model = nn.Module()
        model.trainable = nn.Parameter(torch.tensor(1.))
        model.frozen_prefix = nn.Parameter(torch.tensor(2.), requires_grad=False)
        model.register_buffer("unusual_float_buffer", torch.tensor(3.))
        first = copy.deepcopy(model.state_dict())
        last = copy.deepcopy(first)
        last["trainable"].fill_(9.)
        constant = invariant_keys(model, first)
        self.assertEqual(constant, {"frozen_prefix", "unusual_float_buffer"})
        result = average_states([first, last], [1., 1.], constant_keys=constant)
        self.assertEqual(float(result["trainable"]), 5.)
        for name in constant:
            bad = copy.deepcopy(last)
            bad[name].add_(1.)
            with self.assertRaisesRegex(ValueError, "Invariant"):
                average_states([first, bad], [.5, .5], constant_keys=constant)

    def test_pcl_nonfloat_and_running_statistics_cannot_be_averaged(self):
        values = {"pcl_scale.scale": torch.tensor([1.]),
                  "pcl_scale.source_visits": torch.tensor(2),
                  "head.running_mean": torch.tensor([0.]),
                  "head.flag": torch.tensor(False)}
        for name in values:
            changed = copy.deepcopy(values)
            if changed[name].dtype == torch.bool:
                changed[name].fill_(True)
            else:
                changed[name].add_(1)
            with self.assertRaisesRegex(ValueError, "Invariant"):
                average_states([values, changed], [.5, .5])
            # An exact endpoint may choose a different calibration snapshot.
            endpoint = average_states([values, changed], [0., 1.])
            torch.testing.assert_close(endpoint[name], changed[name], rtol=0, atol=0)

    def test_bad_weights_and_incompatible_snapshots_are_rejected(self):
        state = {"head.weight": torch.ones(2)}
        for coefficients in ([], [0.], [-1.], [float("nan")], [float("inf")], ["bad"]):
            with self.assertRaises(ValueError):
                average_states([state], coefficients)
        with self.assertRaises(ValueError):
            average_states([state, state], [1e308, 1e308])
        for other in ({}, {"other": torch.ones(2)}, {"head.weight": torch.ones(3)},
                      {"head.weight": torch.ones(2, dtype=torch.float64)},
                      {"head.weight": torch.tensor([float("nan"), 1.])},
                      {"head.weight": torch.ones(2, dtype=torch.complex64)},
                      {"head.weight": [1., 1.]}):
            with self.assertRaises(ValueError):
                average_states([state, other], [1., 1.])
        with self.assertRaisesRegex(ValueError, "Invariant keys"):
            average_states([state], [1.], constant_keys={"missing"})


if __name__ == "__main__":
    unittest.main()
