"""Numerical tests of retained weights, first-order updates and side isolation."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayer, TinyLayout
from w2v_v3189.test_model import tiny_cfg, tiny_model as parent_tiny
from w2v_v3189.test_step import CountingDetector, fixture as pcl_fixture
from w2v_v3189.common import partial_state, apply_partial
from .model import Detector, optimizer_for, adapter_inventory
from .meta import train_step, rng_state
from .side import side_pass
from .schedule import schedule
from .common import RECIPE, schema_for


def config(depth=4, active=2):
    cfg = tiny_cfg(depth)
    cfg.update(version="3.19", recipe=RECIPE, encoder_layers=depth,
               lora_layers=active, retained_lora_layers=depth, checkpoint_layers=active,
               experiment="E2", warm_epochs=0, epochs=1, device="cpu",
               backend_precision="fp32", consistency="dual", lr_floor=.5, meta_beta=.5, meta_ramp_updates=2)
    return cfg


def detector(cfg):
    encoder = TinyEncoder()
    encoder.layers = nn.ModuleList([TinyLayer() for _ in range(cfg["encoder_layers"])])
    return Detector(TinyFrontend(), encoder, cfg, TinyLayout)


def tasks():
    cfg, original = pcl_fixture()
    cfg.update(pcl_ramp_updates=1, backend_precision="fp32", device="cpu", pcl_variance_weight=0.)
    units = []
    for side in ("A", "B"):
        for unit in copy.deepcopy(original):
            row = unit[0]
            for item in (row, *row.get("pcl_refs", {}).values()):
                for key in ("id", "source_id", "group_id"):
                    item[key] = side + "-" + item[key]
                item["canonical_source"] = item["source_id"]
            row.update(meta_side=side, task_kind="noise", batch_group_quotas=[1, 1, 1, 1])
            if side == "B":
                row.pop("pcl_refs", None)
                row.pop("pcl_pairs", None)
                row["wave"] *= 1.3
            units.append(unit)
    return cfg, units


def sgd(model):
    return torch.optim.SGD([dict(params=[p for p in model.parameters() if p.requires_grad],
                                 name="fixture", lr=.01, initial_lr=.01)], lr=.01)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_parent_prediction_preserved_and_prefix_not_updated(self):
        torch.manual_seed(19)
        old_cfg = tiny_cfg(4)
        old = parent_tiny(old_cfg)
        with torch.no_grad():
            for name, p in old.named_parameters():
                if name.endswith(".lora_b"):
                    p.normal_(0, .08)
        cfg = config()
        torch.manual_seed(19)
        model = detector(cfg)
        # Tiny pretrained fixtures use identical initialization; partial_state
        # contains all four inherited adapter layers, not just the active two.
        apply_partial(model, partial_state(old))
        waves = [np.random.default_rng(4).normal(size=5200).astype(np.float32)]
        old.eval()
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(waves), old(waves), rtol=0, atol=0)
        inventory = adapter_inventory(model)
        self.assertEqual(inventory["trainable_layers"], [3, 4])
        self.assertEqual(inventory["retained_layers"], [1, 2, 3, 4])
        prefix = {n: p.detach().clone() for n, p in model.named_parameters()
                  if n.startswith(("encoder.layers.0.", "encoder.layers.1."))}
        opt = optimizer_for(model, cfg)
        model.train()
        model(waves).sum().backward()
        opt.step()
        for n, p in model.named_parameters():
            if n in prefix:
                torch.testing.assert_close(p, prefix[n], rtol=0, atol=0)
                self.assertIsNone(p.grad)
        torch.manual_seed(19)
        restored = detector(cfg)
        apply_partial(restored, partial_state(model))
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(waves), restored(waves), rtol=0, atol=0)

    def test_first_order_gradient_matches_independent_virtual_ce(self):
        cfg, units = tasks()
        cfg.update(consistency="none", max_grad_norm=1000., label_smoothing=.02)
        torch.manual_seed(7)
        model = CountingDetector()
        reference = copy.deepcopy(model)
        opt, expected_opt = sgd(model), sgd(reference)
        parameters = list(reference.parameters())
        originals = [p.detach().clone() for p in parameters]

        def loss(side):
            rows = [u[0] for u in units if u[0]["meta_side"] == side]
            z = reference.classify_sequences(reference.encode([r["wave"] for r in rows]))
            return F.cross_entropy(z, torch.tensor([r["label"] for r in rows]), label_smoothing=.02)

        alpha, beta = .03, .5
        ga = torch.autograd.grad(loss("A"), parameters)
        with torch.no_grad():
            for p, g in zip(parameters, ga):
                p.add_(g, alpha=-alpha)
        gb = torch.autograd.grad(loss("B"), parameters)
        with torch.no_grad():
            for p, original, a, b in zip(parameters, originals, ga, gb):
                p.copy_(original)
                p.grad = (a + beta*b)/(1+beta)
        expected_opt.step()
        result = train_step(model, opt, units, cfg, alpha, beta)
        self.assertFalse(result["amp"]["skipped_update"])
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
        self.assertEqual(result["encoded_views"], 16)

    def test_dual_pcl_on_A_references_never_classified_or_sent_to_B(self):
        cfg, units = tasks()
        model = CountingDetector()
        main = {id(u[0]["wave"]) for u in units}
        refs = {id(r["wave"]) for u in units for r in u[0].get("pcl_refs", {}).values()}
        stats = train_step(model, sgd(model), units, cfg, .01, .5, audit=True)
        self.assertGreater(stats["pcl_losses"]["noise"]["weighted_loss"], 0)
        self.assertGreater(stats["pcl_losses"]["platform"]["weighted_loss"], 0)
        self.assertEqual(set(model.classified), main)
        self.assertEqual(set(model.encoded), main | refs)
        self.assertEqual(stats["auxiliary_reference_views"], 8)
        self.assertGreater(stats["meta"]["A_logit_change"]["max_abs"], 0)
        self.assertFalse(getattr(model, "_v31810_virtual_active", False))

    def test_B_exception_restores_parameters_buffers_rng_and_optimizer(self):
        cfg, units = tasks()
        model = CountingDetector()
        opt = sgd(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        state_rng = torch.get_rng_state().clone()
        actual = side_pass

        def fail(model, side_units, *args, **kwargs):
            if side_units[0][0]["meta_side"] == "B":
                torch.rand(10)
                raise RuntimeError("B failure")
            return actual(model, side_units, *args, **kwargs)

        with patch("w2v_v319_meta.meta.side_pass", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "B failure"):
                train_step(model, opt, units, cfg, .1, .5)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), state_rng, rtol=0, atol=0)
        self.assertEqual(opt.state, {})
        self.assertFalse(getattr(model, "_v31810_virtual_active", False))

    def test_same_source_across_sides_is_rejected(self):
        cfg, units = tasks()
        units[-1][0]["canonical_source"] = units[0][0]["canonical_source"]
        model = CountingDetector()
        with self.assertRaisesRegex(ValueError, "repeats"):
            train_step(model, sgd(model), units, cfg, .01, .5)

    def test_one_epoch_schedule_has_no_repeated_warmup_and_reaches_beta(self):
        cfg = config()
        model = detector(cfg)
        opt = optimizer_for(model, cfg)
        initial = [g["initial_lr"] for g in opt.param_groups]
        joint, beta = schedule(opt, cfg, 0, 0, 4)
        self.assertTrue(joint)
        self.assertEqual([g["lr"] for g in opt.param_groups], initial)
        self.assertEqual(beta, .25)
        _, beta = schedule(opt, cfg, 0, 3, 4)
        self.assertEqual(beta, .5)
        self.assertEqual([g["lr"] for g in opt.param_groups], [v*.5 for v in initial])
        schema_for(cfg)


if __name__ == "__main__":
    unittest.main()
