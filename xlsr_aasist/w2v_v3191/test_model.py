"""The narrower trainable suffix must not discard any learned parent adapter."""
import unittest
import numpy as np
import torch
from w2v_v3189.test_model import tiny_cfg, tiny_model
from .common import partial_state, apply_partial
from .model import optimizer_for, adapter_inventory
from .test_lifecycle import detector


class ContinuityTests(unittest.TestCase):
    def test_parent_predictions_equal_and_prefix_remains_frozen(self):
        torch.set_num_threads(1)
        torch.manual_seed(3191)
        cfg = tiny_cfg(4)
        parent = tiny_model(cfg)
        with torch.no_grad():
            for name, parameter in parent.named_parameters():
                if name.endswith(".lora_b"):
                    parameter.normal_(0, .04)
        cfg.update(encoder_layers=4, retained_lora_layers=4, lora_layers=2, checkpoint_layers=2)
        torch.manual_seed(3191)
        model = detector(cfg)
        apply_partial(model, partial_state(parent))
        waves = [np.random.default_rng(29).normal(size=5200).astype(np.float32)]
        model.eval()
        parent.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(waves), parent(waves), rtol=0, atol=0)
        self.assertEqual(adapter_inventory(model)["trainable_layers"], [3, 4])
        self.assertEqual(adapter_inventory(model)["retained_layers"], [1, 2, 3, 4])
        prefix = {name: p.detach().clone() for name,p in model.named_parameters()
                  if name.startswith(("encoder.layers.0.", "encoder.layers.1."))}
        optimizer = optimizer_for(model, cfg)
        model.train()
        model(waves).sum().backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in prefix:
                torch.testing.assert_close(parameter, prefix[name], rtol=0, atol=0)
                self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
