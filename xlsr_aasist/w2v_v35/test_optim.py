import copy
import unittest
import torch
from .optim import EMA, optimizer_for, schedule_scale, apply_learning_rates
from .test_losses import model


class OptimizerTests(unittest.TestCase):
    def test_head_ema_never_copies_frozen_encoder_and_joint_expands_once(self):
        detector = model()
        cfg = dict(trainable_layers=2, encoder_lr=1e-6, joint_head_lr=1e-5, head_lr=1e-4, layer_decay=.9)
        optimizer_for(detector, cfg, 'head')
        ema = EMA(detector, decay=.9)
        self.assertTrue(all(name.startswith('head.') for name in ema.shadow))
        with torch.no_grad():
            next(detector.head.parameters()).add_(.1)
        ema.update(detector)
        before = {name: value.clone() for name, value in detector.state_dict().items()}
        with ema.average_parameters(detector):
            self.assertFalse(torch.equal(next(detector.head.parameters()), before[next(iter(ema.shadow))]))
        for name, value in detector.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        opt = optimizer_for(detector, cfg, 'joint')
        ema.add_trainable(detector)
        self.assertTrue(any(name.startswith('backbone.encoder.layers.0.') for name in ema.shadow))
        self.assertFalse(any(name.startswith('backbone.feature_projection') for name in ema.shadow))
        rates = apply_learning_rates(opt, .5)
        self.assertAlmostEqual(rates['encoder.layer_00.decay'] / rates['encoder.layer_01.decay'], .9)
        self.assertEqual(rates['head.decay'], 5e-6)

    def test_ema_resume_next_update_exact_and_exception_restores_raw(self):
        detector = model()
        ema = EMA(detector, .9)
        ema.update(detector)
        other = copy.deepcopy(detector)
        resumed = EMA(other, .9)
        resumed.load_state_dict(copy.deepcopy(ema.state_dict()), other)
        with torch.no_grad():
            for parameter in detector.parameters():
                if parameter.requires_grad:
                    parameter.add_(.01)
            other.load_state_dict(detector.state_dict())
        ema.update(detector); resumed.update(other)
        for key in ema.shadow:
            torch.testing.assert_close(ema.shadow[key], resumed.shadow[key], atol=0, rtol=0)
        raw = copy.deepcopy(detector.state_dict())
        with self.assertRaisesRegex(RuntimeError, 'validation'):
            with ema.average_parameters(detector):
                raise RuntimeError('validation failed')
        for key in raw:
            torch.testing.assert_close(raw[key], detector.state_dict()[key], atol=0, rtol=0)

    def test_phase_schedule_warmup_and_cosine_endpoint(self):
        self.assertEqual(schedule_scale(0, 100, .05, .1), .2)
        self.assertEqual(schedule_scale(4, 100, .05, .1), 1.)
        self.assertEqual(schedule_scale(5, 100, .05, .1), 1.)
        self.assertAlmostEqual(schedule_scale(99, 100, .05, .1), .1)


if __name__ == '__main__':
    unittest.main()
