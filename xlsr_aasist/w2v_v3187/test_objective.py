"""Objective invariants: balanced CE and source-local, symmetric phone gradients."""
import math
import unittest
import torch
from .objective import phoneme_loss, frame_span, micro_losses, source_weight, auxiliary_weights, confidence_regularization, batch_quotas
from w2v_v3183r2.common import GROUPS


class TestObjective(unittest.TestCase):
    def test_existing_ce_mass_survives_unequal_quotas(self):
        quotas = [4, 3, 5, 4]
        units = []
        for language, label in GROUPS:
            for _ in range(quotas[GROUPS.index((language, label))]):
                row = dict(language=language, label=label)
                units.extend([[row], [row, row]])
        logits = torch.zeros(48, 2, requires_grad=True)
        terms = micro_losses(logits, units, quotas, .25, .02)
        total = sum(value for value, _ in terms)
        self.assertAlmostEqual(float(total), math.log(2), places=6)
        for group in GROUPS:
            mass = sum(float(value) for unit, (value, _) in zip(units, terms)
                       if (unit[0]['language'], unit[0]['label']) == group)
            self.assertAlmostEqual(mass, math.log(2) * .25, places=6)
        total.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_sample_boundaries_map_frame_centres(self):
        self.assertEqual(frame_span(0, 200, 10), (0, 0))
        self.assertEqual(frame_span(200, 520, 10), (0, 1))
        self.assertEqual(frame_span(520, 840, 10), (1, 2))
        self.assertEqual(frame_span(0, 20000, 10), (0, 10))
        with self.assertRaises(ValueError):
            frame_span(-1, 200, 10)

    def test_same_phone_instances_have_symmetric_gradients(self):
        left = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        right = torch.tensor([[.6, .8], [.6, .8], [.8, .6], [.8, .6]], requires_grad=True)
        matches = [[0, 840, 0, 840], [840, 1480, 840, 1480]]
        loss, count = phoneme_loss(left, right, matches)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(loss), .4, places=6)
        loss.backward()
        self.assertGreater(float(left.grad.norm()), 0.)
        self.assertGreater(float(right.grad.norm()), 0.)
        same, _ = phoneme_loss(left.detach(), left.detach() * 3, matches)
        self.assertAlmostEqual(float(same), 0., places=6)

    def test_empty_or_short_phones_do_not_inflate_other_sources(self):
        a = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(3, 4, requires_grad=True)
        value, count = phoneme_loss(a, b, [[0, 100, 0, 100]])
        self.assertEqual(count, 0)
        self.assertEqual(float(value), 0.)
        value.backward()
        self.assertEqual(float(a.grad.abs().sum()), 0.)
        cfg = dict(group_quotas=[4, 3, 5, 4], platform_sources_per_group=2)
        row = dict(language='en', label=1)
        self.assertAlmostEqual(source_weight('noise', row, cfg), .25 / 3)
        self.assertAlmostEqual(source_weight('platform', row, cfg), .125)

    def test_auxiliary_ramp_and_classification_only(self):
        cfg = dict(pcl_ramp_updates=300, pcl_noise_weight=.15, pcl_platform_weight=.15)
        self.assertAlmostEqual(auxiliary_weights(dict(cfg, current_update=150))['noise'], .075)
        self.assertAlmostEqual(auxiliary_weights(dict(cfg, current_update=900))['platform'], .15)
        self.assertEqual(auxiliary_weights(dict(cfg, consistency='none')), dict(noise=0., platform=0.))

    def test_reflection_frames_receive_no_direct_pcl_gradient(self):
        a=torch.randn(12,4,requires_grad=True);b=torch.randn(12,4,requires_grad=True)
        loss,count=phoneme_loss(a,b,[[0,3920,0,3920]],left_valid_frames=2,right_valid_frames=3)
        self.assertEqual(count,1)
        loss.backward()
        self.assertEqual(float(a.grad[2:].abs().sum()),0.)
        self.assertEqual(float(b.grad[3:].abs().sum()),0.)
        zero,count=phoneme_loss(a,b,[[0,3920,0,3920]],left_valid_frames=0,right_valid_frames=3)
        self.assertEqual(count,0)
        self.assertEqual(float(zero),0.)

    def test_confidence_cap_only_correct_overconfident_en_real(self):
        units=[[dict(language=l,label=y)] for l,y in [('en',1),('en',1),('en',1),('zh',1),('en',0)]]
        logits=torch.tensor([[0.,6.],[0.,2.],[0.,-6.],[0.,6.],[6.,0.]],requires_grad=True)
        cfg=dict(en_real_confidence_weight=.01,en_real_confidence_margin=3.5,current_update=300,pcl_ramp_updates=300)
        loss,telemetry=confidence_regularization(logits,units,[1,3,1,1],cfg)
        self.assertAlmostEqual(float(loss),.01*(.125/3)*(6-3.5)**2,places=8)
        self.assertEqual(telemetry.tolist(),[1.,2.5,3.])
        loss.backward()
        self.assertGreater(float(logits.grad[0,1]),0.)
        self.assertLess(float(logits.grad[0,0]),0.)
        self.assertEqual(float(logits.grad[1:].abs().sum()),0.)
        disabled,_=confidence_regularization(logits,units,[1,3,1,1],dict(cfg,joint_phase=False))
        self.assertEqual(float(disabled),0.)

    def test_confidence_normalization_independent_of_microbatch_partition(self):
        row=dict(language='en',label=1)
        units=[[row],[row],[row,row],[row,row]]
        logits=torch.tensor([[0.,v] for v in (4.,5.,6.,3.,5.,8.)],requires_grad=True)
        cfg=dict(en_real_confidence_weight=.01,en_real_confidence_margin=3.5,current_update=300,pcl_ramp_updates=300)
        all_loss,_=confidence_regularization(logits,units,[1,2,1,1],cfg)
        first,_=confidence_regularization(logits[:2],units[:2],[1,2,1,1],cfg)
        last,_=confidence_regularization(logits[2:],units[2:],[1,2,1,1],cfg)
        torch.testing.assert_close(all_loss,first+last)

    def test_batch_metadata_cannot_disagree_or_be_partial(self):
        cfg=dict(group_quotas=[6,6,6,6])
        self.assertEqual(batch_quotas([[dict(batch_group_quotas=(3,4,4,4))]],cfg),(3,4,4,4))
        with self.assertRaisesRegex(ValueError,'disagree'):
            batch_quotas([[dict(batch_group_quotas=(3,4,4,4)),dict(batch_group_quotas=(4,4,4,4))]],cfg)
        with self.assertRaisesRegex(ValueError,'Missing actual'):
            batch_quotas([[dict(batch_group_quotas=(3,4,4,4)),{}]],cfg)


if __name__ == '__main__':
    unittest.main()
