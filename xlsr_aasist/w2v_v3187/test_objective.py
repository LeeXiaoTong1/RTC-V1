"""Objective invariants: balanced CE and source-local, symmetric phone gradients."""
import math
import unittest
import torch
from .objective import (phoneme_loss, phoneme_variance, frame_span, micro_losses,
                        source_weight, auxiliary_weights, confidence_regularization, batch_quotas)
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
        loss, count = phoneme_loss(left, right, matches, scale=torch.ones(2))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(loss), .4, places=6)
        loss.backward()
        self.assertGreater(float(left.grad.norm()), 0.)
        self.assertGreater(float(right.grad.norm()), 0.)
        changed, _ = phoneme_loss(left.detach(), left.detach() * 3, matches, scale=torch.ones(2))
        self.assertAlmostEqual(float(changed), 2., places=6)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            phoneme_loss(left, right, matches)

    def test_shared_offset_cannot_hide_phone_mismatch(self):
        left=torch.tensor([[1.,2.],[2.,4.]],requires_grad=True)
        right=torch.tensor([[1.,3.],[3.,4.]],requires_grad=True)
        kwargs=dict(matches=[[0,520,0,520],[520,1000,520,1000]],scale=torch.tensor([.5,2.]))
        a,_=phoneme_loss(left,right,**kwargs)
        b,_=phoneme_loss(left+10000.,right+10000.,**kwargs)
        torch.testing.assert_close(a,b)

    def test_variance_counts_distinct_sources_and_caps_phones(self):
        x=torch.arange(40,dtype=torch.float32).reshape(20,2)/1000
        x.requires_grad_()
        records=[(str(i),x,x+.01,.25) for i in range(4)]
        loss,stats=phoneme_variance(records,torch.ones(2),{})
        self.assertEqual((stats['sources'],stats['phones'],stats['applied']),(4,32,True))
        self.assertGreater(float(loss),0.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.norm()),0.)
        duplicate,stats=phoneme_variance([records[0]]*4,torch.ones(2),{})
        self.assertEqual((stats['sources'],stats['phones'],stats['applied']),(1,8,False))
        self.assertEqual(stats['reason'],'insufficient_sources')
        self.assertEqual(float(duplicate),0.)

    def test_variance_rejects_phone_shortage_and_releases_healthy_variance(self):
        x=torch.tensor([[-2.,-2.],[2.,2.]],requires_grad=True)
        records=[(str(i),x,x,.25) for i in range(4)]
        loss,stats=phoneme_variance(records,torch.ones(2),{})
        self.assertEqual(stats['reason'],'insufficient_phones')
        self.assertEqual(float(loss),0.)
        loss,stats=phoneme_variance(records,torch.ones(2),dict(pcl_variance_min_phones=8))
        self.assertTrue(stats['applied'])
        self.assertEqual(float(loss),0.)

    def test_empty_or_short_phones_do_not_inflate_other_sources(self):
        a = torch.randn(3, 4, requires_grad=True)
        b = torch.randn(3, 4, requires_grad=True)
        value, count = phoneme_loss(a, b, [[0, 100, 0, 100]], scale=torch.ones(4))
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
        loss,count=phoneme_loss(a,b,[[0,3920,0,3920]],left_valid_frames=2,right_valid_frames=3,scale=torch.ones(4))
        self.assertEqual(count,1)
        loss.backward()
        self.assertEqual(float(a.grad[2:].abs().sum()),0.)
        self.assertEqual(float(b.grad[3:].abs().sum()),0.)
        zero,count=phoneme_loss(a,b,[[0,3920,0,3920]],left_valid_frames=0,right_valid_frames=3,scale=torch.ones(4))
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
