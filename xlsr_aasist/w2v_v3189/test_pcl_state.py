"""Train-only frozen phone scales survive checkpoints and cannot learn shortcuts."""
import copy
import unittest
import torch
from .pcl_state import PCLScale
from .objective import phoneme_features


class PCLScaleTests(unittest.TestCase):
    def test_source_visit_weighting_and_resume_are_exact(self):
        state=PCLScale(2)
        first=torch.tensor([[0.,1.],[2.,3.]],requires_grad=True)
        second=torch.tensor([[4.,5.]]*8,requires_grad=True)
        state.observe(first,'a')
        restored=PCLScale(2)
        restored.load_state_dict(copy.deepcopy(state.state_dict()))
        for value in (state,restored):
            value.observe(second,'b')
            value.freeze()
        self.assertEqual(int(state.source_visits),2)
        self.assertEqual(int(state.observed_phones),10)
        torch.testing.assert_close(state.mean,torch.tensor([2.5,3.5]))
        torch.testing.assert_close(state.scale,torch.tensor([2.75,2.75]).sqrt())
        self.assertFalse(state.scale.requires_grad)
        self.assertEqual(list(state.parameters()),[])
        for key,value in state.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(restored.is_frozen)
        with self.assertRaisesRegex(RuntimeError,'Frozen'):
            restored.observe(first,'a')

    def test_no_dev_calibration_and_zero_variance_fails_closed(self):
        state=PCLScale(2)
        with self.assertRaisesRegex(ValueError,'Train'):
            state.observe(torch.ones(8,2),'dev',split='dev')
        with self.assertRaisesRegex(RuntimeError,'two valid'):
            state.freeze()
        state.observe(torch.ones(8,2),'a')
        state.observe(torch.ones(8,2),'b')
        with self.assertRaisesRegex(RuntimeError,'no usable variance'):
            state.freeze()

    def test_direct_buffer_restore_refreshes_frozen_cache(self):
        first=PCLScale(2)
        for source in ('a','b'):
            first.observe(torch.tensor([[0.,1.],[1.,2.]]),source)
        first.freeze()
        restored=PCLScale(2)
        with torch.no_grad():
            for name,value in first.state_dict().items():
                restored.state_dict()[name].copy_(value)
        self.assertFalse(restored.is_frozen)
        restored.sync_state()
        self.assertTrue(restored.is_frozen)

    def test_scale_floor_prevents_tiny_channel_amplification(self):
        state=PCLScale(3)
        values=torch.tensor([[-1.,0.,-2.],[1.,0.,2.]])
        state.observe(values,'a');state.observe(values,'b');state.freeze()
        self.assertGreaterEqual(float(state.scale[1]),.1)
        self.assertTrue(bool(torch.isfinite(state.scale).all()))

    def test_calibration_pooling_excludes_reflected_frames(self):
        frames=torch.arange(24,dtype=torch.float32).reshape(12,2)
        pooled=phoneme_features(frames,[[0,3920]],valid_frames=2)
        torch.testing.assert_close(pooled,frames[:2].mean(0,keepdim=True))
        empty=phoneme_features(frames,[[0,3920]],valid_frames=0)
        self.assertEqual(tuple(empty.shape),(0,2))


if __name__=='__main__':unittest.main()
