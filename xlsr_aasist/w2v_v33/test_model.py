"""Shared-frame interface preserves classifier/checkpoints and native masks."""
import copy
import unittest
import torch
from w2v_v3.test_model_step import tiny_detector
from .model import Detector,install_runtime

torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(331)

    def test_original_checkpoint_keys_and_eval_logits_unchanged(self):
        old=tiny_detector().eval()
        state={**old.architecture(),'model':old.state_dict()}
        current=Detector.from_checkpoint(state,checkpointing=True).eval()
        self.assertEqual(set(current.state_dict()),set(old.state_dict()))
        self.assertEqual(sum(p.numel() for p in current.parameters()),sum(p.numel() for p in old.parameters()))
        for length in (9,17,31):
            x=torch.randn(2,length,160);mask=torch.ones(2,length).long()
            with torch.no_grad():a=old(x,mask);b=current(x,mask)
            for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_install_keeps_parameter_identity_rng_and_training_logits(self):
        model=tiny_detector().configure_trainable_layers(1).train()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0
        reference=copy.deepcopy(model)
        rng=torch.get_rng_state().clone();params=dict(model.named_parameters())
        self.assertIs(install_runtime(model),model)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        self.assertTrue(all(p is params[k] for k,p in model.named_parameters()))
        x=torch.randn(2,17,160);mask=torch.ones(2,17).long()
        expected=reference(x,mask)
        seen=[];hook=model.backbone.register_forward_pre_hook(lambda *_:seen.append(1))
        got=model.forward_training(x,mask);hook.remove()
        self.assertEqual(len(seen),1)
        self.assertEqual(got[2].shape,(2,17,model.head.config.projection))
        self.assertEqual(got[3].shape,mask.shape)
        for left,right in zip(expected,got[:2]):torch.testing.assert_close(left,right,rtol=3e-5,atol=3e-6)
        got[2].square().mean().backward()
        self.assertGreater(float(model.head.blocks[-1].down.weight.grad.abs().sum()),0.)

    def test_padded_feature_return_matches_separate_native_lengths(self):
        model=install_runtime(tiny_detector()).configure_trainable_layers(1).train()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0
        a=torch.randn(1,13,160);b=torch.randn(1,19,160)
        ea=model.forward_training(a,torch.ones(1,13).long())
        eb=model.forward_training(b,torch.ones(1,19).long())
        x=torch.nn.utils.rnn.pad_sequence([a[0],b[0]],batch_first=True)
        mask=torch.arange(19)[None,:]<torch.tensor([13,19])[:,None]
        x[~mask]=float('nan')
        z,h,f,m=model.forward_training(x,mask.long())
        torch.testing.assert_close(z,torch.cat([ea[0],eb[0]]),rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(f[0,:13],ea[2][0],rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(f[1],eb[2][0],rtol=3e-5,atol=3e-6)
        self.assertEqual(float(f[0,13:].abs().sum()),0.)
        self.assertTrue(torch.equal(m,mask))


if __name__=='__main__':unittest.main()
