import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .model import Residual, CachedDetector, Detector, evidence, global_statistics
from w2v_v31810.test_partial_lora import detector, waves


class ModelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)

    def test_zero_residual_exact_parent_one_encoder_and_original_classifier(self):
        parent=detector(checkpoints=False).requires_grad_(False).eval()
        x=waves((12,23,160));original=parent(x)
        with patch.object(parent,'encode',wraps=parent.encode) as enc:
            g,l,z=evidence(parent,x)
        self.assertEqual(enc.call_count,1)
        torch.testing.assert_close(z,original,rtol=0,atol=0)
        self.assertEqual(g.shape,(3,256));self.assertEqual(l.shape,(3,384))
        cached=CachedDetector(parent.head.aasist.out_layer,Residual()).eval()
        torch.testing.assert_close(cached(g,l),original,rtol=0,atol=0)
        live=Detector(parent,Residual()).eval()
        torch.testing.assert_close(live(x),original,rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for p in live.parent.parameters()))

    def test_nonzero_residual_cached_live_and_short_inputs_match(self):
        parent=detector(checkpoints=False).requires_grad_(False).eval()
        from w2v_v3189.data import ready
        x=[np.arange(350,dtype=np.float32)/1000]+waves((18,150))
        g,l,_=evidence(parent,[ready(w) for w in x],[len(w) for w in x])
        residual=Residual().eval()
        with torch.no_grad():residual.net[-1].weight.normal_(0,.002)
        cached=CachedDetector(parent.head.aasist.out_layer,copy.deepcopy(residual)).eval()
        live=Detector(parent,residual).eval()
        torch.testing.assert_close(live(x),cached(g,l),rtol=1e-6,atol=1e-6)
        self.assertTrue(torch.equal(g[0],torch.zeros(256)))

    def test_global_moments_use_full_valid_sequence_and_constant_std_zero(self):
        a=torch.arange(30*128,dtype=torch.float32).reshape(30,128)
        b=torch.ones(12,128);b[3:]=999
        result=global_statistics([a,b],[400+29*320,400+2*320])
        torch.testing.assert_close(result[0,:128],a.mean(0))
        torch.testing.assert_close(result[0,128:],a.std(0,correction=0))
        self.assertTrue(torch.equal(result[1,:128],torch.ones(128)))
        self.assertTrue(torch.equal(result[1,128:],torch.zeros(128)))

    def test_exception_removes_hook_and_wrong_classifier_rejected(self):
        parent=detector(checkpoints=False).requires_grad_(False).eval()
        def fail(g,l):raise RuntimeError('fixture failure')
        with self.assertRaisesRegex(RuntimeError,'fixture failure'):evidence(parent,waves(),correction=fail)
        self.assertFalse(parent.head.aasist.out_layer._forward_pre_hooks)
        parent.head.aasist.out_layer=torch.nn.Linear(320,2)
        with self.assertRaises(ValueError):evidence(parent,waves())

    def test_step_updates_only_residual_and_parent_dropout_stays_eval(self):
        parent=detector(checkpoints=False).requires_grad_(False).eval()
        before={k:v.clone() for k,v in parent.state_dict().items()}
        live=Detector(parent,Residual()).train()
        self.assertFalse(parent.training)
        self.assertTrue(all(not m.training for m in parent.modules()))
        g,l,_=evidence(parent,waves((12,19)))
        model=CachedDetector(parent.head.aasist.out_layer,live.residual)
        optim=torch.optim.SGD(model.residual.parameters(),lr=.01)
        torch.nn.functional.cross_entropy(model(g,l),torch.tensor([0,1])).backward();optim.step()
        self.assertTrue(torch.count_nonzero(model.residual.net[-1].weight)>0)
        self.assertTrue(all(p.grad is None for p in parent.parameters()))
        self.assertTrue(all(p.grad is None for p in model.classifier.parameters()))
        for k,v in before.items():torch.testing.assert_close(v,parent.state_dict()[k],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
