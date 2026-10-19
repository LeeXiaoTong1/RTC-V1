"""Synthetic CPU acceptance of all three frozen feature capture points."""
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from .probes import _extract, _fit_one, _calibration


class MockFusion:
    layers=(1,2,3,4)
    def project_layer(self,hidden,index,weights=None):
        return hidden[:,:,:128] if hidden.shape[-1]>=128 else hidden


class MockHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.aasist=nn.Module()
        self.aasist.out_layer=nn.Linear(384,2)


class MockDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion=MockFusion()
        self.cfg={'encoder_dim':256}
        self.head=MockHead()
    def encode(self,waves):
        # Model's actual stream invokes project_layer final exactly once.
        n=[(len(w)-400)//320+1 for w in waves]
        z=torch.ones((len(waves),max(n),256),dtype=torch.float32)
        z[:,:,0]=torch.arange(max(n))
        self.fusion.project_layer(z,4)
        return [z[i,:k,:128] for i,k in enumerate(n)]
    def classify_sequences(self,sequences):
        result=[]
        for seq in sequences:
            v=torch.cat([seq.mean(0),seq.std(0,unbiased=False),torch.ones(128)])
            result.append(v)
        return self.head.aasist.out_layer(torch.stack(result))


class FeatureTests(unittest.TestCase):
    def test_last_fused_and_preclassifier_hooks(self):
        fake=types.ModuleType('w2v_v31810.model')
        fake.Detector=MockDetector
        parent=types.ModuleType('w2v_v31810');parent.__path__=[]
        with patch.dict(sys.modules,{'w2v_v31810':parent,'w2v_v31810.model':fake}):
            model=MockDetector().eval()
            before=model.head.aasist.out_layer.weight.detach().clone()
            result=_extract(model,[np.zeros(6400,np.float32),np.zeros(7040,np.float32)])
            self.assertEqual(result['ssl_last'].shape,(2,512))
            self.assertEqual(result['fused'].shape,(2,256))
            self.assertEqual(result['preclassifier'].shape,(2,384))
            self.assertEqual(result['logits'].shape,(2,2))
            self.assertTrue(torch.equal(before,model.head.aasist.out_layer.weight))
            self.assertEqual(len(model.head.aasist.out_layer._forward_pre_hooks),0)
            self.assertEqual(model.fusion.project_layer.__func__,MockFusion.project_layer)

    def test_train_only_calibration_and_cv_uses_group_ids(self):
        rng=np.random.default_rng(27);n=48
        label=np.array([0,1]*(n//2));language=np.array(['en','zh']*(n//2))
        group=np.array([f'G{i//2}' for i in range(n)])
        # Each origin has two views in a real task. Label must be shared.
        label=np.repeat(np.arange(n//2)%2,2)
        data=dict(id=np.asarray([f'v{i}' for i in range(n)]),canonical_source=group,
            panel=np.asarray(['official_online','regular_noisy']*(n//2)),language=language,
            label=label,logits=np.column_stack(((1-2*label)*2+rng.normal(0,.3,n),np.zeros(n))),
            ssl_last=rng.normal(size=(n,16)),fused=rng.normal(size=(n,8)),
            preclassifier=rng.normal(size=(n,12)))
        for k in ('ssl_last','fused','preclassifier'):
            data[k][:,0]+=(1-2*label)*2
        bias=_calibration(data,data,None,seed=4)
        self.assertIn('train_holdout',bias)
        self.assertNotIn('historical_audit_descriptive',bias)
        fitted=_fit_one('fused',data,data,None,seed=4)
        self.assertEqual(fitted['dim'],8)
        self.assertEqual(fitted['train_samples'],n)
        self.assertEqual(len(fitted['cv_candidates']),3)


if __name__=='__main__':unittest.main()
