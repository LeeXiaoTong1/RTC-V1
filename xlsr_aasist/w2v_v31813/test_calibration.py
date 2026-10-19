from contextlib import redirect_stdout
import copy
import io
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from w2v_v31810.common import capture_rng
from w2v_v31810.test_meta import assert_state_equal
from .calibration import calibrate
from .test_core import cfg


class MixedHead(nn.Module):
    def __init__(self):super().__init__();self.precision='fp16'
    def forward(self,x):return x.half() if self.precision=='fp16' else x


class Observer(nn.Module):
    def __init__(self):
        super().__init__();self.fusion=nn.Linear(1,1,bias=False);self.head=MixedHead()
        with torch.no_grad():self.fusion.weight.fill_(1.)


def components(model,units,settings,component):
    z=model.head(model.fusion(torch.ones(1,1))).float().sum()
    loss=F.softplus(z-24) if component=='prior' else z.square()
    yield component,loss


class CalibrationTests(unittest.TestCase):
    def test_reproduces_half_underflow_and_fp32_observer_recovers_signal(self):
        model=Observer().train();before=copy.deepcopy(model.state_dict());rng=capture_rng()
        _,loss=next(components(model,[],{},'prior'))
        old=torch.autograd.grad(loss,tuple(model.fusion.parameters()))[0]
        self.assertGreater(float(loss.detach()),0.);self.assertEqual(float(old.abs().sum()),0.)
        model.fusion.weight.grad=torch.ones_like(model.fusion.weight)*7
        with patch('w2v_v31813.step.components',components),redirect_stdout(io.StringIO()):
            result=calibrate(model,[[]]*6,cfg(backend_precision='fp16'))
        self.assertGreater(result['median_gradient_norms']['prior'],1e-12)
        self.assertEqual(result['weights']['prior'],.1)
        self.assertEqual(model.head.precision,'fp16');self.assertTrue(model.training)
        self.assertTrue(torch.equal(model.fusion.weight.grad,torch.ones_like(model.fusion.weight)*7))
        assert_state_equal(self,before,model.state_dict());assert_state_equal(self,rng,capture_rng())
        self.assertEqual(len(result['records']),6)

    def test_connected_zero_auxiliary_keeps_bounded_weights_for_future_pairs(self):
        def zeros(model,units,settings,component):
            z=model.fusion.weight.sum()
            yield component,z.square() if component=='ce' else z*0
        with patch('w2v_v31813.step.components',zeros),redirect_stdout(io.StringIO()):
            result=calibrate(Observer(),[[]],cfg())
        self.assertEqual(result['weights'],dict(prior=.1,pcl=.01))
        self.assertEqual(set(result['status'].values()),{'bounded_cap_small_gradient'})

    def test_missing_or_disconnected_pairs_still_fail(self):
        def missing(model,units,settings,component):
            if component!='prior':yield component,model.fusion.weight.square().sum()
        def disconnected(model,units,settings,component):
            yield component,torch.ones((),requires_grad=True) if component=='prior' else model.fusion.weight.square().sum()
        for function,reason in ((missing,'No usable prior calibration terms'),(disconnected,'disconnected')):
            model=Observer().train()
            with patch('w2v_v31813.step.components',function),redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,reason):
                calibrate(model,[[]],cfg())
            self.assertEqual(model.head.precision,'fp16');self.assertTrue(model.training)

    def test_nonfinite_still_fails_and_restores_precision(self):
        def invalid(model,units,settings,component):yield component,model.fusion.weight.sum()*float('nan')
        model=Observer().train();rng=capture_rng()
        with patch('w2v_v31813.step.components',invalid),self.assertRaises(FloatingPointError):calibrate(model,[[]],cfg())
        self.assertEqual(model.head.precision,'fp16');self.assertTrue(model.training)
        assert_state_equal(self,rng,capture_rng())


if __name__=='__main__':unittest.main()
