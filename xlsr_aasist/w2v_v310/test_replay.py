"""Do not confuse cached-runtime drift with an adapter changing the detector."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v39.common import read_json
from .data import bundles, probe_split
from .model import load_model
from .replay import baseline, check_initial, original_path, fp32_inference, replay_indices
from .train import infer
from .test_workflow import fixture, synthetic_expanded_bundles


class ReplayTests(unittest.TestCase):
    def test_historical_numeric_drift_does_not_replace_same_runtime_reference(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            _,cfg,_,_ = fixture(root)
            run = root/'replay'; run.mkdir()
            model = load_model(cfg)
            with synthetic_expanded_bundles(cfg) as (train,dev):
                _,probe,_,split = probe_split(train['rows'],cfg)
                # Same data/checkpoint, historical execution yields shifted logits.
                # Old code's 1e-3 cached-logit assertion would reject this.
                historical = dict(dev,logits=np.asarray(dev['logits']).copy()+.02)
                logits,metrics,probes = baseline(model,historical,probe,split,cfg,run,infer)
                replay = read_json(run/'startup_replay.json')
                self.assertEqual(replay['status'],'passed')
                self.assertGreater(replay['historical_cache_vs_original']['max_logit_delta'],.019)
                self.assertLess(replay['original_vs_adapter']['max_logit_delta'],1e-5)
                self.assertFalse(np.allclose(logits,historical['logits'],rtol=3e-5,atol=1e-3))
                actual,_ = infer(model,dev['rows'],cfg,'test new model')
                np.testing.assert_allclose(logits,actual,rtol=3e-5,atol=1e-3)
                with patch('w2v_v310.replay.check_initial',side_effect=AssertionError('No repeated startup inference')):
                    again = baseline(model,historical,probe,split,cfg,run,
                                     lambda *a,**kw: self.fail('No repeated baseline/probe encoder passes'))
                np.testing.assert_array_equal(logits,again[0])
                self.assertEqual(metrics,again[1]); self.assertEqual(probes,again[2])
                (run/'baseline_probes.json').write_text('{}',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Pinned input changed'):
                    baseline(model,historical,probe,split,cfg,run,infer)

    def test_actual_adapter_or_runtime_change_still_fails_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            _,cfg,_,_ = fixture(root)
            run = root/'replay'; run.mkdir()
            model = load_model(cfg)
            with bundles(cfg) as (_,dev):
                with torch.no_grad():
                    model.head.classifier.adapter[-1].bias.fill_(.1)
                with self.assertRaisesRegex(ValueError,'not zero-initialized'):
                    check_initial(model,dev,cfg,run,infer)
                self.assertEqual(read_json(run/'startup_replay.json')['reason'],'adapter_not_zero_initialized')
                with torch.no_grad():
                    model.head.classifier.adapter[-1].bias.zero_()
                runtime = type(model)
                forward = runtime.forward
                def wrong_path(self,*args,**kwargs):
                    scores,hidden = forward(self,*args,**kwargs)
                    return scores+.02,hidden
                with patch.object(runtime,'forward',wrong_path):
                    with self.assertRaisesRegex(ValueError,'differs from original under identical execution'):
                        check_initial(model,dev,cfg,run,infer)
                recorded = read_json(run/'startup_replay.json')
                self.assertEqual(recorded['status'],'failed')
                self.assertGreater(recorded['original_vs_adapter']['max_logit_delta'],.019)

    def test_reference_context_restores_parameter_identity_and_inference_policy(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            _,cfg,_,records = fixture(Path(temp))
            model = load_model(cfg)
            classes = type(model),type(model.head),type(model.head.classifier)
            before = {name:(id(p),p.requires_grad) for name,p in model.named_parameters()}
            with self.assertRaisesRegex(RuntimeError,'test exception'):
                with original_path(model):
                    self.assertFalse(model.training)
                    raise RuntimeError('test exception')
            self.assertTrue(model.training)
            self.assertEqual(classes,(type(model),type(model.head),type(model.head.classifier)))
            self.assertEqual(before,{name:(id(p),p.requires_grad) for name,p in model.named_parameters()})
            old = torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
            with torch.autocast('cpu',dtype=torch.bfloat16):
                scores,_ = infer(model,records['dev'],cfg,'test nested precision',capture=True)
                self.assertEqual(scores.dtype,np.float32)
                with fp32_inference('cpu'):
                    value = torch.randn(2,3) @ torch.randn(3,2)
                    self.assertEqual(value.dtype,torch.float32)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(old,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))
            self.assertEqual(len(model.head.classifier[-1]._forward_pre_hooks),0)
            chosen = replay_indices(records['dev'])
            self.assertEqual({(records['dev'][i]['condition'],records['dev'][i]['language'],records['dev'][i]['label']) for i in chosen},
                             {(r['condition'],r['language'],r['label']) for r in records['dev']})

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA same-runtime replay runs during server setup')
    def test_cuda_original_and_adapter_replay_matches_under_explicit_fp32(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            _,cfg,_,_ = fixture(root)
            cfg['device'] = 'cuda:0'
            model = load_model(cfg)
            run = root/'replay'; run.mkdir()
            with bundles(cfg) as (_,dev):
                report = check_initial(model,dev,cfg,run,infer)
            self.assertEqual(report['status'],'passed')
            self.assertEqual(report['original_vs_adapter']['decision_changes'],0)


if __name__=='__main__':
    unittest.main()
