"""Coverage, augmentation, pre-delete validation and CLI production wiring."""
from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from w2v_aasist.runtime import sha256
from .data import FullEpochPlan, AudioDataset, class_weights
from .config import parser, configuration
from . import workflow


def corpus():
    ordinary, noisy = [], []
    for label, number in ((0,83),(1,19)):
        for i in range(number):
            r = dict(id=f'{label}/{i}',label=label,language='en' if i%3==0 else 'zh',
                     domain='offline' if i%2==0 else 'online',noisy=False)
            ordinary.append(r)
            if r['domain']=='offline':
                noisy += [dict(r,noisy=True,full_length=True,version=v) for v in (0,1)]
    return ordinary,noisy


class CoverageTests(unittest.TestCase):
    def test_uneven_strata_full_coverage_no_duplicate_or_dropped_views(self):
        ordinary,noisy = corpus()
        plan = FullEpochPlan(ordinary,[noisy],16,16,123)
        previous = None
        for epoch in (1,2,3):
            batches = plan.batches(epoch)
            raw = [x for b in batches for x in b if isinstance(x,int)]
            views = [x[0] for b in batches for x in b if isinstance(x,tuple)]
            self.assertEqual(Counter(raw),Counter(range(len(ordinary))))
            self.assertEqual(Counter(views),Counter(range(len(ordinary),len(ordinary)+len(noisy))))
            self.assertTrue(all(any(isinstance(x,int) for x in b) and any(isinstance(x,tuple) for x in b) for b in batches))
            self.assertEqual(batches,plan.batches(epoch))
            if previous is not None: self.assertNotEqual(batches,previous)
            previous = batches
        for rows,weights in ((ordinary,class_weights(ordinary)),(noisy,plan.noisy_weights)):
            mass = [sum(float(weights[y]) for r in rows if r['label']==y)/len(rows) for y in (0,1)]
            self.assertAlmostEqual(mass[0],.5,places=6);self.assertAlmostEqual(mass[1],.5,places=6)
        with self.assertRaises(ValueError): FullEpochPlan(ordinary,[noisy[:-1]])

    def test_full_audio_rawboost_gate_and_noisy_never_reaugmented(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'full.wav'
            waveform = np.random.default_rng(2).normal(0,.02,80000).astype(np.float32)
            sf.write(p,waveform,16000,subtype='FLOAT')
            row = dict(id='offline/en/x.wav',audio=str(p),language='en',label=1,domain='offline',noisy=False,band=-1)
            ds = AudioDataset([row],training=True,rawboost=5,raw_config={},rawboost_probability=0.)
            np.testing.assert_array_equal(ds[0]['wave'],waveform)
            ds.rawboost_probability=1.
            with patch('utils.data_utils.process_rawboost_feature',side_effect=lambda w,*_: w*.5) as augment:
                np.testing.assert_array_equal(ds[0]['wave'],waveform*.5)
                augment.assert_called_once()
            noisy=dict(row,noisy=True,full_length=True,version=0,output_samples=len(waveform),source_sha256=sha256(p))
            ds=AudioDataset([noisy],training=True,rawboost=5,raw_config={},rawboost_probability=1.)
            with patch('utils.data_utils.process_rawboost_feature') as augment:
                result=ds[(0,0,'full',1)]
                augment.assert_not_called()
            self.assertEqual(result['composition'],'full_v0')
            np.testing.assert_array_equal(result['wave'],waveform)


class WorkflowTests(unittest.TestCase):
    def test_config_requires_original_sha_and_records_independent_component_recipe(self):
        with tempfile.TemporaryDirectory() as td:
            baseline=Path(td)/'baseline.pt';baseline.write_bytes(b'protected')
            args=parser().parse_args(['--baseline',str(baseline),'--full-noisy-cache',str(Path(td)/'full')])
            source={k:str(Path(td)/k) for k in ('train_protocol','dev_protocol','train_data_path',
                'dev_data_path','ssl_path','dev_noisy_cache','dev_heldout_cache')}
            with self.assertRaises(ValueError):configuration(source,args)
            with patch('w2v_v3.config.BASELINE_SHA256',sha256(baseline)):
                cfg=configuration(source,args)
            self.assertIsNone(cfg['warm_checkpoint'])
            self.assertEqual((cfg['head_epochs'],cfg['joint_epochs']),(1,5))
            self.assertEqual((cfg['ordinary_batch'],cfg['noisy_batch']),(16,16))
            self.assertEqual(cfg['rawboost_probability'],.5)
            self.assertEqual(cfg['decision_threshold'],.5)

    def run_preparation(self, root, fail=False, keep=False):
        original=root/'original.pt';original.write_bytes(b'original')
        active=root/'active.json';active.write_text('{}')
        cfg=dict(baseline=str(original),baseline_sha256=sha256(original),train_caches=[str(root/'full')])
        plan=SimpleNamespace(coverage=lambda:{'ordinary_rows_per_epoch':4,'noisy_rows_per_epoch':8})
        steps=[]
        def read(_):
            steps.append('reader')
            if fail: raise ValueError('invalid replacement')
            return plan,{},torch.ones(2),torch.tensor([2,2]),{str(active):sha256(active)}
        argv=['workflow','--prepare-only']+(['--keep-old-caches'] if keep else [])
        with patch('sys.argv',argv),patch.object(workflow,'ROOT',root), \
             patch.object(workflow,'check_environment'),patch.object(workflow,'ensure_idle'), \
             patch.object(workflow,'find_source',return_value=(root/'old.json',{})), \
             patch.object(workflow,'configuration',return_value=cfg), \
             patch.object(workflow,'find_ffmpeg',return_value='ffmpeg'), \
             patch.object(workflow,'prepare',side_effect=lambda *a:steps.append('prepare')), \
             patch('w2v_v3.data.build_data',side_effect=read), \
             patch.object(workflow,'retire',side_effect=lambda *a:steps.append('retire')):
            if fail:
                with self.assertRaisesRegex(ValueError,'invalid replacement'):workflow.main()
            else:workflow.main()
        self.assertEqual(original.read_bytes(),b'original')
        return steps

    def test_cleanup_only_after_actual_reader_validated_and_optional_keep(self):
        for fail,keep,wanted in ((True,False,['prepare','reader']),
                                 (False,False,['prepare','reader','retire']),
                                 (False,True,['prepare','reader'])):
            with self.subTest(fail=fail,keep=keep), tempfile.TemporaryDirectory() as td:
                self.assertEqual(self.run_preparation(Path(td),fail,keep),wanted)

    def test_v3_progress_supervisor_forwards_to_v3_without_extra_run_flag(self):
        import live_progress
        with patch('live_progress.subprocess.call',return_value=0) as call,patch('live_progress.publish'):
            live_progress.run_job('v3',['--','--prepare-only'])
        self.assertEqual(call.call_args.args[0][-2:],['w2v_v3.workflow','--prepare-only'])


if __name__=='__main__':unittest.main()
