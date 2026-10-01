"""Processing coverage, fixed cache semantics and independent view features."""
import copy
from collections import Counter
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_aasist.data import FeatureCollator, read_wave
from w2v_v3.data import loader as old_loader
from w2v_v31.data import AudioDataset as PreviousDataset
from w2v_v31.test_data_step import write_audio
from .augment import OPERATORS, apply_plan, process_wave, processing_plan, processing_settings
from .data import AudioDataset, ViewCollator, build_data, loader

torch.set_num_threads(1)


def noisy_row(row, wave, version=0):
    return dict(row, noisy=True, full_length=True, output_samples=len(wave),
                source_sha256='same-official-source', version=version)


class ProcessingTests(unittest.TestCase):
    def test_default_draw_has_unchanged_single_and_two_operator_coverage(self):
        plans = [processing_plan(str(i), i%2, seed=1234, epoch=1) for i in range(4096)]
        counts = Counter(len(plan) for plan in plans)
        for count, probability in ((0,.5),(1,.4),(2,.1)):
            self.assertAlmostEqual(counts[count]/len(plans), probability, delta=.03)
        self.assertEqual({p['operator'] for plan in plans for p in plan}, set(OPERATORS))
        self.assertEqual(len({tuple(sorted(p['operator'] for p in plan)) for plan in plans if len(plan)==2}),3)
        for plan in plans:
            self.assertEqual(len(plan),len({p['operator'] for p in plan}))

    def test_determinism_changes_by_epoch_and_version_without_touching_global_rng(self):
        py, np_state = random.getstate(), np.random.get_state()
        def recipes(epoch,version):
            return [processing_plan(str(i),version,seed=12,epoch=epoch) for i in range(32)]
        first = recipes(1,0)
        self.assertEqual(first,recipes(1,0))
        self.assertNotEqual(first,recipes(2,0))
        self.assertNotEqual(first,recipes(1,1))
        self.assertEqual(random.getstate(),py)
        np.testing.assert_array_equal(np.random.get_state()[1],np_state[1])

    def test_all_operators_preserve_shape_finiteness_and_input_for_edge_lengths(self):
        plans = [
            [{'operator':'frequency_response','highpass_hz':80.,'lowpass_hz':4200.,'order':2}],
            [{'operator':'smooth_gain','knot_seconds':.5,'maximum_db':3.,'envelope_seed':8}],
            [{'operator':'local_attenuation','duration_seconds':.2,'depth_db':9.,
              'position':.4,'maximum_fraction':.1}],
        ]
        for samples in (1,16000,16001,32000,32001,120001):
            wave=np.random.default_rng(3).normal(0,.1,samples).astype(np.float32)
            before=wave.copy()
            for plan in plans + [plans[0]+plans[1], []]:
                with self.subTest(samples=samples,plan=plan):
                    result=apply_plan(wave,plan)
                    self.assertEqual(result.shape,wave.shape)
                    self.assertEqual(result.dtype,np.float32)
                    self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_array_equal(wave,before)
                    self.assertFalse(np.shares_memory(result,wave))
                    if not plan:np.testing.assert_array_equal(result,wave)

    def test_local_dip_never_erases_audio_or_touches_over_tenth_of_recording(self):
        wave=np.ones(16000,dtype=np.float32)
        result=apply_plan(wave,[dict(operator='local_attenuation',duration_seconds=.24,
                    depth_db=9.,position=.5,maximum_fraction=.1)])
        self.assertGreaterEqual(float(result.min()),10**(-9/20)-1e-6)
        self.assertLessEqual(int((result!=wave).sum()),len(wave)//10)
        self.assertTrue(np.all(result<=1))
        self.assertLess(float(np.max(np.abs(np.diff(result)))),.002)

    def test_frequency_response_attenuates_high_band_without_resampling(self):
        t=np.arange(16000)/16000
        low=np.sin(2*np.pi*1000*t).astype(np.float32)
        high=np.sin(2*np.pi*7000*t).astype(np.float32)
        plan=[dict(operator='frequency_response',highpass_hz=80.,lowpass_hz=4000.,order=2)]
        a,b=apply_plan(low,plan),apply_plan(high,plan)
        self.assertGreater(np.std(a[200:]),.65)
        self.assertLess(np.std(b[200:]),.1)
        self.assertEqual(len(a),len(t))

    def test_disabled_and_invalid_settings(self):
        wave=np.arange(100,dtype=np.float32)
        result,metadata=process_wave(wave,'id',0,seed=2,epoch=1,processing_enabled=False)
        np.testing.assert_array_equal(result,wave)
        self.assertEqual(metadata['processing_condition'],'unchanged')
        self.assertFalse(metadata['processing_applied'])
        for cfg in ({'processing_enabled':'yes'},{'processing_identity_probability':float('nan')},
                    {'processing_identity_probability':.8,'processing_single_probability':.5}):
            with self.assertRaises(ValueError):processing_settings(cfg)
        with self.assertRaises(ValueError):apply_plan(np.array([np.nan]),[])


class ProcessingDataTests(unittest.TestCase):
    def test_processes_full_wave_once_then_crops_shared_condition_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as td:
            row,wave=write_audio(Path(td),'long.wav')
            row=noisy_row(row,wave)
            before=Path(row['audio']).read_bytes()
            ds=AudioDataset([row],training=True,epoch=1,seed=12,
                            processing_identity_probability=0.,processing_single_probability=1.)
            with patch('w2v_aasist.data.read_wave',wraps=read_wave) as reader, \
                 patch('w2v_v32.data.process_wave',wraps=process_wave) as processor:
                full,short=ds[(0,0,'full',0)]
            self.assertEqual(reader.call_count,1)
            self.assertEqual(processor.call_count,1)
            self.assertEqual(len(processor.call_args.args[0]),len(wave))
            self.assertEqual(full['processing_condition'],short['processing_condition'])
            self.assertTrue(full['processing_applied'])
            start,count=short['crop_start_sample'],short['crop_samples']
            np.testing.assert_array_equal(short['wave'],full['wave'][start:start+count])
            self.assertAlmostEqual(full['view_weight']+short['view_weight'],1.)
            self.assertEqual(Path(row['audio']).read_bytes(),before)
            for a,b in zip(ds[(0,0,'full',0)],ds[(0,0,'full',0)]):
                np.testing.assert_array_equal(a['wave'],b['wave'])

    def test_same_source_and_version_have_identical_processing_for_both_classes_languages(self):
        with tempfile.TemporaryDirectory() as td:
            row,wave=write_audio(Path(td),'shared.wav')
            rows=[dict(noisy_row(row,wave),label=label,language=lang)
                  for label in (0,1) for lang in ('en','zh')]
            ds=AudioDataset(rows,training=True,epoch=2,processing_identity_probability=0.,
                            processing_single_probability=0.)
            actual=[ds[(i,i,'full',i)] for i in range(4)]
            for views in actual[1:]:
                for a,b in zip(actual[0],views):
                    self.assertEqual(a['processing_parameters'],b['processing_parameters'])
                    np.testing.assert_array_equal(a['wave'],b['wave'])
            self.assertEqual(len({v[0]['source_group'] for v in actual}),4)

    def test_ordinary_rawboost_is_unchanged_and_processing_never_runs(self):
        with tempfile.TemporaryDirectory() as td:
            row,wave=write_audio(Path(td),'ordinary.wav')
            arguments=dict(training=True,epoch=1,seed=2,rawboost=5,rawboost_probability=1.)
            with patch('utils.data_utils.process_rawboost_feature',side_effect=lambda w,*_:w*2), \
                 patch('w2v_v32.data.process_wave',side_effect=AssertionError('No processing on ordinary')):
                new=AudioDataset([row],**arguments)[0]
                old=PreviousDataset([row],**arguments)[0]
            for a,b in zip(new,old):
                np.testing.assert_array_equal(a['wave'],b['wave'])
                self.assertEqual(a['augmentation'],'rawboost')
                self.assertEqual(a['processing_condition'],'ordinary_unchanged')
                self.assertFalse(a['processing_applied'])

    def test_processing_precedes_independent_feature_normalization(self):
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ssl=root/'ssl';SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            row,wave=write_audio(root,'features.wav',samples=128000)
            ds=AudioDataset([noisy_row(row,wave)],training=True,epoch=1,
                            short_min_seconds=3.,short_max_seconds=3.,short_prefix_probability=1.,
                            processing_identity_probability=0.,processing_single_probability=1.)
            views=ds[(0,0,'full',0)]
            full,short=ViewCollator(ssl)([views])
            expected=FeatureCollator(ssl)([views[1]])[0]
            torch.testing.assert_close(short['features'],expected['features'],rtol=0,atol=0)
            self.assertFalse(torch.allclose(full['features'][:,:short['features'].shape[1]],
                                           short['features'],rtol=1e-4,atol=1e-4))

    def test_short_recording_not_repeated_and_validation_exactly_unchanged(self):
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ssl=root/'ssl';SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            row,wave=write_audio(root,'short.wav',samples=12800)
            views=AudioDataset([noisy_row(row,wave)],training=True,epoch=1)[0]
            self.assertEqual(len(views),1)
            self.assertEqual(len(views[0]['wave']),len(wave))
            self.assertEqual(views[0]['view_weight'],1.)
            cfg=dict(ssl_path=str(ssl),seed=2,workers=0,rawboost=0,raw_config={},eval_batch=2)
            with patch('w2v_v32.data.process_wave',side_effect=AssertionError('No validation augmentation')):
                new=next(iter(loader([row],cfg,training=False)))
                old=next(iter(old_loader([row],cfg,training=False)))
            self.assertEqual(set(new[0]),set(old[0]))
            torch.testing.assert_close(new[0]['features'],old[0]['features'],rtol=0,atol=0)

    def test_loader_pins_cuda_only_and_uses_requested_prefetch_without_cache_changes(self):
        cfg=dict(ssl_path='unused',seed=2,workers=2,rawboost=0,raw_config={},
                 eval_batch=2,device='cuda:0',prefetch_factor=3)
        actual=loader([],cfg,training=True)
        self.assertTrue(actual.pin_memory)
        self.assertEqual(actual.prefetch_factor,3)
        cfg['device']='cpu'
        self.assertFalse(loader([],cfg,training=True).pin_memory)
        sentinel=object()
        with patch('w2v_v32.data.full_build_data',return_value=sentinel) as checked:
            self.assertIs(build_data(cfg),sentinel)
            checked.assert_called_once_with(cfg)


if __name__=='__main__':
    unittest.main()
