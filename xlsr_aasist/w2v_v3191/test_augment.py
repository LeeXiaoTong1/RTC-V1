import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .mechanisms import (TRAIN_MECHANISMS, HELDOUT_MECHANISMS, apply_pair,
    apply_view, mechanism_catalog, model_provenance, quality_check)


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(29)
        self.wave = (rng.standard_normal(32000)*.06).astype(np.float32)
        self.noisy = self.wave+(rng.standard_normal(len(self.wave))*.01).astype(np.float32)

    def test_filters_preserve_length_finite_and_reset_state(self):
        for mechanism in ('spectral_color', 'packet_repeat', 'spectral_wiener', 'color_packet'):
            with self.subTest(mechanism=mechanism):
                a, ma = apply_view(self.wave, mechanism, 10)
                apply_view(self.wave*.3, mechanism, 31)
                b, mb = apply_view(self.wave, mechanism, 10)
                self.assertEqual(a.shape, self.wave.shape)
                self.assertTrue(np.isfinite(a).all())
                np.testing.assert_array_equal(a, b)
                self.assertEqual(ma, mb)
                self.assertFalse(np.array_equal(a, self.wave))

    def test_matched_pair_uses_same_parameters_and_independent_state(self):
        for mechanism in ('spectral_color', 'packet_repeat', 'spectral_wiener', 'color_packet'):
            a, b, meta = apply_pair(self.wave, self.wave.copy(), mechanism, 137)
            np.testing.assert_array_equal(a, b)
            self.assertTrue(meta['shared_parameters'])
            self.assertTrue(meta['fresh_state_per_view'])

    def test_packet_replacement_changes_only_declared_spans(self):
        out, meta = apply_view(self.wave, 'packet_repeat', 5)
        changed = np.flatnonzero(out != self.wave)
        self.assertGreater(len(changed), 0)
        spans = meta['affected_spans']
        mask = np.zeros(len(out), bool)
        for a, b in spans:
            mask[a:b] = True
        self.assertTrue(mask[changed].all())
        self.assertLessEqual(len(changed), len(out)*.05)
        np.testing.assert_array_equal(out[:320], self.wave[:320])

    def test_short_inputs_and_silence_do_not_break_processing(self):
        for size in (1, 63, 400, 999):
            for mechanism in ('spectral_color', 'packet_repeat', 'spectral_wiener'):
                x = np.zeros(size, np.float32)
                output, _ = apply_view(x, mechanism, 100)
                self.assertEqual(len(output), size)
                self.assertTrue(np.isfinite(output).all())

    def test_quality_rejects_erasure_nonfinite_clipping_and_geometry(self):
        self.assertTrue(quality_check(self.wave, self.wave, self.noisy)['passed'])
        self.assertIn('active_energy_lost', quality_check(self.wave, self.wave,
            np.zeros_like(self.wave))['reasons'])
        self.assertIn('clipping', quality_check(self.wave, self.wave,
            np.ones_like(self.wave))['reasons'])
        bad = self.wave.copy(); bad[0] = np.nan
        self.assertEqual(quality_check(self.wave, self.wave, bad)['reason'], 'nonfinite')
        self.assertEqual(quality_check(self.wave, self.wave, self.noisy[:-1])['reason'], 'length')

    def test_heldout_algorithms_cannot_enter_train_catalog(self):
        catalog = mechanism_catalog()
        self.assertFalse(set(catalog['train']) & set(catalog['heldout']))
        self.assertNotEqual(catalog['implementations']['spectral_wiener'],
                            catalog['implementations']['spectral_color'])
        with self.assertRaises(ValueError):
            mechanism_catalog(dict(robust_train_mechanisms=['spectral_wiener']))

    def test_neural_requires_pinned_asset_and_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, 'pinned'):
            apply_view(self.wave, 'rnnoise', 1, {})
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/'model.rnnn'
            model.write_bytes(b'example model')
            cfg = dict(rnnoise_model=str(model),
                rnnoise_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), ffmpeg='ffmpeg')
            self.assertEqual(model_provenance(cfg)['bytes'], 13)
            with patch('w2v_v3191.mechanisms.subprocess.run') as run:
                run.return_value.returncode = 1
                run.return_value.stderr = b'missing arnndn'
                with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                    apply_view(self.wave, 'rnnoise', 1, cfg)
            model.write_bytes(b'changed model')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                model_provenance(cfg)

    def test_neural_unknown_delay_disables_alignment_instead_of_reusing_phone_spans(self):
        with patch('w2v_v3191.mechanisms._neural',
                return_value=(np.r_[self.wave, np.zeros(1600)], {'sha256': 'test'})), \
             patch('w2v_v3186.augment.estimate_delay',
                return_value=dict(valid=False, offset=0., score=0.)):
            output, meta = apply_view(self.wave, 'rnnoise', 1, {})
        np.testing.assert_array_equal(output, self.wave)
        self.assertIn([0, len(self.wave)], meta['affected_spans'])
        self.assertFalse(meta['alignment']['valid'])

    def test_weak_neural_alignment_uses_startup_delay_without_clipping_last_speech(self):
        delayed=np.r_[np.zeros(160),self.wave,np.zeros(1600)]
        cfg=dict(robust_runtime=dict(rnnoise_smoke=dict(
            alignment=dict(valid=True,offset=160.))))
        with patch('w2v_v3191.mechanisms._neural',
                return_value=(delayed,{'sha256':'test'})), \
             patch('w2v_v3186.augment.estimate_delay',
                return_value=dict(valid=False,offset=0.,score=0.)):
            output,meta=apply_view(self.wave,'rnnoise',1,cfg)
        np.testing.assert_array_equal(output,self.wave)
        self.assertEqual(meta['alignment']['startup_delay_fallback'],160)
        self.assertIn([0,len(self.wave)],meta['affected_spans'])


if __name__ == '__main__':
    unittest.main()
