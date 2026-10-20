import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from .data import Waves, microbatches


class CandidateDataTests(unittest.TestCase):
    def setUp(self):
        self.wave = np.random.default_rng(10).normal(0, .05, 32000).astype(np.float32)
        off = dict(id='off', source_id='source', group_id='g', language='en', label=1, split='train')
        on = dict(off, id='on')
        self.cfg = dict(seed=3191001, warm_epochs=0, consistency='dual',
            rtc_activity_policy='mixed', robust_candidate_count=3,
            robust_train_mechanisms=['spectral_color', 'packet_repeat'])
        self.dataset = Waves([off, on], self.cfg)
        self.dataset.original = lambda _: self.wave.copy()
        self.dataset.bank = object(); self.dataset.engines = object()
        self.dataset._phones = lambda _: [dict(instance=0, token='a', start_sample=1000,
            end_sample=16000, posterior=.9, valid=True)]
        self.ticket = dict(index=0, kind='offline', base_variant=0, base_family='original',
            base_generation=0, base_dynamic=False, base_view_id='view', occurrence='visit',
            epoch=0, meta_side='A', canonical_source='canonical', task_kind='noise',
            domain_override=dict(scene='rain', family='webrtc'), nuisance_seed=1234,
            pcl_enabled=True, pcl_noise=True, pcl_online_index=1)

    def generate(self, ticket=None):
        def matched(source, recipe, *_):
            return source.copy(), source.copy(), dict(recipe=recipe, pair_eligible=True)
        with patch('w2v_v3191.data.generate_matched', side_effect=matched), \
             patch('w2v_v3191.data.estimate_delay', return_value=dict(valid=True, offset=0.)), \
             patch('w2v_v3191.data.reliable_spans', return_value=[True]):
            return self.dataset[ticket or self.ticket]

    def test_candidate_units_are_nonrecursive_and_keep_source_and_clean_reference(self):
        output = self.generate()
        candidates = output[0]['robust_candidates']
        self.assertEqual(len(candidates), 3)
        self.assertEqual({u[0]['robust_mechanism'] for u in candidates},
                         {'baseline', 'spectral_color', 'packet_repeat'})
        for unit in candidates:
            row = unit[0]
            self.assertNotIn('robust_candidates', row)
            self.assertEqual(row['canonical_source'], 'canonical')
            self.assertEqual(row['label'], 1)
            self.assertEqual(row['pcl_refs']['clean']['source_id'], row['source_id'])
            self.assertEqual(row['pcl_refs']['clean']['role'], 'clean_ref')
            self.assertTrue(row['robust_quality']['passed'])
        np.testing.assert_array_equal(candidates[0][0]['wave'], self.wave)
        self.assertEqual(len(pickle.loads(pickle.dumps(output))[0]['robust_candidates']), 3)

    def test_candidates_ignore_label_and_arm_and_are_reproducible(self):
        original = self.generate()[0]['robust_candidates']
        self.dataset.cfg['robust_arm'] = 'hard'
        for row in self.dataset.rows:
            row['label'] = 0
        changed = self.generate()[0]['robust_candidates']
        for a, b in zip(original, changed):
            self.assertEqual(a[0]['robust_candidate_id'], b[0]['robust_candidate_id'])
            np.testing.assert_array_equal(a[0]['wave'], b[0]['wave'])
            np.testing.assert_array_equal(a[0]['pcl_refs']['clean']['wave'],
                                          b[0]['pcl_refs']['clean']['wave'])

    def test_clean_ce_reference_still_exists_with_pcl_disabled(self):
        output = self.generate(dict(self.ticket, pcl_enabled=False))
        for candidate in output[0]['robust_candidates']:
            self.assertIn('clean', candidate[0]['pcl_refs'])
            self.assertEqual(candidate[0]['pcl_pairs'], {})
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            list(microbatches([output], 1, 10000, weights={}))
        self.assertEqual(len(list(microbatches([output], 2, 10000, weights={}))), 1)

    def test_online_is_not_processed_or_given_extra_candidates(self):
        result = self.generate(dict(self.ticket, kind='online', index=1))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['role'], 'online')
        self.assertNotIn('robust_candidates', result[0])
        self.assertNotIn('pcl_refs', result[0])
        np.testing.assert_array_equal(result[0]['wave'], self.wave)

    def test_baseline_skips_unused_dsp_without_changing_training_wave_or_pcl(self):
        self.dataset.cfg['robust_arm'] = 'baseline'
        full = self.generate(dict(self.ticket, step=0))[0]
        with patch('w2v_v3191.data.apply_pair', side_effect=AssertionError('unused DSP')):
            compact = self.generate(dict(self.ticket, step=1))[0]
        self.assertEqual(len(full['robust_candidates']), 3)
        self.assertEqual(len(compact['robust_candidates']), 1)
        a, b = full['robust_candidates'][0][0], compact['robust_candidates'][0][0]
        np.testing.assert_array_equal(a['wave'], b['wave'])
        self.assertEqual(a['pcl_pairs'], b['pcl_pairs'])
        self.assertEqual(a['pcl_coverage'], b['pcl_coverage'])
        self.assertEqual(set(a['pcl_refs']), set(b['pcl_refs']))
        for name in a['pcl_refs']:
            np.testing.assert_array_equal(a['pcl_refs'][name]['wave'], b['pcl_refs'][name]['wave'])
        self.assertEqual(compact['robust_candidate_generation'],
                         dict(requested=3, generated=1, omitted_for_baseline=True))

    def test_unpaired_dev_inference_is_still_a_single_row(self):
        row=dict(self.dataset.rows[0],wave=self.wave,role='noisy')
        self.assertEqual(len(list(microbatches([[row]],1,10000))),1)

    def test_replaced_intervals_are_excluded_from_noise_and_platform_pcl(self):
        output = self.generate()
        packet = next(u[0] for u in output[0]['robust_candidates']
                      if u[0]['robust_mechanism']=='packet_repeat')
        details = packet['robust_candidate']['processing']
        spans = details['clean']['affected_spans']
        self.assertTrue(spans)
        for a, b, c, d in packet['pcl_pairs']['noise']:
            for start, end in spans:
                self.assertFalse(start < b and end > a)
                self.assertFalse(start < d and end > c)
        for a, b, c, d in packet['pcl_pairs']['platform']:
            for start, end in spans:
                self.assertFalse(start < d and end > c)


if __name__ == '__main__':
    unittest.main()
