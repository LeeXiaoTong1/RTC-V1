import unittest

import numpy as np

from w2v_v315.augment import settings_grid
from w2v_v39.metrics import measure as historical_measure
from .diagnostics import build_panel, panel_partition, panel_metrics, paired_prediction_changes
from .metrics import ap_eer, binary_metrics, measure


def originals(count=140):
    rows = []
    for language in ('en', 'zh'):
        for label in (0, 1):
            for i in range(count):
                key = f'{language}:{label}:{i}'
                rows.append(dict(split='dev', condition='offline', group_id=key,
                    source_id=key, source_sha256=key, audio_sha256=key, language=language, label=label))
    return rows


class DiagnosticTests(unittest.TestCase):
    def test_bounded_balanced_disjoint_and_order_invariant(self):
        cfg = dict(panel_seed=123, panel_sources_per_group=128)
        a, ar, am = build_panel(originals(), cfg)
        b, br, bm = build_panel(list(reversed(originals())), cfg)
        self.assertEqual((a, ar, am), (b, br, bm))
        self.assertEqual(len(a), 512)
        tune, tr = panel_partition(a, ar, 'tune')
        audit, audit_recipes = panel_partition(a, ar, 'audit')
        self.assertEqual(len(tune), 256)
        self.assertFalse({r['group_id'] for r in tune} & {r['group_id'] for r in audit})
        self.assertEqual({r['family'] for r in audit_recipes},{'g711_mulaw','g711_alaw'})
        self.assertFalse({r['family'] for r in audit_recipes} & {r['family'] for r in tr})
        audit_metrics=panel_metrics(audit,audit_recipes,np.array([[1-r['label'],r['label']] for r in audit]))
        self.assertTrue(audit_metrics['complete']);self.assertEqual(audit_metrics['macro_f1'],1.)
        for language in ('en', 'zh'):
            for label in (0, 1):
                self.assertEqual(sum(r['language'] == language and r['label'] == label for r in tune), 64)
        for recipe in tr:
            self.assertIn(recipe['settings'], settings_grid(recipe['family'], 'dev'))
            self.assertNotIn(recipe['settings'], settings_grid(recipe['family'], 'train'))

    def test_panel_duplicate_sources_do_not_consume_extra_budget(self):
        rows = originals(16)
        a, _, _ = build_panel(rows, dict(panel_sources=64))
        rows += [dict(rows[0], source_id='zzz-duplicate')]
        b, _, _ = build_panel(rows, dict(panel_sources=64))
        self.assertEqual(a, b)

    def test_no_train_or_inferred_source_identity_allowed(self):
        rows = originals(16)
        rows[0]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'official Dev'):
            build_panel(rows, dict(panel_sources=64))
        rows = originals(16)
        rows[0]['source_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity'):
            build_panel(rows, dict(panel_sources=64))

    def test_panel_metrics_separate_families_and_never_change_weighted(self):
        rows, recipes, _ = build_panel(originals(), {})
        rows, recipes = panel_partition(rows, recipes, 'tune')
        logits = np.array([[1-r['label'], r['label']] for r in rows])
        value = panel_metrics(rows, recipes, logits)
        self.assertEqual(value['macro_f1'], 1.)
        self.assertEqual(value['macro_auc'], 1.)
        self.assertNotIn('weighted_f1', value)
        self.assertFalse(value['included_in_weighted'])
        self.assertEqual(set(value['families']), {'ffmpeg', 'webrtc', 'light'})
        self.assertTrue(value['alerts'])  # Per-family fake support below 100 is stated.

    def test_audit_and_tune_cannot_be_pooled(self):
        rows, recipes, _ = build_panel(originals(), {})
        with self.assertRaisesRegex(ValueError, 'Never pool'):
            panel_metrics(rows, recipes, np.ones((len(rows), 2)))

    def test_historical_f1_is_bit_exact_with_more_diagnostics(self):
        rows, logits = [], []
        rng = np.random.default_rng(8)
        for condition in ('online', 'seen', 'heldout'):
            for language in ('en', 'zh'):
                for label in (0, 1):
                    for _ in range(20):
                        rows.append(dict(condition=condition, language=language, label=label))
                        logits.append(rng.normal(size=2))
        old, new = historical_measure(rows, logits), measure(rows, logits)
        for name in ('weighted_f1', 'clean_f1', 'noisy_f1', 'seen_f1', 'heldout_f1'):
            self.assertEqual(old[name], new[name])
        for name in old['groups']:
            self.assertEqual(old['groups'][name]['confusion'], new['groups'][name]['confusion'])
            self.assertIn('ap', new['groups'][name])

    def test_ranking_ties_and_missing_class(self):
        self.assertEqual(ap_eer([0, 1], [1., 0.]), dict(ap=1., eer=0.))
        self.assertEqual(ap_eer([0, 1], [0., 0.]), dict(ap=.5, eer=.5))
        self.assertEqual(ap_eer([0, 1], [0., 1.]), dict(ap=.5, eer=1.))
        self.assertEqual(ap_eer([1, 1], [0., 1.]), dict(ap=None, eer=None))
        self.assertIsNone(binary_metrics([1, 1], [0., 1.])['matched'])

    def test_equal_aggregate_does_not_hide_prediction_swaps(self):
        rows = [dict(language='en', label=1) for _ in range(2)]
        value = paired_prediction_changes(rows, [[1, 0], [0, 1]], [[0, 1], [1, 0]])
        self.assertEqual(value['decision_changes'], 2)
        self.assertEqual(value['groups']['en/real']['rescued'], 1)
        self.assertEqual(value['groups']['en/real']['newly_wrong'], 1)


if __name__ == '__main__':
    unittest.main()
