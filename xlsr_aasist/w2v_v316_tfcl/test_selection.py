import copy
import unittest

import numpy as np

from w2v_v39.metrics import GROUPS
from .selection import acceptance, guarded_score, update_selections


def fixed():
    return dict(complete=True, weighted_f1=.966, clean_f1=.98, noisy_f1=.96,
        groups={name: dict(class_counts=[400, 100], recall=[.995, .85], auc=.98) for name in GROUPS},
        matched={name: dict(real_recall=.8) for name in GROUPS})


def panel(f1=.60, auc=.90, matched=.70, partition='tune'):
    return dict(partition=partition, complete=True, views=256, macro_f1=f1, macro_auc=auc,
        groups={language: dict(class_counts=[64, 64], recall=[.90, .70], auc=auc)
            for language in ('en', 'zh')},
        macro_matched_real=matched, families={name: dict(macro_f1=f1, auc=auc,
        class_counts=[42, 42]) for name in ('ffmpeg', 'webrtc', 'light')})


class SelectionTests(unittest.TestCase):
    def test_weak_baseline_nonregression_alone_is_insufficient(self):
        ok, reasons = acceptance(fixed(), fixed(), {}, panel(), panel())
        self.assertFalse(ok)
        self.assertIn('no_broader_panel_improvement', reasons)

    def test_panel_improvement_can_win_without_historical_weighted_increase(self):
        candidate = fixed()
        candidate['weighted_f1'] -= .0004
        self.assertTrue(acceptance(fixed(), candidate, {}, panel(), panel(.62, .904))[0])

    def test_threshold_shift_needs_ranking_evidence(self):
        ok, reasons = acceptance(fixed(), fixed(), {}, panel(), panel(.62))
        self.assertFalse(ok)
        self.assertIn('no_ranking_improvement_beyond_boundary_shift', reasons)
        candidate = fixed()
        for name in ('seen/en', 'heldout/en'):
            candidate['matched'][name]['real_recall'] += .01
        self.assertTrue(acceptance(fixed(), candidate, {}, panel(), panel(.62))[0])

    def test_audit_partition_never_selects_model(self):
        with self.assertRaisesRegex(ValueError, 'audit'):
            acceptance(fixed(), fixed(), {}, panel(partition='audit'), panel(.62, .904, partition='audit'))

    def test_mechanism_collapse_cannot_hide_behind_improved_average(self):
        candidate = panel(.62, .91)
        candidate['families']['webrtc']['macro_f1'] = .50
        self.assertIn('webrtc_panel_f1_collapse', acceptance(fixed(), fixed(), {}, panel(), candidate)[1])

    def test_two_regressing_mechanisms_block_one_lucky_engine(self):
        candidate = panel(.62, .91)
        for name in ('webrtc', 'ffmpeg'):
            candidate['families'][name]['macro_f1'] = .58
        self.assertIn('majority_panel_mechanisms_regressed', acceptance(fixed(), fixed(), {}, panel(), candidate)[1])

    def test_fixed_dev_regression_remains_guarded(self):
        candidate = fixed()
        candidate['groups']['online/zh']['recall'][1] -= .03
        self.assertFalse(acceptance(fixed(), candidate, {}, panel(), panel(.64, .93))[0])

    def test_panel_language_recall_collapse_is_not_hidden_by_mean_gain(self):
        candidate = panel(.64, .93)
        candidate['groups']['en']['recall'][1] -= 3/64
        self.assertIn('en_panel_real_recall_collapse', acceptance(fixed(), fixed(), {}, panel(), candidate)[1])
        candidate['groups']['en']['recall'][1] += 1/64
        self.assertTrue(acceptance(fixed(), fixed(), {}, panel(), candidate)[0])

    def test_panel_language_auc_collapse_is_guarded(self):
        candidate = panel(.64, .93)
        candidate['groups']['zh']['auc'] = .88
        self.assertIn('zh_panel_language_auc_collapse', acceptance(fixed(), fixed(), {}, panel(), candidate)[1])

    def test_tiny_language_groups_do_not_impose_quantized_recall_guard(self):
        base, candidate = panel(), panel(.64, .93)
        base['groups']['en']['class_counts'] = candidate['groups']['en']['class_counts'] = [12, 12]
        candidate['groups']['en']['recall'][1] -= 1/12
        self.assertTrue(acceptance(fixed(), fixed(), {}, base, candidate)[0])

    def test_independent_selectors_and_bounded_candidate_storage(self):
        selections = dict(best_weighted='starting_parent', best_guarded='starting_parent')
        scores = dict(best_weighted=.966, best_guarded=guarded_score(fixed(), panel()))
        candidate = fixed()
        candidate['weighted_f1'] = .9657
        candidates, promoted = update_selections(selections, scores, {}, 'candidate', candidate,
            True, {'weights': 1}, panel(.64, .92))
        self.assertEqual(promoted, ['best_guarded'])
        self.assertEqual(selections['best_weighted'], 'starting_parent')
        self.assertEqual(set(candidates), {'candidate'})

    def test_missing_or_nonfinite_panel_fails_closed(self):
        self.assertFalse(acceptance(fixed(), fixed(), {}, panel(), None)[0])
        candidate = panel(.62, .92)
        candidate['macro_auc'] = np.nan
        self.assertFalse(acceptance(fixed(), fixed(), {}, panel(), candidate)[0])


if __name__ == '__main__':
    unittest.main()
