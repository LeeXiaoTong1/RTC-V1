import copy
import unittest
from .control import Controller, lr_scale, quality, acceptance, pair_weight_for_exposure


def dev(weighted=.95, noisy=.94, clean=.97, real=.86, fake=.98):
    return {'weighted_f1': weighted, 'noisy_f1': noisy, 'clean_f1': clean,
            'groups': {k: {'recall': [fake, real]} for k in
                       ('offline/en', 'online/en', 'seen/en', 'heldout/en')}}


def ready(cfg=None):
    c = Controller({'joint_epochs': 2, 'clean_tolerance': .001, **(cfg or {})})
    c.initialize(dev())
    return c


class ControllerTests(unittest.TestCase):
    def test_baseline_initialization_consumes_no_training_or_evaluation_budget(self):
        c = Controller({'joint_epochs': 2})
        result = c.initialize(dev())
        self.assertEqual(c.state['phase'], 'joint')
        self.assertEqual(c.state['phase_steps'], 0)
        self.assertEqual(c.state['phase_evals'], 0)
        self.assertEqual(c.state['evaluations'], 0)
        self.assertEqual(result['remaining_evaluations'], 4)
        self.assertEqual(result['restore_tag'], 'baseline')
        self.assertEqual(set(result['save']), {'best_weighted', 'best_noisy', 'best_safe'})
        with self.assertRaises(RuntimeError):
            c.initialize(dev(), 'replacement')

    def test_requires_baseline_before_updates(self):
        with self.assertRaises(RuntimeError):
            Controller({}).observe(dev(), 'first')

    def test_clean_improvements_cannot_mask_noisy_english_real_decline(self):
        c = ready()
        d = dev(.953, .942)
        for key, value in zip(d['groups'], (.96, .94, .77, .77)):
            d['groups'][key]['recall'][1] = value
        result = c.observe(d, 'bad_tradeoff')
        self.assertAlmostEqual(quality(d)['en_real'], .86)
        self.assertNotIn('best_safe', result['save'])
        self.assertIn('best_weighted', result['save'])
        self.assertIn('best_noisy', result['save'])
        self.assertIn('seen_en_real_below_baseline', result['warnings'])
        self.assertIn('heldout_en_real_below_baseline', result['warnings'])
        self.assertEqual(result['restore_tag'], 'baseline')

    def test_seen_and_heldout_recall_are_protected_independently(self):
        for condition in ('offline', 'online', 'seen', 'heldout'):
            with self.subTest(condition=condition):
                c = ready()
                d = dev(.951, .941, real=.90)
                d['groups'][condition + '/en']['recall'][1] = .854
                result = c.observe(d, 'candidate')
                self.assertIn(condition + '_en_real_below_baseline', result['warnings'])
                self.assertNotIn('best_safe', result['save'])
        for condition in ('seen', 'heldout'):
            with self.subTest(fake_condition=condition):
                c = ready()
                d = dev(.951, .941, fake=.99)
                d['groups'][condition + '/en']['recall'][0] = .976
                result = c.observe(d, 'candidate')
                self.assertIn(condition + '_en_fake_below_baseline', result['warnings'])
                self.assertNotIn('best_safe', result['save'])

    def test_noisy_f1_cannot_decline_even_if_weighted_improves(self):
        c = ready()
        result = c.observe(dev(.951, .939999), 'clean_only')
        self.assertIn('noisy_f1_below_baseline', result['warnings'])
        self.assertNotIn('best_safe', result['save'])
        self.assertIn('best_weighted', result['save'])

    def test_clean_f1_has_its_own_small_tolerance(self):
        c = ready()
        result = c.observe(dev(.952, .945, clean=.9689), 'clean_loss')
        self.assertIn('clean_f1_below_baseline', result['warnings'])
        self.assertNotIn('best_safe', result['save'])

    def test_exact_tolerance_boundary_and_numerical_equality_are_allowed(self):
        c = ready()
        result = c.observe(dev(.951, .94 - 1e-14, clean=.969, real=.855, fake=.977), 'allowed')
        self.assertEqual(result['warnings'], [])
        self.assertIn('best_safe', result['save'])

    def test_fixed_anchor_prevents_cumulative_recall_drift(self):
        c = ready()
        baseline = copy.deepcopy(c.state['anchor'])
        self.assertIn('best_safe', c.observe(dev(.951, .941, real=.856), 'a')['save'])
        result = c.observe(dev(.952, .942, real=.852), 'b')
        self.assertNotIn('best_safe', result['save'])
        self.assertEqual(c.state['best_safe']['tag'], 'a')
        self.assertEqual(c.state['anchor'], baseline)

    def test_anchor_is_not_tightened_by_a_temporary_recall_peak(self):
        c = ready()
        self.assertIn('best_safe', c.observe(dev(.951, .941, real=.90), 'a')['save'])
        self.assertIn('best_safe', c.observe(dev(.952, .942, real=.86), 'b')['save'])
        self.assertEqual(c.state['anchor']['seen_en_real'], .86)

    def test_worse_or_tiny_improvement_cannot_overwrite_protected_fallback(self):
        c = ready()
        self.assertNotIn('best_safe', c.observe(dev(.949, .939), 'worse')['save'])
        self.assertNotIn('best_safe', c.observe(dev(.95005, .941), 'tiny')['save'])
        self.assertEqual(c.state['best_safe']['tag'], 'baseline')
        self.assertEqual(c.state['best_noisy']['tag'], 'tiny')

    def test_plateau_reduction_preserves_two_complete_lower_lr_trials(self):
        c = ready()
        self.assertEqual(c.observe(dev(), 'a')['action'], 'continue')
        self.assertEqual(c.observe(dev(), 'b')['action'], 'restore_reduce')
        self.assertEqual(c.state['lr_scale'], .5)
        self.assertEqual(c.observe(dev(), 'c')['action'], 'continue')
        self.assertEqual(c.observe(dev(), 'd')['action'], 'phase_complete')
        self.assertEqual(c.state['since_reduction'], 2)
        self.assertEqual(c.state['reductions'], 1)
        with self.assertRaises(RuntimeError):
            c.observe(dev(), 'over_budget')

    def test_severe_early_drift_allows_only_one_recovery(self):
        c = ready()
        result = c.observe(dev(.94, .93, real=.80), 'a')
        self.assertEqual(result['action'], 'restore_reduce')
        self.assertEqual(result['restore_tag'], 'baseline')
        self.assertEqual(c.observe(dev(real=.80), 'b')['action'], 'continue')
        self.assertEqual(c.observe(dev(real=.80), 'c')['action'], 'continue')
        self.assertEqual(c.observe(dev(real=.80), 'd')['action'], 'phase_complete')
        self.assertEqual(c.state['reductions'], 1)
        self.assertEqual(c.state['since_reduction'], 3)

    def test_no_reduction_without_time_for_two_more_evaluations(self):
        c = ready({'joint_epochs': 1})
        self.assertEqual(c.observe(dev(real=.80), 'a')['action'], 'continue')
        self.assertEqual(c.observe(dev(real=.80), 'b')['action'], 'phase_complete')
        self.assertEqual(c.state['reductions'], 0)

    def test_resume_restores_controller_exactly(self):
        c = ready()
        c.observe(dev(), 'a')
        c.observe(dev(), 'b')
        restored = Controller(c.cfg, c.dump())
        self.assertEqual(c.observe(dev(.951, .941), 'c'), restored.observe(dev(.951, .941), 'c'))
        self.assertEqual(c.dump(), restored.dump())
        exported = c.dump()
        exported['anchor']['en_real'] = 0
        self.assertNotEqual(c.state['anchor']['en_real'], 0)

    def test_invalid_metrics_and_legacy_state_are_rejected(self):
        with self.assertRaises(ValueError):
            Controller({}).initialize(dev(float('nan')))
        with self.assertRaises(ValueError):
            Controller({}, {'phase': 'joint'})
        with self.assertRaises(ValueError):
            Controller({'joint_epochs': 3})
        with self.assertRaises(ValueError):
            Controller({'metric_epsilon': .001})

    def test_report_quality_exposes_each_guarded_condition(self):
        q = quality(dev())
        for condition in ('offline', 'online', 'seen', 'heldout'):
            self.assertEqual(q[condition + '_en_real'], .86)
        self.assertEqual(q['noisy_en_real'], .86)
        self.assertEqual(q['noisy_en_fake'], .98)

    def test_recovery_scaling_survives_lr_scheduler(self):
        cfg = dict(lr_warmup_steps=10, min_lr_scale=.1)
        self.assertAlmostEqual(lr_scale(cfg, 20, 100, .5), .5 * lr_scale(cfg, 20, 100))
        self.assertAlmostEqual(lr_scale(cfg, 99, 100), .1)


class SourceControlTests(unittest.TestCase):
    def test_default_is_one_epoch_with_two_validations_and_fixed_clean_floor(self):
        controller = Controller({})
        controller.initialize(dev())
        self.assertEqual(controller.phase_budget(), 2)
        result = controller.observe(dev(.951, .941, clean=.968), 'small_safe_gain')
        self.assertIn('best_safe', result['save'])
        self.assertEqual(controller.state['anchor']['clean'], .97)
        self.assertFalse(acceptance(controller.state['best_safe'], controller.state['anchor'])['success'])

    def test_success_requires_both_predeclared_targets_not_just_micro_gain(self):
        controller = ready()
        controller.observe(dev(.953, .943, real=.88), 'large_gain')
        verdict = acceptance(controller.state['best_safe'], controller.state['anchor'])
        self.assertTrue(verdict['success'])
        selected = dict(controller.state['best_safe'], noisy=.941)
        self.assertIn('noisy_macro_f1_gain_below_target', acceptance(selected, controller.state['anchor'])['unmet'])
        selected = dict(controller.state['best_safe'], noisy_en_real=.879)
        self.assertIn('noisy_en_real_recall_gain_below_target', acceptance(selected, controller.state['anchor'])['unmet'])
        self.assertFalse(acceptance(controller.state['anchor'], controller.state['anchor'])['success'])

    def test_pair_ramp_counts_source_exposures_and_control_never_uses_pair_loss(self):
        cfg = {'arm': 'candidate', 'pair_weight': .02, 'pair_ramp_fraction': .1}
        self.assertEqual(pair_weight_for_exposure(cfg, 0, 1000), 0)
        self.assertAlmostEqual(pair_weight_for_exposure(cfg, 50, 1000), .01)
        self.assertAlmostEqual(pair_weight_for_exposure(cfg, 100, 1000), .02)
        self.assertAlmostEqual(pair_weight_for_exposure(cfg, 2000, 1000), .02)
        self.assertEqual(pair_weight_for_exposure({**cfg, 'arm': 'control'}, 100, 1000), 0)
        with self.assertRaises(ValueError):
            pair_weight_for_exposure(cfg, 10, 0)

    def test_standalone_success_checks_all_fixed_anchor_floors(self):
        baseline = {'tag': 'baseline', **quality(dev())}
        candidate = {'tag': 'candidate', **quality(dev(.953, .943, real=.88))}
        self.assertTrue(acceptance(candidate, baseline)['success'])
        for key, value in (('weighted', .95), ('clean', .967),
                           ('offline_en_real', .854), ('online_en_real', .854),
                           ('seen_en_real', .854), ('heldout_en_real', .854),
                           ('seen_en_fake', .976), ('heldout_en_fake', .976)):
            with self.subTest(metric=key):
                result = acceptance(dict(candidate, **{key: value}), baseline)
                self.assertFalse(result['success'])
                self.assertTrue(result['unmet'])
        with self.assertRaises(ValueError):
            acceptance(dict(candidate, clean=float('nan')), baseline)

    def test_full_scale_lr_warmup_is_one_hundred_updates(self):
        cfg = {'lr_warmup_steps': 100}
        self.assertAlmostEqual(lr_scale(cfg, 0, 10000), .109)
        self.assertAlmostEqual(lr_scale(cfg, 99, 10000), 1.)
        self.assertAlmostEqual(lr_scale(cfg, 100, 10000), 1.)


if __name__ == '__main__':
    unittest.main()
