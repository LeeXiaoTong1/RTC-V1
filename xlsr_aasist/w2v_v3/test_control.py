import unittest
from .control import Controller, lr_scale


def dev(weighted=.95, noisy=.94, real=.86, fake=.97):
    return {'weighted_f1': weighted, 'noisy_f1': noisy, 'clean_f1': .97,
            'groups': {k: {'recall': [fake, real]} for k in
                       ('offline/en', 'online/en', 'seen/en', 'heldout/en')}}


def config():
    return dict(head_epochs=2, joint_epochs=4, evals_per_epoch=2,
                plateau_evals=2, reduced_lr_evals=2, max_lr_reductions=2,
                en_real_tolerance=.02, noisy_fake_tolerance=.005)


def joint(initial=None):
    c = Controller(config())
    c.observe(initial or dev(), 'a')
    c.begin_joint()
    return c


class ControllerTests(unittest.TestCase):
    def test_noisy_winner_can_survive_recall_guard_without_being_promoted(self):
        c = joint()
        r = c.observe(dev(.951, .948, .81, .98), 'b')
        self.assertIn('best_noisy', r['save'])
        self.assertIn('best_weighted', r['save'])
        self.assertNotIn('best_safe', r['save'])
        self.assertEqual(c.state['best_safe']['tag'], 'a')
        self.assertEqual(r['action'], 'restore_reduce')

    def test_metric_tradeoff_is_allowed_and_no_requirement_all_metrics_improve(self):
        c = joint()
        r = c.observe(dev(.952, .939, .85, .971), 'b')
        self.assertIn('best_safe', r['save'])
        self.assertNotIn('best_noisy', r['save'])

    def test_recall_floor_does_not_ratchet_down_with_each_candidate(self):
        c = joint(dev(real=.88))
        c.observe(dev(.951, real=.865), 'b')
        r = c.observe(dev(.952, real=.85), 'c')
        self.assertNotIn('best_safe', r['save'])
        self.assertEqual(c.state['anchor']['en_real'], .88)

    def test_reduction_gets_two_new_evaluation_intervals(self):
        c = joint()
        r = c.observe(dev(real=.82), 'b')
        self.assertEqual(r['action'], 'restore_reduce')
        self.assertEqual(c.state['lr_scale'], .5)
        self.assertEqual(c.observe(dev(real=.82), 'c')['action'], 'continue')
        self.assertEqual(c.observe(dev(real=.82), 'd')['action'], 'restore_reduce')
        self.assertEqual(c.state['reductions'], 2)
        self.assertEqual(c.observe(dev(real=.82), 'e')['action'], 'continue')
        self.assertEqual(c.observe(dev(real=.82), 'f')['action'], 'phase_complete')
        self.assertEqual(c.state['since_reduction'], 2)

    def test_no_pointless_last_interval_lr_reduction(self):
        c = Controller(config()); c.observe(dev(), 'a'); c.observe(dev(), 'b')
        self.assertEqual(c.observe(dev(), 'c')['action'], 'continue')
        self.assertEqual(c.state['reductions'], 0)
        self.assertEqual(c.observe(dev(), 'd')['action'], 'phase_complete')

    def test_checkpoint_restores_controller_exactly(self):
        c = joint(); c.observe(dev(real=.8), 'b')
        b = Controller(config(), c.dump())
        self.assertEqual(c.observe(dev(.952), 'c'), b.observe(dev(.952), 'c'))
        self.assertEqual(c.dump(), b.dump())
        self.assertEqual(c.state['phase'], 'joint')
        self.assertEqual(c.state['anchor']['tag'], 'a')
        self.assertEqual(c.phase_budget(), 8)

    def test_guard_starts_from_best_trained_head_not_random_early_recall(self):
        c = Controller(config())
        c.observe(dev(.8, real=.4, fake=.999), 'half')
        self.assertIn('best_safe', c.observe(dev(.95, real=.88, fake=.97), 'full')['save'])
        c.begin_joint()
        self.assertAlmostEqual(c.state['anchor']['en_real'], .88)
        self.assertAlmostEqual(c.state['anchor']['noisy_en_fake'], .97)

    def test_bad_metrics_rejected(self):
        with self.assertRaises(ValueError): Controller(config()).observe(dev(float('nan')), 'bad')

    def test_lr_schedule_reductions_are_not_overwritten(self):
        cfg = dict(lr_warmup_steps=10, min_lr_scale=.1)
        self.assertAlmostEqual(lr_scale(cfg, 20, 100, .5), .5*lr_scale(cfg, 20, 100))
        self.assertAlmostEqual(lr_scale(cfg, 99, 100), .1)


if __name__ == '__main__':
    unittest.main()
