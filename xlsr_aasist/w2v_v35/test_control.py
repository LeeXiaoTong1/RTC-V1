import unittest
from .control import Controller


def dev(clean, noisy):
    return dict(clean_f1=clean, noisy_f1=noisy, weighted_f1=.3 * clean + .7 * noisy,
                groups={'offline/en': {'recall': [1., 0.]}})


class ControllerTests(unittest.TestCase):
    def test_offline_recall_never_vetoes_online_promotion_and_tracks_cold_start(self):
        controller = Controller(dict(joint_epochs=5, patience=2))
        controller.initialize(dev(.98, .94))
        first = controller.observe(dev(.9, .8), 'epoch_1', 'head')
        self.assertEqual(first['save'], ['best_train'])
        self.assertEqual(controller.state['best_selected']['tag'], 'reference')
        second = controller.observe(dev(.97, .96), 'epoch_2', 'joint')
        self.assertTrue(second['promoted'])
        self.assertEqual(controller.state['best_selected']['tag'], 'epoch_2')

    def test_patience_counts_only_complete_joint_epochs_and_resume_identical(self):
        controller = Controller(dict(joint_epochs=5, patience=2))
        controller.initialize(dev(.98, .94))
        controller.observe(dev(.96, .9), 'epoch_1', 'head')
        controller.observe(dev(.97, .91), 'epoch_2', 'joint')
        self.assertFalse(controller.state['completed'])
        controller.observe(dev(.96, .9), 'epoch_3', 'joint')
        resumed = Controller(controller.cfg, controller.dump())
        decision = controller.observe(dev(.96, .9), 'epoch_4', 'joint')
        self.assertEqual(decision, resumed.observe(dev(.96, .9), 'epoch_4', 'joint'))
        self.assertEqual(decision['action'], 'phase_complete')
        with self.assertRaises(ValueError):
            resumed.observe(dev(.9, .9), 'interim', 'joint', full_epoch=False)


if __name__ == '__main__':
    unittest.main()
