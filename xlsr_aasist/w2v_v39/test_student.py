"""Centered real-only supervision and stable best-checkpoint selection."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import torch

from .model import CenteredStudent
from .student import train_student, qualified


def fixture():
    generator = torch.Generator().manual_seed(19)
    rows, x, g = [], [], []
    for language, sign in (('en', 1.), ('zh', -1.)):
        for label in (0, 1):
            for source in range(8):
                key = f'{language}/{label}/{source}'
                for condition in ('offline', 'online', 'noisy_a', 'noisy_b'):
                    rows.append(dict(source_id=key, group_id=key, language=language,
                        label=label, condition=condition, split='train'))
                    point = torch.randn(8, generator=generator) * .02
                    point[0], point[1] = 1. if label else -1., sign
                    x.append(point)
                    g.append(torch.tensor([sign * .4, 0., .7, 1.]))
    return torch.stack(x), torch.stack(g), rows


def config():
    return dict(seed=3901, student_hidden=8, student_epochs=12,
        student_learning_rate=.01, batch_rows=1024,
        min_student_r2=.02, min_student_language_accuracy=.65,
        min_teacher_language_accuracy=.7)


class StudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fake_features_and_targets_do_not_affect_fit_or_holdout(self):
        x, g, rows = fixture()
        damaged_x, damaged_g = x.clone(), g.clone()
        fake = torch.tensor([row['label'] == 0 for row in rows])
        damaged_x[fake], damaged_g[fake] = float('nan'), float('inf')
        with redirect_stdout(io.StringIO()):
            spec, quality = train_student(x, g, rows, config(), validation=(x, g, rows))
            same, same_quality = train_student(damaged_x, damaged_g, rows, config(),
                validation=(damaged_x, damaged_g, rows))
        self.assertEqual(quality, same_quality)
        for key in spec['state']:
            torch.testing.assert_close(spec['state'][key], same['state'][key], rtol=0, atol=0)
        self.assertTrue(qualified(quality, config())[0], quality)
        self.assertGreater(quality['best_epoch'], 0)

    def test_last_epoch_deterioration_restores_actual_best_for_full_fit_and_holdout(self):
        x, g, rows = fixture()
        cfg = dict(config(), student_epochs=3, student_learning_rate=.03)
        adamw = torch.optim.AdamW

        class LateCorruption(adamw):
            """Force a late optimizer failure after a genuine improving first step."""
            def __init__(self, params, **kwargs):
                super().__init__(params, **kwargs)
                self.step_count = 0

            def step(self, closure=None):
                result = super().step(closure)
                self.step_count += 1
                if self.step_count >= 2:
                    with torch.no_grad():
                        self.param_groups[0]['params'][-1].fill_(100.)
                return result

        for holdout in (False, True):
            with self.subTest(holdout=holdout), redirect_stdout(io.StringIO()):
                validation = (x, g, rows) if holdout else None
                expected, _ = train_student(x, g, rows, cfg, validation=validation, epochs=1)
                with patch('w2v_v39.student.torch.optim.AdamW', LateCorruption):
                    actual, quality = train_student(x, g, rows, cfg, validation=validation)
                self.assertEqual(quality['best_epoch'], 1)
                self.assertEqual(quality['selected_epoch'], 1)
                self.assertEqual(quality['epochs_completed'], 3)
                self.assertGreater(quality['final_centered_mse'], quality['best_centered_mse'] * 100.)
                self.assertAlmostEqual(quality['prediction_mse'], quality['best_centered_mse'], places=6)
                for key in actual['state']:
                    torch.testing.assert_close(actual['state'][key], expected['state'][key], rtol=0, atol=0)
                restored = CenteredStudent.restore(actual)
                self.assertTrue(all(not value.requires_grad for value in restored.parameters()))
                self.assertTrue(bool(torch.isfinite(restored(x)).all()))

    def test_short_refit_keeps_holdout_schedule_prefix(self):
        x, g, rows = fixture()
        cfg = config()
        with redirect_stdout(io.StringIO()):
            _, long = train_student(x, g, rows, cfg)
            _, short = train_student(x, g, rows, cfg, epochs=4)
        self.assertEqual(short['learning_rate_trace'], long['learning_rate_trace'][:4])
        self.assertEqual(short['schedule_horizon'], cfg['student_epochs'])
        self.assertEqual(short['learning_rate_trace'][0], cfg['student_learning_rate'])
        self.assertTrue(all(a > b for a, b in zip(long['learning_rate_trace'], long['learning_rate_trace'][1:])))

    def test_constant_teacher_and_dev_rows_are_rejected(self):
        x, g, rows = fixture()
        with redirect_stdout(io.StringIO()):
            spec, quality = train_student(x, torch.ones_like(g), rows, config())
        self.assertIsNone(spec)
        self.assertEqual(quality['status'], 'uninformative_teacher')
        with self.assertRaisesRegex(ValueError, 'Train rows only'):
            train_student(x, g, [dict(row, split='dev') for row in rows], config())


if __name__ == '__main__':
    unittest.main()
