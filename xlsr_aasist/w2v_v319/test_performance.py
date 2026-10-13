"""CPU coverage of rollback, physical budgets, selection and resume invariants."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from w2v_v39.common import atomic_json
from .performance import (ProbeSnapshot, execution_candidates, memory_budget,
                          select_execution, _winner, GIB)


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.encoder = nn.Module()
        self.backbone.encoder.gradient_checkpointing = False
        self.head = nn.Module()
        self.head.classifier = nn.Linear(3, 2)
        self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
        self.register_buffer('running', torch.zeros(3))
        self.register_buffer('temporary', torch.ones(2), persistent=False)

    def forward(self, x):
        self.running.add_(1.)
        self.temporary.mul_(.9)
        return self.head.classifier(x + self.frozen)


def config(**changes):
    result = dict(version='3.19', device='cpu', autotune=True,
                  source_batch=16, microbatch=48, frame_budget=24000,
                  checkpointing=False, gpu_reserve_bytes=4 * GIB)
    result.update(changes)
    return result


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_six_physical_candidates_preserve_logical_objective(self):
        cfg = config()
        before = copy.deepcopy(cfg)
        choices = execution_candidates(cfg)
        self.assertEqual(len(choices), 6)
        self.assertEqual({(c['microbatch'], c['frame_budget']) for c in choices},
                         {(12, 7200), (24, 14400), (48, 24000)})
        self.assertEqual(cfg, before)
        self.assertTrue(all(set(c) == {'microbatch', 'frame_budget', 'checkpointing'} for c in choices))
        small = execution_candidates(config(source_batch=4, microbatch=48, frame_budget=4000))
        self.assertEqual(len(small), 2)
        self.assertTrue(all(c['microbatch'] == 12 and c['frame_budget'] == 4000 for c in small))
        for key, value in (('microbatch', 2), ('microbatch', True), ('frame_budget', 0), ('source_batch', 0)):
            with self.assertRaises(ValueError):
                execution_candidates(config(**{key: value}))

    def test_memory_reserve_and_fastest_not_largest(self):
        available, reserve = memory_budget(20 * GIB, 20 * GIB, 40 * GIB, 4 * GIB)
        self.assertEqual((available, reserve), (36 * GIB, 4 * GIB))
        available, reserve = memory_budget(10 * GIB, 20 * GIB, 40 * GIB, 4 * GIB)
        self.assertEqual(available, 26 * GIB)
        self.assertEqual(memory_budget(80 * GIB, 0, 80 * GIB, 4 * GIB)[1], 8 * GIB)
        attempts = [dict(microbatch=48, status='ok', eligible=True, seconds_per_update=2.1),
                    dict(microbatch=24, status='ok', eligible=True, seconds_per_update=1.5),
                    dict(microbatch=48, status='ok', eligible=False, seconds_per_update=.9),
                    dict(microbatch=12, status='out_of_memory', eligible=False)]
        self.assertEqual(_winner(attempts)['microbatch'], 24)
        with self.assertRaises(RuntimeError):
            _winner([dict(status='ok', eligible=False, seconds_per_update=1.)])

    def test_real_adam_and_all_state_restore_after_repeated_trials(self):
        torch.manual_seed(319)
        np.random.seed(319)
        random.seed(319)
        model, auxiliary = TinyDetector(), nn.Linear(2, 2)
        model.head.eval()
        optimizer = torch.optim.AdamW(list(model.head.parameters()) + list(auxiliary.parameters()), lr=.01)

        def update():
            optimizer.zero_grad(set_to_none=True)
            value = auxiliary(model(torch.randn(5, 3)))
            value.square().mean().backward()
            optimizer.step()
            random.random()
            np.random.random()

        update()  # Exercise nonempty Adam moments and CPU step tensors.
        snapshot = ProbeSnapshot(model, auxiliary, optimizer)
        parameter_ids = [id(p) for p in model.parameters()]
        for _ in range(3):
            update()
            with torch.no_grad():
                model.frozen.add_(10.)
            model.train()
            model.backbone.encoder.gradient_checkpointing = True
            snapshot.restore()
            snapshot.assert_restored()
        self.assertEqual(parameter_ids, [id(p) for p in model.parameters()])

    def test_empty_adam_state_is_restored_after_failure(self):
        model, auxiliary = TinyDetector(), nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(list(model.head.parameters()) + list(auxiliary.parameters()), lr=.01)
        snapshot = ProbeSnapshot(model, auxiliary, optimizer)
        try:
            auxiliary(model(torch.randn(3, 3))).sum().backward()
            optimizer.step()
            raise RuntimeError('simulated profiling failure')
        except RuntimeError:
            snapshot.restore()
        snapshot.assert_restored()
        self.assertEqual(optimizer.state_dict()['state'], {})

    def test_cpu_plan_reuses_and_rejects_changed_configuration_or_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config()
            plan = select_execution(None, None, None, None, cfg, directory)
            path = Path(directory) / 'execution_plan.json'
            original = path.read_bytes()
            self.assertEqual(plan['attempts'], [])
            self.assertEqual(plan['selected']['microbatch'], 48)
            self.assertEqual(select_execution(None, None, None, None, cfg, directory), plan)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                select_execution(None, None, None, None, config(frame_budget=12000), directory)
            mutated = dict(plan, device_name='another_gpu')
            atomic_json(path, mutated)
            with self.assertRaisesRegex(ValueError, 'hardware'):
                select_execution(None, None, None, None, cfg, directory)
            mutated = dict(plan, logical_sources=32)
            atomic_json(path, mutated)
            with self.assertRaisesRegex(ValueError, 'logical objective'):
                select_execution(None, None, None, None, cfg, directory)


if __name__ == '__main__':
    unittest.main()
