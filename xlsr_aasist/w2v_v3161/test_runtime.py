"""Execution profiling must restore nonempty Adam/RNG and test real full views."""
from contextlib import redirect_stdout
import copy
import io
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_v3.test_model_step import tiny_detector
from w2v_v32.model import install_runtime
from w2v_v313.test_workflow import assert_nested_equal
from .objectives import TFCL
from .performance import select_execution
from .state import partial_state,capture_rng

class RuntimeTests(unittest.TestCase):
    def test_tuner_rolls_back_parameters_nonempty_moments_and_rng_after_oom(self):
        model = install_runtime(tiny_detector(checkpointing=False))
        model.configure_trainable_layers(1)
        aux = TFCL(8, 2, 21)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+list(aux.parameters()), lr=.01)
        # Prime actual Adam moments so rollback cannot pass by merely clearing them.
        initial_loss = sum(p.square().mean() for p in model.parameters() if p.requires_grad)
        initial_loss += sum(p.square().mean() for p in aux.parameters())
        initial_loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        original, original_aux = partial_state(model), copy.deepcopy(aux.state_dict())
        original_moments, rng = copy.deepcopy(optimizer.state_dict()), capture_rng()
        def simulated_step(model, auxiliary, optim, examples, cfg, warm):
            random.random(); np.random.rand(); torch.rand(3)
            optim.zero_grad(set_to_none=True)
            loss = sum(p.square().mean() for p in model.parameters() if p.requires_grad)
            loss += sum(p.square().mean() for p in auxiliary.parameters())
            loss.backward(); optim.step()
            if cfg['microbatch']==18 and not cfg['checkpointing']:
                raise torch.cuda.OutOfMemoryError('fixture peak')
        cfg = dict(device='cuda:0', autotune=True, microbatch=18, frame_budget=10800,
                   source_batch=16, gpu_reserve_bytes=6*1024**3)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch('w2v_v3161.performance.probe_examples', return_value=[]), \
             patch('w2v_v3161.performance.train_step', side_effect=simulated_step), \
             patch('torch.cuda.get_device_name', return_value='mock GPU'), \
             patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.empty_cache'), \
             patch('torch.cuda.mem_get_info', return_value=(30*1024**3, 40*1024**3)), \
             patch('torch.cuda.memory_reserved', return_value=8*1024**3), \
             patch('torch.cuda.max_memory_reserved', return_value=12*1024**3), \
             patch('torch.cuda.max_memory_allocated', return_value=10*1024**3):
            result = select_execution(model, aux, optimizer, None, cfg, directory)
            self.assertTrue(result['rollback_verified'])
            self.assertTrue(any(value['status']=='out_of_memory' for value in result['attempts']))
            assert_nested_equal(self, original_moments, optimizer.state_dict())
            assert_nested_equal(self, original, partial_state(model))
            assert_nested_equal(self, original_aux, aux.state_dict())
            assert_nested_equal(self, rng, capture_rng())

    def test_profile_tickets_keep_pair_identity_recipe_and_longest_online(self):
        import json
        from w2v_v316_tfcl.data import SourcePlan
        from w2v_v316_tfcl.test_data import rows_fixture
        from .performance import probe_examples
        from w2v_v315.augment import recipe
        rows = rows_fixture(9)
        for index, row in enumerate(rows):
            row['output_samples'] = 16000+index
            if row['condition']=='online' and row['id'].endswith('_0.wav'):
                row['output_samples'] = 1000000
        plan = SourcePlan(rows, 16, 11)
        seen = []
        class CaptureDataset:
            def __getitem__(self, ticket):
                seen.append(ticket)
                return [dict(features=np.zeros((1, 4, 160), np.float32), mask=np.ones((1, 4), np.int64))
                        for _ in range(3)]
        class Collator:
            def __call__(self, groups):
                return [row for group in groups for row in group]
        with patch('w2v_v3161.performance.Triplets', return_value=CaptureDataset()), \
             patch('w2v_v3161.performance.TripletCollator', return_value=Collator()):
            result = probe_examples(plan, dict(seed=11, ssl_path='unused',sampling_epoch_offset=4,
                source_committed_updates=4*plan.steps), '.')
        self.assertEqual(len(result), 48)
        self.assertTrue(all(isinstance(row['features'], torch.Tensor) for row in result))
        for ticket in seen:
            if ticket.online is not None:self.assertEqual(rows[ticket.original]['source_id'], rows[ticket.online]['source_id'])
            self.assertTrue(ticket.occurrence.startswith('profile:'))
            self.assertEqual(json.loads(ticket.recipe_json), recipe(11, ticket.occurrence, ticket.phase,
                rows[ticket.original]['group_id'], warm=ticket.warm))
        self.assertTrue(all(rows[ticket.online]['output_samples']==1000000 for ticket in seen[-4:]))


if __name__=="__main__":unittest.main()
