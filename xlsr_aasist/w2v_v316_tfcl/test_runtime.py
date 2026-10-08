"""Pinned parent, transactional resume and deployment selection contracts."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from w2v_v3.test_model_step import tiny_detector
from w2v_v32.model import install_runtime
from w2v_v313.test_workflow import assert_nested_equal
from w2v_v315.state import SCHEMA as PARENT_SCHEMA
from w2v_v39.common import atomic_json, read_json, digest
from .config import source_configuration, verify_parent
from .model import load_model
from .objectives import TFCL
from .performance import select_execution
from .state import identity, partial_state, capture_rng, SCHEMA, load_selected, load_resume, atomic_save


def weights(value):
    return dict(weight=torch.tensor([[value]*4, [-value]*4], dtype=torch.float32), bias=torch.zeros(2))


def parent_fixture(directory):
    root = Path(directory)
    source = root/'protected_v312_last.pt'
    torch.save({'source': torch.ones(4)}, source)
    run = root/'parent_v315'
    run.mkdir()
    cfg = dict(version='3.15', source_fingerprints={}, data_fingerprints={}, code_fingerprints={},
        base_checkpoint=str(source), base_checkpoint_sha256=digest(source),
        starting_checkpoint=str(source), starting_checkpoint_sha256=digest(source), starting_tag='v312_last',
        disk_margin_bytes=0, rolling_cache_bytes=0)
    atomic_json(run/'execution_plan.json', {'parent_profile': True})
    sha = digest(run/'execution_plan.json')
    selection = dict(best_guarded='parent_guarded', best_weighted='parent_weighted')
    state = dict(schema=PARENT_SCHEMA, identity=identity(cfg), cursor=3, last_tag='parent_last', selections=selection,
        model=weights(.3), candidates={tag: dict(kind='partial', state=weights(value))
            for tag, value in (('parent_guarded', .1), ('parent_weighted', .2))},
        history=[dict(tag='parent_last', cursor=3, committed=True)], execution_plan_sha256=sha)
    atomic_save(run/'last.pt', state, 0)
    atomic_json(run/'config.json', cfg)
    done = dict(version='3.15', status='complete', state_file='last.pt', checkpoint_sha256=digest(run/'last.pt'),
        selections=selection, last_tag='parent_last', committed_updates=3, execution_plan_sha256=sha,
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'], starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'])
    atomic_json(run/'completed.json', done)
    return run, cfg


def completed_fixture(directory):
    parent, _ = parent_fixture(directory)
    cfg = source_configuration(parent, 'best_guarded')
    cfg.update(version='3.16',variant='offline_reference_tfcl_v1',init_mode='parent',pretrained_fingerprints={})
    run = Path(directory)/'v3151'
    run.mkdir()
    atomic_json(run/'execution_plan.json', {'child_profile': True})
    sha = digest(run/'execution_plan.json')
    selection = dict(best_weighted='child_weighted', best_guarded='child_guarded')
    state = dict(schema=SCHEMA, identity=identity(cfg), cursor=2, last_tag='child_last', selections=selection,
        model=weights(.5), candidates={tag: dict(kind='partial', state=weights(value))
            for tag, value in (('child_weighted', .4), ('child_guarded', .35))},
        history=[dict(tag='child_last', cursor=2, committed=True)], execution_plan_sha256=sha,
        optimizer={'large_state': torch.ones(20000)}, rng=capture_rng(), auxiliary={'discard': torch.ones(200)},
        parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'], parent_selected_tag=cfg['parent_selected_tag'])
    atomic_save(run/'last.pt', state, 0)
    atomic_json(run/'config.json', cfg)
    done = dict(version='3.16',variant='offline_reference_tfcl_v1',init_mode='parent',pretrained_fingerprints={}, status='complete', state_file='last.pt', checkpoint_sha256=digest(run/'last.pt'),
        selections=selection, last_tag='child_last', committed_updates=2, execution_plan_sha256=sha,
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'], starting_checkpoint_sha256=cfg['starting_checkpoint_sha256'],
        parent_run=cfg['parent_run'], parent_selector=cfg['parent_selector'], parent_selected_tag=cfg['parent_selected_tag'],
        parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'])
    atomic_json(run/'completed.json', done)
    for kind in ('best_guarded', 'best_weighted', 'last'):
        atomic_json(run/(kind+'.json'), {'checkpoint': 'last.pt'})
    return run, cfg


def rewrite_state(run, state):
    atomic_save(run/'last.pt', state, 0)
    done = read_json(run/'completed.json')
    done.update(checkpoint_sha256=digest(run/'last.pt'), selections=state['selections'])
    atomic_json(run/'completed.json', done)


class RuntimeTests(unittest.TestCase):

    def test_mutated_parent_weights_and_parent_completion_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, _ = parent_fixture(directory)
            cfg = source_configuration(parent)
            done = read_json(parent/'completed.json')
            done['checkpoint_sha256'] = '0'*64
            atomic_json(parent/'completed.json', done)
            with self.assertRaisesRegex(ValueError, 'parent changed'):
                verify_parent(cfg)
            done['checkpoint_sha256'] = digest(parent/'last.pt')
            atomic_json(parent/'completed.json', done)
            with (parent/'last.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                verify_parent(cfg)

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
             patch('w2v_v316_tfcl.performance.probe_examples', return_value=[]), \
             patch('w2v_v316_tfcl.performance.train_step', side_effect=simulated_step), \
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

    def test_checkpoint_disk_failure_preserves_previous_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            atomic_save(path, {'x': torch.ones(20)}, 0)
            before = digest(path)
            with patch('w2v_v313.state.shutil.disk_usage', return_value=type('Disk', (), {'free': 0})()):
                with self.assertRaises(OSError):
                    atomic_save(path, {'x': torch.zeros(20)}, 0)
            self.assertEqual(digest(path), before)

    def test_resume_requires_same_config_execution_and_full_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run, cfg = completed_fixture(directory)
            load_resume(run/'last.pt', cfg)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                load_resume(run/'last.pt', dict(cfg, parent_selected_tag='wrong'))
            original_execution = read_json(run/'execution_plan.json')
            atomic_json(run/'execution_plan.json', {'changed': True})
            with self.assertRaisesRegex(ValueError, 'execution plan differs'):
                load_resume(run/'last.pt', cfg)
            atomic_json(run/'execution_plan.json', original_execution)
            state = torch.load(run/'last.pt', weights_only=True)
            state['inference_only'] = True
            atomic_save(run/'last.pt', state, 0)
            with self.assertRaisesRegex(ValueError, 'inference only'):
                load_resume(run/'last.pt', cfg)

    def test_distinct_export_candidates_and_missing_candidate_never_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _ = completed_fixture(directory)
            for selector, expected in (('best_guarded', .35), ('best_weighted', .4), ('last', .5)):
                candidate, meta = load_selected(run, selector)
                self.assertFalse(meta['baseline_fallback'])
                assert_nested_equal(self, weights(expected), candidate['candidate']['state'])
            state = torch.load(run/'last.pt', weights_only=True)
            state['candidates'].pop('child_guarded')
            rewrite_state(run, state)
            with self.assertRaisesRegex(ValueError, 'refusing fallback'):
                load_selected(run, 'best_guarded')


    def test_compaction_interruption_preserves_parent_and_all_export_choices(self):
        from .cleanup import cleanup
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run, cfg = completed_fixture(directory)
            protected = digest(cfg['parent_checkpoint'])
            real = atomic_json
            def interrupted(path, value):
                if Path(path).name=='completed.json':
                    raise InterruptedError('manifest switch')
                return real(path, value)
            with patch('w2v_v316_tfcl.cleanup.atomic_json', side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    cleanup(run, apply=True)
            load_selected(run, 'last')
            self.assertTrue((run/'last.pt').exists())
            cleanup(run, apply=True)
            self.assertFalse((run/'last.pt').exists())
            for selector in ('best_guarded', 'best_weighted', 'last'):
                load_selected(run, selector)
            self.assertEqual(protected, digest(cfg['parent_checkpoint']))

    def test_completed_history_and_execution_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, cfg = completed_fixture(directory)
            state = torch.load(run/'last.pt', weights_only=True)
            state['history'][-1]['committed'] = False
            rewrite_state(run, state)
            with self.assertRaisesRegex(ValueError, 'committed validation'):
                load_selected(run, 'last')
            state['history'][-1]['committed'] = True
            rewrite_state(run, state)
            atomic_json(run/'execution_plan.json', {'changed': True})
            with self.assertRaisesRegex(ValueError, 'execution plan differs'):
                load_selected(run)


    def test_profile_tickets_keep_pair_identity_recipe_and_longest_online(self):
        import json
        from .data import SourcePlan
        from .test_data import rows_fixture
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
        with patch('w2v_v316_tfcl.performance.Triplets', return_value=CaptureDataset()), \
             patch('w2v_v316_tfcl.performance.TripletCollator', return_value=Collator()):
            result = probe_examples(plan, dict(seed=11, ssl_path='unused'), '.')
        self.assertEqual(len(result), 48)
        self.assertTrue(all(isinstance(row['features'], torch.Tensor) for row in result))
        for ticket in seen:
            if ticket.online is not None:self.assertEqual(rows[ticket.original]['source_id'], rows[ticket.online]['source_id'])
            self.assertTrue(ticket.occurrence.startswith('profile:'))
            self.assertEqual(json.loads(ticket.recipe_json), recipe(11, ticket.occurrence, ticket.phase,
                rows[ticket.original]['group_id'], warm=ticket.warm))
        self.assertTrue(all(rows[ticket.online]['output_samples']==1000000 for ticket in seen[-4:]))


if __name__=='__main__':
    unittest.main()
