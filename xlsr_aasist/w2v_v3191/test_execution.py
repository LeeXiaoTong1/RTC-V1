"""Decision/update equivalence, omitted work and strict saved-run compatibility."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import step
from .execution import (EXECUTION_POLICY, MANIFEST, diagnostic_step,
                        verify_execution_code, record_execution)
from .meta import rng_state, restore_rng
from .test_step import Detector, fixture, optimizer, assert_rng_equal
from w2v_v31810.test_meta import assert_state_equal


class EfficientScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fast_and_full_choose_identical_sources_with_quality_fallbacks(self):
        for seed in range(6):
            torch.manual_seed(seed)
            cfg, units = fixture()
            rng = np.random.default_rng(seed)
            for unit in units:
                for candidate in unit[0].get('robust_candidates', [])[1:]:
                    candidate[0]['robust_quality']['passed'] = bool(rng.random() > .35)
            model = Detector(.5)
            for arm in ('baseline', 'expanded', 'hard'):
                for fraction in (0., .5, 1.):
                    options = dict(cfg, robust_arm=arm, robust_hard_fraction=fraction)
                    fast, a = step.select_candidates(model, units, options, diagnostics=False)
                    full, b = step.select_candidates(model, units, options, diagnostics=True)
                    key = lambda rows: [(u[0]['source_id'], u[0].get('robust_mechanism')) for u in rows]
                    self.assertEqual(key(fast), key(full))
                    self.assertEqual(a['selected_mechanism_by_class'], b['selected_mechanism_by_class'])
                    self.assertEqual(a['counts'], b['counts'])
                    if arm != 'hard' or fraction == 0.:
                        self.assertEqual(a['scoring_encoded_views'], 0)
                    else:
                        self.assertLess(a['scoring_encoded_views'], b['scoring_encoded_views'])
                    json.dumps(a, allow_nan=False)

    def test_updates_optimizer_and_dropout_rng_equal_full_scoring(self):
        cfg, units = fixture()
        for arm in ('baseline', 'expanded', 'hard'):
            torch.manual_seed(16)
            one = Detector(.35)
            two = copy.deepcopy(one)
            a, b = optimizer(one), optimizer(two)
            before = rng_state()
            fast = step.train_step(one, a, units, dict(cfg, robust_arm=arm, _candidate_diagnostics=False))
            after = rng_state()
            restore_rng(before)
            full = step.train_step(two, b, units, dict(cfg, robust_arm=arm, _candidate_diagnostics=True))
            assert_state_equal(self, one.state_dict(), two.state_dict())
            assert_state_equal(self, a.state_dict(), b.state_dict())
            assert_rng_equal(self, after, rng_state())
            self.assertEqual(fast['classification_loss'], full['classification_loss'])
            self.assertEqual(fast['loss'], full['loss'])
            self.assertEqual(fast['cells'], full['cells'])
            self.assertEqual(fast['training_encoded_views'], full['training_encoded_views'])

    def test_only_hard_noisy_candidates_reach_scorer(self):
        cfg, units = fixture()
        model = Detector()
        original = model.encode
        seen = []
        def encode(waves):
            seen.extend(id(wave) for wave in waves)
            return original(waves)
        model.encode = encode
        _, status = step.select_candidates(model, units, dict(cfg, robust_arm='hard'),
                                           diagnostics=False)
        clean = {id(c[0]['pcl_refs']['clean']['wave']) for u in units
                 for c in u[0].get('robust_candidates', [])}
        legacy = {id(u[0]['robust_candidates'][0][0]['wave']) for u in units if u[0]['role'] == 'noisy'}
        self.assertEqual(len(seen), 36)
        self.assertFalse(set(seen) & (clean | legacy))
        self.assertEqual(status['scoring_encoded_views'], 36)
        self.assertIsNone(status['mean_legacy_margin'])
        self.assertIsNone(status['mean_noisy_minus_clean_margin'])
        self.assertEqual(status['transitions'], {})
        self.assertEqual(status['margin_observations']['selected'], 12)

    def test_both_pcl_branches_keep_identical_update(self):
        cfg, units = fixture()
        for unit in units:
            for candidate in unit[0].get('robust_candidates', []):
                row = candidate[0]
                row['pcl_noisy_index'] = 0
                row['pcl_refs']['official'] = dict(row['pcl_refs']['clean'],
                    role='official_ref', wave=row['wave']*.4)
                row['pcl_pairs'] = dict(noise=[[0, 2000, 0, 2000]],
                                        platform=[[0, 2000, 0, 2000]])
        torch.manual_seed(24)
        a = Detector(.2)
        a.pcl_scale = SimpleNamespace(is_frozen=True, scale=torch.ones(5))
        b = copy.deepcopy(a)
        oa, ob = optimizer(a), optimizer(b)
        runtime = dict(cfg, consistency='dual', pcl_variance_weight=0., robust_arm='hard')
        before = rng_state()
        fast = step.train_step(a, oa, units, dict(runtime, _candidate_diagnostics=False))
        restore_rng(before)
        full = step.train_step(b, ob, units, dict(runtime, _candidate_diagnostics=True))
        for branch in ('noise', 'platform'):
            self.assertGreater(fast['pcl_losses'][branch]['weighted_loss'], 0.)
            self.assertEqual(fast['pcl_losses'][branch], full['pcl_losses'][branch])
        assert_state_equal(self, a.state_dict(), b.state_dict())
        assert_state_equal(self, oa.state_dict(), ob.state_dict())

    def test_unscored_is_missing_not_perfect_diagnostics(self):
        cfg, units = fixture()
        with patch.object(Detector, 'encode', side_effect=AssertionError('unexpected scoring')):
            for arm in ('baseline', 'expanded'):
                _, value = step.select_candidates(Detector(), units, dict(cfg, robust_arm=arm),
                                                  diagnostics=False)
                self.assertEqual(value['scoring_batches'], 0)
                self.assertEqual(value['transitions'], {})
                self.assertIsNone(value['mean_selected_margin'])
                self.assertEqual(value['margin_observations']['selected'], 0)

    def test_epoch_diagnostic_schedule_matches_worker_contract(self):
        cfg, units = fixture()
        for epoch_step in (0, 1, 199, 200, 201):
            full = diagnostic_step(epoch_step)
            _, info = step.select_candidates(Detector(), units,
                dict(cfg, robust_arm='baseline', _candidate_diagnostics=full))
            self.assertEqual(info['scoring_encoded_views'], 192 if full else 0)
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                diagnostic_step(invalid)


class ExecutionCompatibilityTests(unittest.TestCase):
    def test_registered_manifest_upgrade_preserves_saved_inventory(self):
        from .config import code_inventory
        from .execution import PREVIOUS_MANIFESTS
        root = Path(__file__).resolve().parents[1]
        saved = code_inventory(root)
        saved[str(root/MANIFEST)] = PREVIOUS_MANIFESTS[0]
        before = copy.deepcopy(saved)
        result = verify_execution_code(saved, root)
        self.assertIn(MANIFEST, result['upgraded_files'])
        self.assertEqual(saved, before)
        saved[str(root/MANIFEST)] = 'unregistered manifest'
        with self.assertRaisesRegex(ValueError, 'Unregistered source change'):
            verify_execution_code(saved, root)

    def make(self, root):
        files = dict(changed=b'new implementation', stable=b'unchanged')
        old_hash = hashlib.sha256(b'old implementation').hexdigest()
        current, saved = {}, {}
        for name, payload in files.items():
            relative = f'w2v_v3191/{name}.py'
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            value = hashlib.sha256(payload).hexdigest()
            saved[str(path)] = old_hash if name == 'changed' else value
            if name == 'changed':
                current[relative] = value
        added = root / 'w2v_v3191/new.py'
        added.write_bytes(b'new dependency')
        current['w2v_v3191/new.py'] = hashlib.sha256(added.read_bytes()).hexdigest()
        spec = dict(execution_policy=EXECUTION_POLICY, predecessor='fixture',
                    current=current, previous={'w2v_v3191/changed.py':[old_hash]})
        (root / MANIFEST).write_text(json.dumps(spec), encoding='utf-8')
        return saved

    def test_only_exact_registered_old_to_new_bytes_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = self.make(root)
            before = copy.deepcopy(saved)
            result = verify_execution_code(saved, root)
            self.assertEqual(list(result['upgraded_files']), ['w2v_v3191/changed.py'])
            self.assertEqual(saved, before)
            bad = dict(saved, **{str(root/'w2v_v3191/changed.py'):'unknown prior bytes'})
            with self.assertRaisesRegex(ValueError, 'Unregistered source change'):
                verify_execution_code(bad, root)
            (root/'w2v_v3191/stable.py').write_bytes(b'unrelated modification')
            with self.assertRaisesRegex(ValueError, 'Unregistered source change'):
                verify_execution_code(saved, root)

    def test_modified_replacement_and_new_dependency_are_rejected(self):
        for name in ('changed', 'new'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                saved = self.make(root)
                (root/f'w2v_v3191/{name}.py').write_bytes(b'not the tested replacement')
                with self.assertRaisesRegex(ValueError, 'registered bytes'):
                    verify_execution_code(saved, root)

    def test_release_inventory_supports_previous_resume_without_rewriting_config(self):
        from .config import code_inventory
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root/MANIFEST).read_text(encoding='utf-8'))
        current = code_inventory(root)
        previous = {}
        for path, checksum in current.items():
            relative = Path(path).relative_to(root).as_posix()
            if relative == MANIFEST or (relative in spec['current'] and relative not in spec['previous']):
                continue
            previous[path] = spec.get('previous', {}).get(relative, [checksum])[0]
        cfg = dict(code_fingerprints=previous, classification_ce_weights={
            'online':.25, 'clean_rtc':.25, 'noisy':.5})
        unchanged = copy.deepcopy(cfg)
        with tempfile.TemporaryDirectory() as directory:
            state = dict(cursor=81, epoch=0, next_step=81, optimizer={'untouched':True})
            record_execution(directory, cfg, state)
            record_execution(directory, cfg, state)
            receipt = json.loads((Path(directory)/'execution_policy.json').read_text())
            self.assertEqual(len(receipt['events']), 1)
            self.assertEqual(receipt['events'][0]['from_update'], 81)
            self.assertTrue(receipt['events'][0]['upgraded_files'])
            self.assertEqual(state['optimizer'], {'untouched':True})
        self.assertEqual(cfg, unchanged)
        self.assertFalse(verify_execution_code(current, root)['upgraded_files'])


if __name__ == '__main__':
    unittest.main()
