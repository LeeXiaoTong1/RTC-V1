"""Retain actual warmup optimizer/RNG state while repairing only calibration."""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from .calibration import CANDIDATES, LEGACY_CANDIDATES, CalibrationError
from .common import ROOT, atomic_json, atomic_save, digest, fingerprint, read_json
from .config import code_inventory
from .identity import REGISTRY_SCHEMA
from .recover_calibration import OLD_CODE, recover, verify_code_upgrade
from .test_meta import assert_state_equal
from . import test_training as training_fixture
from .train import run_experiment


def old_inventory():
    inventory = code_inventory()
    inventory.pop(str(ROOT / 'w2v_v31810' / 'recover_calibration.py'))
    for name, checksum in OLD_CODE.items():
        inventory[str(ROOT / name)] = checksum
    return inventory


def failed_calibration(*args, **kwargs):
    records = [dict(alpha=alpha, finite=True, relative_drop=.09) for alpha in LEGACY_CANDIDATES for _ in range(12)]
    raise CalibrationError('No inner-step candidate passed the fixed Train calibration gates', dict(
        status='failed', candidates=list(LEGACY_CANDIDATES), task_count=12,
        selected_alpha=None, records=records))


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, stack):
        calibrator = training_fixture.LifecycleTests.patches(self, stack, failed_calibration)
        cfg, protocol = training_fixture.settings()
        cfg.update(inner_candidates=list(LEGACY_CANDIDATES), calibration_tasks=12,
                   code_fingerprints=old_inventory(), source_registry=dict(
                       schema=REGISTRY_SCHEMA, registry_hash='fixture', train_identity_hash='fixture'))
        run = training_fixture.make_run(root, 'original', cfg)
        manifests = {'train_rows.json': [], 'dev_rows.json': [],
                     'validation_protocol.json': protocol, 'source_registry.json': {}}
        for name, value in manifests.items():
            atomic_json(run / name, value)
        cfg['saved_manifests'] = {name: digest(run / name) for name in manifests}
        atomic_json(run / 'config.json', cfg)
        atomic_json(run / 'input_audit.json', dict(fixture=True))
        atomic_json(run / 'report_request.json', dict(destination=str(root / 'downloads'), upload=True))
        atomic_json(run / 'submission_request.json', dict(out=str(root / 'downloads' / 'old_submission'), upload=True))
        with self.assertRaises(CalibrationError):
            run_experiment(cfg, run, [], [], protocol, model_factory=training_fixture.tiny_model,
                           evaluator=training_fixture.Evaluator(self, run))
        return run, cfg, protocol, calibrator

    def test_recovery_copies_exact_warmup_and_resumes_without_retraining_it(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            run, cfg, protocol, calibrator = self.fixture(root, stack)
            original = {p.relative_to(run).as_posix(): digest(p) for p in run.rglob('*') if p.is_file()}
            before = torch.load(run / 'last.pt', weights_only=True)
            recovered = recover(run)
            self.assertNotEqual(run, recovered)
            self.assertEqual(original, {p.relative_to(run).as_posix(): digest(p) for p in run.rglob('*') if p.is_file()})
            after = torch.load(recovered / 'last.pt', weights_only=True)
            new_cfg = read_json(recovered / 'config.json')
            self.assertEqual(after['identity'], fingerprint(new_cfg))
            for field in before:
                if field != 'identity':
                    assert_state_equal(self, before[field], after[field])
            self.assertEqual({key for key in cfg if cfg[key] != new_cfg[key]}, {'inner_candidates', 'code_fingerprints'})
            self.assertEqual(new_cfg['inner_candidates'], list(CANDIDATES))
            self.assertEqual(read_json(recovered / 'calibration_recovery.json')['status'], 'ready')
            self.assertFalse((recovered / 'failure.log').exists())
            self.assertFalse((recovered / 'inner_calibration_failure.json').exists())
            self.assertTrue((recovered / 'recovery' / 'original_calibration_failure.json').is_file())
            self.assertEqual(Path(read_json(recovered / 'submission_request.json')['out']).name,
                             recovered.name + '_submission_best')
            def accept(model, optimizer, factory, settings, **kwargs):
                self.assertEqual(settings['inner_candidates'], list(CANDIDATES))
                return dict(status='complete', selected_alpha=.0001)
            calibrator.side_effect = accept
            evaluator = training_fixture.Evaluator(self, recovered)
            result = run_experiment(new_cfg, recovered, [], [], protocol,
                                    model_factory=training_fixture.tiny_model, evaluator=evaluator)
            self.assertEqual(result['completed_epochs'], 2)
            resumed = torch.load(recovered / 'last.pt', weights_only=True)
            self.assertEqual(resumed['cursor'], 4)
            self.assertEqual(len(resumed['history']), 2)
            self.assertEqual(resumed['history'][0], before['history'][0])
            self.assertEqual(evaluator.calls, ['select', 'audit'])

    def test_unrelated_model_change_cannot_be_bypassed(self):
        current = code_inventory()
        old = old_inventory()
        old[str(ROOT / 'w2v_v31810' / 'model.py')] = 'not_the_published_model'
        with self.assertRaisesRegex(ValueError, 'Unrelated or unknown'):
            verify_code_upgrade(old, current)
        old = old_inventory()
        old.pop(str(ROOT / 'w2v_v31810' / 'meta.py'))
        with self.assertRaisesRegex(ValueError, 'Unexpected code inventory'):
            verify_code_upgrade(old, current)

    def test_already_joint_or_calibrated_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            run, cfg, protocol, _ = self.fixture(root, stack)
            state = torch.load(run / 'last.pt', weights_only=True)
            for changed in ({'alpha': .001}, {'next_step': 1}, {'epoch': 2}, {'cursor': 3}):
                modified = copy.deepcopy(state)
                modified.update(changed)
                atomic_save(run / 'last.pt', modified)
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    recover(run)
            self.assertEqual(list(root.iterdir()), [run])

    def test_modified_source_manifest_blocks_copy(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            run, cfg, protocol, _ = self.fixture(root, stack)
            atomic_json(run / 'train_rows.json', [dict(changed=True)])
            with self.assertRaisesRegex(ValueError, 'Pinned input'):
                recover(run)
            self.assertEqual(list(root.iterdir()), [run])

    def test_successful_calibration_report_cannot_be_relabelled_as_failure(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            run, cfg, protocol, _ = self.fixture(root, stack)
            report = read_json(run / 'inner_calibration_failure.json')
            for row in report['details']['records']:
                row['relative_drop'] = .01
            atomic_json(run / 'inner_calibration_failure.json', report)
            with self.assertRaisesRegex(ValueError, 'completed five-candidate'):
                recover(run)
            self.assertEqual(list(root.iterdir()), [run])


if __name__ == '__main__':
    unittest.main()
