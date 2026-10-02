"""An exact old release may receive the reviewed CPU transport repair only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from w2v_aasist.runtime import sha256
from . import SCHEMA
from . import resume_compat as compat


def fixture():
    dependencies = {'torch': '2.5.1+cu118', 'transformers': '4.38.2',
                    'w2v_v3/train.py': 'unchanged-old-training-code',
                    'w2v_v32/performance.py': 'unchanged-performance-code'}
    before = {**dependencies, **compat.PUBLISHED_V33}
    target = {key: hashlib.sha256(key.encode()).hexdigest()
              for key in compat.CHANGED_FILES | compat.ADDED_FILES
              if key != 'w2v_v33/resume_compat.py'}
    after = {**before, **target,
             'w2v_v33/resume_compat.py': sha256(Path(compat.__file__).resolve())}
    cfg = {'seed': 7, 'workers': 4, 'source_batch': 16, 'pair_weight': .02}
    inputs = {'train/manifest.jsonl': 'identical-data-hash', 'protected-best.pt': 'unchanged'}
    state = {'schema': SCHEMA, 'kind': 'training', 'config': cfg,
             'data_fingerprints': inputs, 'source_hashes': before, 'tag': 'baseline',
             'optimizer': {'state': {'0': 'unchanged Adam state'}},
             'rng': {'python': 'unchanged RNG'}, 'source_exposures': 0}
    return target, before, after, cfg, inputs, state


class ResumeCompatibilityTests(unittest.TestCase):
    def test_deployed_target_hashes_match_exact_utf8_lf_files(self):
        root = Path(compat.__file__).resolve().parent.parent
        for name, digest in {**compat.PUBLISHED_V33, **compat.TARGET_HASHES}.items():
            # Git publishes UTF-8/LF; Windows checkouts can use CRLF locally.
            content = (root/name).read_text(encoding='utf-8').encode('utf-8')
            self.assertEqual(hashlib.sha256(content).hexdigest(), digest, name)
        self.assertEqual(set(compat.TARGET_HASHES),
                         (compat.CHANGED_FILES | compat.ADDED_FILES)-{'w2v_v33/resume_compat.py'})

    def test_exact_new_source_resume_needs_no_exception_or_audit(self):
        _, _, after, cfg, inputs, state = fixture()
        state['source_hashes'] = after
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(compat.validate_resume(state, cfg, inputs, after,
                                                    Path(td)/'unused.pt', td))
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_known_release_migration_is_audited_without_mutating_state_or_checkpoint(self):
        target, _, after, cfg, inputs, state = fixture()
        original_state = copy.deepcopy(state)
        with tempfile.TemporaryDirectory() as td, patch.object(compat, 'TARGET_HASHES', target):
            checkpoint = Path(td)/'last.pt'
            checkpoint.write_bytes(b'original saved checkpoint bytes')
            original_hash = sha256(checkpoint)
            result = compat.validate_resume(state, cfg, inputs, after, checkpoint, td)
            self.assertEqual(sha256(checkpoint), original_hash)
            self.assertEqual(state, original_state)
            audit = json.loads(result.read_text(encoding='utf-8'))
            self.assertEqual(audit['checkpoint_sha256'], original_hash)
            self.assertFalse(audit['checkpoint_rewritten'])
            self.assertFalse(audit['optimizer_or_training_state_modified'])
            self.assertEqual(set(audit['changed_sources']), compat.CHANGED_FILES | compat.ADDED_FILES)
            self.assertEqual(compat.validate_resume(state, cfg, inputs, after, checkpoint, td), result)
            audit['checkpoint_sha256'] = 'tampered'
            result.write_text(json.dumps(audit), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                compat.validate_resume(state, cfg, inputs, after, checkpoint, td)

    def test_unknown_old_code_including_previously_changed_filenames_is_rejected(self):
        target, before, after, _, _, _ = fixture()
        with patch.object(compat, 'TARGET_HASHES', target):
            self.assertTrue(compat.approved_code_transition(before, after))
            for name in compat.PUBLISHED_V33:
                with self.subTest(name=name):
                    self.assertFalse(compat.approved_code_transition({**before, name: 'unknown'}, after))
            self.assertFalse(compat.approved_code_transition({**before, 'w2v_v33/extra.py': 'x'}, after))
            missing = dict(before); missing.pop('w2v_v33/test_train.py')
            self.assertFalse(compat.approved_code_transition(missing, after))

    def test_unknown_target_code_and_non_v33_or_dependency_changes_are_rejected(self):
        target, before, after, _, _, _ = fixture()
        with patch.object(compat, 'TARGET_HASHES', target):
            for name in after:
                with self.subTest(name=name):
                    self.assertFalse(compat.approved_code_transition(before, {**after, name: 'unknown'}))
            self.assertFalse(compat.approved_code_transition(before, {**after, 'w2v_v33/extra.py': 'x'}))
            self.assertFalse(compat.approved_code_transition(before, {**after, 'new_dependency': 'x'}))
            missing = dict(after); missing.pop('w2v_v33/transport.py')
            self.assertFalse(compat.approved_code_transition(before, missing))

    def test_schema_config_and_input_changes_cannot_use_compatibility(self):
        target, _, after, cfg, inputs, state = fixture()
        with tempfile.TemporaryDirectory() as td, patch.object(compat, 'TARGET_HASHES', target):
            for mutation in ({'schema': 'other'}, {'kind': 'weights'},
                             {'config': {**cfg, 'workers': 0}},
                             {'config': {**cfg, 'pair_weight': 0}},
                             {'data_fingerprints': {**inputs, 'train/manifest.jsonl': 'changed'}}):
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'identical config, metadata, code, versions'):
                    compat.validate_resume({**state, **mutation}, cfg, inputs, after,
                                           Path(td)/'not-read.pt', td)
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_real_saved_baseline_migration_preserves_optimizer_and_replays_exact_training(self):
        from . import train as training
        from .test_train import fixture as training_fixture, TrainingIntegrationTests
        target, before, after, _, _, _ = fixture()
        with tempfile.TemporaryDirectory() as td, patch.object(compat, 'TARGET_HASHES', target):
            root = Path(td)
            cfg, discovery, _, _ = training_fixture(root)
            cfg['joint_epochs'] = 1
            interrupted, reference = root/'interrupted', root/'reference'
            with patch.object(training, 'build_data', discovery), patch.object(
                    training, 'source_fingerprints', return_value=before), patch.object(
                    training, 'supervised_step', side_effect=RuntimeError('interrupt before update')):
                with self.assertRaisesRegex(RuntimeError, 'interrupt before update'):
                    training.train(cfg, interrupted)
            checkpoint = interrupted/'last.pt'
            original_hash = sha256(checkpoint)
            old = training.read_state(checkpoint)
            self.assertEqual(old['global_steps'], 0)
            self.assertFalse(old['optimizer']['state'])
            with patch.object(training, 'build_data', discovery), patch.object(
                    training, 'source_fingerprints', return_value=after):
                training.train(cfg, interrupted, checkpoint)
                training.train(cfg, reference)
            actual, expected = training.read_state(checkpoint), training.read_state(reference/'last.pt')
            TrainingIntegrationTests().assert_same_training(actual, expected)
            self.assertEqual(actual['source_hashes'], after)
            audits = list(interrupted.glob('resume_transport_compat_*.json'))
            self.assertEqual(len(audits), 1)
            self.assertEqual(json.loads(audits[0].read_text())['checkpoint_sha256'], original_hash)


if __name__ == '__main__':
    unittest.main()
