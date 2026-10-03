"""Exercise actual selective file deletion, pinning, races and retained Dev display."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from . import maintenance as m


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root/'exp').mkdir()

    def run_fixture(self, name='w2v_old', complete=True):
        run = self.root/'exp'/name
        run.mkdir(parents=True)
        for name in ('best_model.pt', 'best_weighted.pt', 'control_best.pt', 'candidate_best.pt',
                     'last.pt', 'epoch_1.pt', 'epoch_2_step_40.pth', 'custom_model.pt', 'audio.wav'):
            (run/name).write_bytes(name.encode())
        (run/'config.json').write_text('{}')
        if complete:
            (run/'completed.json').write_text('{"success":true}')
        return run

    def test_deletes_only_named_old_intermediates_and_keeps_all_best_and_unknown_data(self):
        run = self.run_fixture()
        plan = m.build_plan(self.root)
        self.assertEqual({Path(x['path']).name for x in plan['checkpoints']},
                         {'last.pt', 'epoch_1.pt', 'epoch_2_step_40.pth'})
        self.assertTrue((run/'last.pt').exists(), 'planning is read-only')
        report = self.root/'exp'/'cleanup.json'
        removed = m.apply_plan(plan, report, lambda: [])
        self.assertEqual(len(removed), 3)
        for name in ('best_model.pt', 'best_weighted.pt', 'control_best.pt', 'candidate_best.pt',
                     'custom_model.pt', 'audio.wav', 'config.json', 'completed.json'):
            self.assertTrue((run/name).is_file(), name)
        self.assertEqual(json.loads(report.read_text())['status'], 'complete')

    def test_latest_resume_current_v35_and_config_references_are_preserved(self):
        old = self.run_fixture()
        current = self.run_fixture('w2v_v35_current')
        (self.root/'exp'/'.latest_v34_run').write_text(str(old))
        (current/'config.json').write_text(json.dumps({'warm_checkpoint': str(old/'epoch_1.pt')}))
        plan = m.build_plan(self.root)
        self.assertEqual([Path(x['path']) for x in plan['checkpoints']], [old/'epoch_2_step_40.pth'])
        self.assertIn(str(old/'last.pt'), plan['protected'])

    def test_failed_unfinished_and_foreign_experiments_are_not_removed(self):
        unfinished = self.run_fixture('w2v_unfinished', False)
        (unfinished/'failed.json').write_text('{"error":"Interrupted"}')
        foreign = self.run_fixture('user_experiment')
        self.assertFalse(m.build_plan(self.root)['checkpoints'])
        (unfinished/'stopped.json').write_text('{"status":"stopped"}')
        self.assertEqual(len(m.build_plan(self.root)['checkpoints']), 3)
        self.assertTrue((foreign/'last.pt').is_file())

    def test_submission_pin_and_inode_alias_are_retained(self):
        run = self.run_fixture()
        submitted = run/'epoch_1.pt'
        metadata = self.root/'submission_meta.json'
        metadata.write_text(json.dumps({'checkpoint': str(submitted), 'checkpoint_sha256': m.sha256(submitted)}))
        alias = run/'epoch_3.pt'
        try:
            alias.hardlink_to(submitted)
        except OSError:
            self.skipTest('Hardlinks unavailable')
        plan = m.build_plan(self.root, metadata_files=[metadata])
        self.assertNotIn(str(submitted), [x['path'] for x in plan['checkpoints']])
        self.assertNotIn(str(alias), [x['path'] for x in plan['checkpoints']])
        self.assertEqual(m.protected_submission(metadata), (submitted, m.sha256(submitted)))

    def test_changed_target_prevents_all_deletion(self):
        run = self.run_fixture()
        plan = m.build_plan(self.root)
        (run/'epoch_1.pt').write_bytes(b'updated')
        with self.assertRaisesRegex(RuntimeError, 'target changed'):
            m.apply_plan(plan, self.root/'exp'/'cleanup.json', lambda: [])
        self.assertTrue((run/'last.pt').is_file())

    def test_changed_protection_metadata_prevents_all_deletion(self):
        run = self.run_fixture()
        plan = m.build_plan(self.root)
        (run/'config.json').write_text(json.dumps({'warm_checkpoint':str(run/'last.pt')}))
        with self.assertRaisesRegex(RuntimeError, 'metadata changed'):
            m.apply_plan(plan, self.root/'exp'/'cleanup.json', lambda: [])
        self.assertTrue((run/'last.pt').is_file())

    def test_active_job_before_or_after_planning_blocks_deletion(self):
        run = self.run_fixture()
        plan = m.build_plan(self.root)
        with self.assertRaisesRegex(RuntimeError, 'Active'):
            m.apply_plan(plan, self.root/'exp'/'cleanup.json', lambda:[{'pid':123,'command':'w2v_v34.train'}])
        states = iter([[], [{'pid':124,'command':'w2v_v35.workflow'}]])
        with self.assertRaisesRegex(RuntimeError, 'Job started'):
            m.apply_plan(plan, self.root/'exp'/'cleanup.json', lambda:next(states))
        self.assertTrue((run/'last.pt').is_file())

    def test_explicit_outside_and_traversal_paths_are_rejected(self):
        run = self.run_fixture()
        external = self.root/'last.pt'
        external.write_bytes(b'outside')
        plan = m.build_plan(self.root)
        plan['checkpoints'][0] = {'path':str(external), 'identity':m.identity(external), 'kind':'checkpoint'}
        with self.assertRaises(ValueError):
            m.apply_plan(plan, self.root/'exp'/'cleanup.json', lambda:[])
        with self.assertRaises(ValueError):
            m.safe_path(run/'..'/'..'/'last.pt', self.root/'exp')
        self.assertTrue(external.is_file())

    def test_symlink_directory_and_metadata_cannot_hide_targets(self):
        run = self.run_fixture()
        external = self.root/'outside'
        external.mkdir()
        (external/'last.pt').write_bytes(b'outside')
        link = run/'linked'
        try:
            link.symlink_to(external, target_is_directory=True)
        except OSError:
            self.skipTest('Symlink creation unavailable')
        plan = m.build_plan(self.root)
        self.assertNotIn(str(link/'last.pt'), [x['path'] for x in plan['checkpoints']])
        with self.assertRaises(ValueError):
            m.safe_path(link/'last.pt', self.root/'exp')
        (run/'source.json').symlink_to(run/'config.json')
        with self.assertRaisesRegex(ValueError, 'metadata'):
            m.build_plan(self.root)

    def test_obsolete_scripts_archived_verbatim_modified_scripts_retained(self):
        self.run_fixture()
        content = b'#!/bin/bash\necho archived\n'
        script = self.root/'obsolete.sh'
        script.write_bytes(content)
        modified = self.root/'changed.sh'
        modified.write_bytes(b'user modified')
        digests = {script.name:hashlib.sha256(content).hexdigest(), modified.name:'0'*64}
        with patch.object(m, 'OBSOLETE_SCRIPTS', digests):
            plan = m.build_plan(self.root)
            self.assertEqual([x['path'] for x in plan['scripts']], [str(script)])
            report = self.root/'exp'/'cleanup.json'
            m.apply_plan(plan, report, lambda:[])
            saved = json.loads(report.read_text())
        with zipfile.ZipFile(saved['script_archive']) as archive:
            self.assertEqual(archive.read(script.name), content)
        self.assertFalse(script.exists())
        self.assertTrue(modified.is_file())

    def test_malformed_metadata_or_mismatched_submission_fails_closed(self):
        run = self.run_fixture()
        (run/'config.json').write_text('not json')
        with self.assertRaises(ValueError):
            m.build_plan(self.root)
        metadata = self.root/'submission_meta.json'
        metadata.write_text(json.dumps({'checkpoint':str(run/'best_model.pt'),'checkpoint_sha256':'0'*64}))
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            m.protected_submission(metadata)

    def test_actual_v33_submission_metadata_resolves_completed_arm(self):
        run = self.root/'exp'/'w2v_v33_original'
        arm = run/'candidate'
        arm.mkdir(parents=True)
        checkpoint = arm/'best_model.pt'
        checkpoint.write_bytes(b'platform-submitted-model')
        (run/'comparison.json').write_text(json.dumps({'status':'complete','selected_arm':'candidate',
            'checkpoint':str(checkpoint),'arms':{'candidate':{'selected':{'tag':'baseline'}}}}))
        metadata = self.root/'submission_meta.json'
        metadata.write_text(json.dumps({'checkpoint_sha256':m.sha256(checkpoint),'checkpoint_tag':'baseline'}))
        self.assertEqual(m.protected_submission(metadata,run),(checkpoint,m.sha256(checkpoint)))
        metadata.write_text(json.dumps({'checkpoint_sha256':m.sha256(checkpoint),'checkpoint_tag':'epoch_1'}))
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            m.protected_submission(metadata,run)


class ConsoleTests(unittest.TestCase):
    def test_reference_and_epoch_commit_render_once_above_progress(self):
        import v35_console
        import v33_console
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            dev = dict(clean_f1=.98, seen_f1=.96, heldout_f1=.96, noisy_f1=.96, weighted_f1=.966)
            for tag in ('reference', 'epoch_1'):
                (run/(tag+'.json')).write_text(json.dumps({'tag':tag, 'arm':'V3.5', 'phase':'head',
                    'dev':dev,'decision':{'save':['best_safe'] if tag=='epoch_1' else [],'action':'continue'}}))
            (run/'report.md').write_text('| Checkpoint | Phase |\n| reference | reference |\n| epoch_1 | head |\n')
            log = run/'log.txt'
            log.write_text('V35_RUN='+str(run)+'\nV35_EVENT={"kind":"validation"}\n')
            state = run/'progress.json'
            state.write_text(json.dumps({'status':'complete','label':'V3.5 complete','current':10,'total':10}))
            output = io.StringIO()
            v35_console.watch(log, state, once=True, stream=output)
            text = output.getvalue()
            self.assertEqual(text.count('protected reference (comparison only)'), 1)
            self.assertEqual(text.count('epoch_1 (epoch complete)'), 1)
            self.assertIn('promoted=True', text)
            self.assertLess(text.index('Weighted=96.600'), text.index('[Phase] V3.5 complete'))
            self.assertIsNone(v33_console.TAG.fullmatch('reference'), 'old viewer globals must remain unchanged')


if __name__ == '__main__':
    unittest.main()
