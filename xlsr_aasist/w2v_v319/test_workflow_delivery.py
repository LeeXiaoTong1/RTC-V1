"""Exercise the CLI workflow while replacing GPU work and external delivery.

Request persistence and CLI validation are real. Only training, final auditing,
artifact export/upload, locks and machine-idleness checks are replaced.
"""
from contextlib import ExitStack, nullcontext
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from w2v_v39.common import atomic_json, read_json
from . import workflow


class WorkflowDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root/'exp').mkdir()
        self.download = self.root/'downloads'
        self.protocol = self.root/'progress.txt'
        self.protocol.write_text('one\ntwo\n', encoding='ascii')
        self.audio = self.root/'audio'
        self.audio.mkdir()
        self.cfg = dict(version='3.19', epochs=5, device='cpu', workers=0,
                        immutable_setting='test-recorded-value')
        self.events = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

        def patched(name, **kwargs):
            return self.stack.enter_context(patch(name, **kwargs))

        self.idle = patched('w2v_v319.workflow.ensure_idle')
        self.lock = patched('w2v_v319.workflow.run_lock', side_effect=lambda *_a, **_k: nullcontext())
        self.stack.enter_context(patch.object(workflow, 'ROOT', self.root))
        self.configuration = patched('w2v_v319.workflow.configuration',
                                     side_effect=lambda args: copy.deepcopy(self.cfg))
        self.verify = patched('w2v_v319.workflow.verify_inputs')
        self.publish = patched('w2v_v319.workflow.publish')
        self.load_selected = patched('w2v_v319.workflow.load_selected',
                                     side_effect=lambda run: self.events.append(('load_selected', str(run))))
        patched('w2v_v319.delivery.read_protocol', return_value=[dict(id='one'), dict(id='two')])

        def train(cfg, run):
            self.events.append(('train', cfg['epochs']))
            (run/'last.pt').write_bytes(b'committed-test-checkpoint')
            atomic_json(run/'completed.json', dict(status='complete'))

        self.train = patched('w2v_v319.workflow.run_experiment', side_effect=train)
        self.audit = patched('w2v_v319.workflow.final_audit',
                             side_effect=lambda cfg, run, kind: self.events.append(('audit', kind)))
        self.finish = patched('w2v_v319.delivery.finish_request', side_effect=self.finish_request)
        self.summary = patched('w2v_v319.workflow.report',
                               side_effect=lambda run: self.events.append(('summary', str(run))))
        self.archive = patched('w2v_v319.reports.report', side_effect=self.archive_report)

    def finish_request(self, run, cfg, request):
        self.events.append(('submission', request['checkpoint_kind'], request['upload']))
        self.assertTrue((run/'completed.json').is_file())
        self.assertEqual(cfg['immutable_setting'], 'test-recorded-value')

    def archive_report(self, run, destination, upload):
        self.events.append(('report', destination, upload))

    def launch(self, args):
        with patch.object(sys, 'argv', ['w2v_v319.workflow', *args]):
            workflow.main()

    def new_args(self, kind='best_guarded'):
        return ['--submit-best', '--upload-submission', '--upload-temp',
            '--submission-checkpoint', kind, '--submission-protocol', str(self.protocol),
            '--submission-audio-root', str(self.audio), '--download-dir', str(self.download)]

    def latest(self):
        return Path((self.root/'exp'/'.latest_v319_run').read_text(encoding='utf-8').strip())

    def saved_run(self):
        run = self.root/'exp'/'existing_v319'
        run.mkdir()
        # Different from the parser default: resume must use this saved value.
        cfg = dict(self.cfg, epochs=7)
        atomic_json(run/'config.json', cfg)
        (run/'last.pt').write_bytes(b'committed-test-checkpoint')
        atomic_json(run/'completed.json', dict(status='complete'))
        args = workflow.parser().parse_args(self.new_args())
        workflow.delivery.prepare_request(args, run, cfg)
        workflow.reports.prepare_request(args, run, self.new_args())
        return run, cfg

    def test_full_cli_trains_audits_requested_best_then_delivers_both_artifacts(self):
        self.launch(self.new_args())
        run = self.latest()
        self.assertEqual([e[0] for e in self.events], ['train', 'audit', 'submission', 'summary', 'report'])
        self.assertEqual(self.events[1], ('audit', 'best_guarded'))
        self.assertEqual(self.events[2], ('submission', 'best_guarded', True))
        self.assertEqual(self.events[-1], ('report', str(self.download), True))
        self.assertEqual(read_json(run/'config.json'), self.cfg)
        self.assertEqual(read_json(run/'submission_request.json')['checkpoint_kind'], 'best_guarded')
        self.publish.assert_called_once_with('V3.19 complete', status='complete', force=True)
        self.load_selected.assert_not_called()
        self.assertEqual(self.idle.call_count, 2)

    def test_submission_failure_still_reports_and_signals_failure(self):
        def fail(run, cfg, request):
            self.finish_request(run, cfg, request)
            raise RuntimeError('test submission upload failure')

        self.finish.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, 'Submission: test submission upload failure'):
            self.launch(self.new_args())
        run = self.latest()
        self.assertEqual([e[0] for e in self.events], ['train', 'audit', 'submission', 'summary', 'report'])
        self.assertEqual(self.events[-1][-1], True)
        self.assertTrue((run/'submission_failure.log').is_file())
        self.assertEqual((run/'last.pt').read_bytes(), b'committed-test-checkpoint')
        self.publish.assert_not_called()

    def test_audit_failure_does_not_prevent_saved_best_delivery(self):
        def fail(cfg, run, kind):
            self.events.append(('audit', kind))
            raise RuntimeError('test audit failure')

        self.audit.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, 'Final audit: test audit failure'):
            self.launch(self.new_args('best_weighted'))
        self.assertEqual(self.events[1], ('audit', 'best_weighted'))
        self.assertEqual(self.events[2], ('submission', 'best_weighted', True))
        self.assertEqual(self.events[-1][0], 'report')
        self.assertTrue((self.latest()/'audit_failure.log').is_file())

    def test_completed_resume_retains_requests_and_exact_recorded_config(self):
        run, cfg = self.saved_run()
        names = ('config.json', 'submission_request.json', 'report_request.json', 'last.pt')
        before = {name: (run/name).read_bytes() for name in names}
        self.launch(['--resume', str(run)])
        self.configuration.assert_not_called()
        self.train.assert_not_called()
        self.verify.assert_called_once_with(cfg)
        self.assertEqual([e[0] for e in self.events], ['load_selected', 'audit', 'submission', 'summary', 'report'])
        self.assertEqual(self.events[1], ('audit', 'best_guarded'))
        self.assertEqual(self.events[2], ('submission', 'best_guarded', True))
        self.assertEqual(self.events[-1], ('report', str(self.download), True))
        for name, data in before.items():
            self.assertEqual((run/name).read_bytes(), data, name)

    def test_incomplete_resume_trains_with_recorded_epochs_and_retained_delivery(self):
        run, cfg = self.saved_run()
        (run/'completed.json').unlink()
        before = (run/'config.json').read_bytes()
        self.launch(['--resume', str(run)])
        self.assertEqual(self.events[0], ('train', 7))
        self.configuration.assert_not_called()
        self.verify.assert_called_once_with(cfg)
        self.assertEqual((run/'config.json').read_bytes(), before)
        self.assertEqual(self.events[2], ('submission', 'best_guarded', True))
        self.assertEqual(self.events[-1][-1], True)

    def test_resume_epoch_override_is_rejected_before_locks_or_work(self):
        run, _ = self.saved_run()
        before = (run/'config.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'recorded training parameters|saved training parameters'):
            self.launch(['--resume', str(run), '--epochs', '3'])
        self.idle.assert_not_called()
        self.lock.assert_not_called()
        self.configuration.assert_not_called()
        self.train.assert_not_called()
        self.finish.assert_not_called()
        self.archive.assert_not_called()
        self.assertEqual((run/'config.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
