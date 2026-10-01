"""Console-only regression tests; no Torch or training process is imported."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v32_console import Display, Events, LogTail, format_record, safe_text, watch


def sample(tag='epoch_1', weighted=.954):
    return {'tag': tag, 'phase': 'joint', 'dev': {
        'clean_f1': .98, 'seen_f1': .94, 'heldout_f1': .95,
        'noisy_f1': .945, 'weighted_f1': weighted,
        'groups': {group: {'recall': [.99, .81 + i * .01]}
                   for i, group in enumerate(('offline/en', 'online/en', 'seen/en', 'heldout/en'))}},
        'decision': {'save': ['best_safe'], 'action': 'continue', 'warnings': ['guardrail_note']},
        'learning_rates': {'encoder': 5e-8, 'head': 2e-6}}


def write_record(root, tag):
    (root / (tag + '.json')).write_text(json.dumps(sample(tag)), encoding='utf-8')


class ConsoleTests(unittest.TestCase):
    def test_partial_lines_and_split_utf8_are_held_until_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'job.log'
            log.write_bytes(b'first\nV32_RUN=')
            tail = LogTail(log)
            self.assertEqual(tail.read(), ['first'])
            payload = '路径'.encode('utf-8')
            with log.open('ab') as stream:
                stream.write(payload[:2])
            self.assertEqual(tail.read(), [])
            with log.open('ab') as stream:
                stream.write(payload[2:] + b'\n')
            self.assertEqual(tail.read(), ['V32_RUN=路径'])
            self.assertEqual(tail.read(), [])

    def test_truncation_discards_unfinished_old_line(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'job.log'
            log.write_text('prior complete\nlong interrupted line', encoding='utf-8')
            tail = LogTail(log)
            self.assertEqual(tail.read(), ['prior complete'])
            log.write_text('new\n', encoding='utf-8')
            self.assertEqual(tail.read(), ['new'])

    def test_reopen_replays_committed_results_and_deduplicates_dev_log(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            tags = ['baseline', 'epoch_1_step_2417', 'epoch_1']
            for tag in tags:
                write_record(run, tag)
            (run / 'report.md').write_text('\n'.join('| ' + tag + ' | metrics |' for tag in tags))
            for _ in range(2):  # New viewer still displays all prior completed rows.
                events = Events()
                result = events.consume('V32_RUN=' + str(run))
                joined = '\n'.join(result)
                self.assertEqual(joined.count('[Dev]'), 3)
                self.assertIn('Starting checkpoint / baseline', joined)
                self.assertIn('intermediate Dev', joined)
                self.assertIn('epoch complete', joined)
                self.assertIn('Noisy=94.500', joined)
                self.assertIn('heldout/en', joined)
                self.assertIn('promoted=True', joined)
                self.assertEqual(events.consume('Dev epoch_1 Clean=98 Seen=94 Heldout=95 Weighted=95.4 promoted=True action=continue'), [])
                for group in ('offline/en', 'online/en', 'seen/en', 'heldout/en'):
                    self.assertEqual(events.consume(group + ' recall [fake,real]=[0.99,0.80]'), [])
                self.assertEqual(events.consume('Selection warnings: guardrail_note'), [])
                self.assertEqual(events.report(force=True), [])

    def test_pending_report_json_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'report.md').write_text('| epoch_1 | metrics |')
            events = Events(run)
            self.assertEqual(events.report(now=1), [])
            (run / 'epoch_1.json').write_text('{unfinished')
            self.assertEqual(events.report(now=4), [])
            write_record(run, 'epoch_1')
            self.assertTrue(events.report(now=7))
            self.assertEqual(events.report(now=10), [])

    def test_legacy_log_fallback_collects_all_recalls_and_multiple_results(self):
        events = Events()
        result = []
        for tag in ('epoch_1_step_2417', 'epoch_1'):
            result += events.consume(f'Dev {tag} Clean=98 Seen=94 Heldout=96 Weighted=95.9 promoted=False action=continue')
            for group in ('offline/en', 'online/en', 'seen/en', 'heldout/en'):
                result += events.consume(group + ' recall [fake,real]=[0.99,0.80]')
        joined = '\n'.join(result)
        self.assertEqual(joined.count('[Dev]'), 2)
        self.assertEqual(joined.count('Noisy=95.000'), 2)
        self.assertEqual(joined.count('real=80.000%'), 8)

    def test_progress_spam_suppressed_and_phase_changes_retained(self):
        events = Events()
        self.assertEqual(events.consume('STEP 100/4833 LOSS=0.001 grad=0.3'), [])
        self.assertEqual(events.consume('Dev clean: 200/3000 elapsed=1.0m ETA=14.0m'), [])
        self.assertEqual(events.consume('baseline Dev clean: 200/3000 elapsed=1.0m ETA=14.0m'), [])
        self.assertEqual(events.phase('train epoch 1'), ['\n[Phase] train epoch 1'])
        self.assertEqual(events.phase('train epoch 1'), [])
        self.assertEqual(events.phase('Dev clean'), ['\n[Phase] Dev clean'])
        self.assertEqual(events.phase('train epoch 1'), ['\n[Phase] train epoch 1'])

    def test_structured_events_use_saved_json_and_reject_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_record(run, 'epoch_1')
            (run / 'report.md').write_text('| epoch_1 | metrics |')
            events = Events(run)
            event = lambda value: events.consume('V32_EVENT=' + json.dumps(value))
            self.assertIn('[Phase]', '\n'.join(event({'kind': 'phase', 'label': 'validating'})))
            self.assertIn('[Dev]', '\n'.join(event({'kind': 'validation', 'tag': 'epoch_1'})))
            self.assertEqual(event({'kind': 'validation', 'tag': 'epoch_1'}), [])
            self.assertEqual(event({'kind': 'validation', 'tag': '../outside'}), [])
            self.assertEqual(events.consume('V32_EVENT={partial'), [])

    def test_uncommitted_epoch_is_hidden_until_report_commit_and_emitted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_record(run, 'baseline')
            report = run / 'report.md'
            report.write_text('| baseline | metrics |')
            events = Events(run)
            baseline = events.report(force=True)
            self.assertEqual('\n'.join(baseline).count('[Dev]'), 1)
            write_record(run, 'epoch_1')  # Evaluation done; checkpoint saving not done.
            self.assertEqual(events.consume('Dev epoch_1 Clean=98 Seen=94 Heldout=95 Weighted=95.4 promoted=True action=continue'), [])
            for group in ('offline/en', 'online/en', 'seen/en', 'heldout/en'):
                self.assertEqual(events.consume(group + ' recall [fake,real]=[0.99,0.81]'), [])
            self.assertEqual(events.report(force=True), [])
            report.write_text('| baseline | metrics |\n| epoch_1 | metrics |')
            event = 'V32_EVENT=' + json.dumps({'kind': 'validation', 'tag': 'epoch_1'})
            committed = events.consume(event)
            self.assertEqual('\n'.join(committed).count('[Dev]'), 1)
            self.assertIn('epoch_1 (epoch complete)', '\n'.join(committed))
            self.assertEqual(events.consume(event), [])
            self.assertEqual(events.report(force=True), [])

    def test_new_result_does_not_erase_previous_results_or_repeat_same_phase(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        events = Events()
        terminal = Terminal()
        display = Display(terminal)
        phase = 'V3.2 joint epoch 1 steps 1-2417'
        display.messages(events.phase(phase))
        display.messages(events.record(sample('baseline')))
        for current in (1, 2, 3):
            display.messages(events.phase(phase))
            display.progress({'label': phase, 'current': current, 'total': 2417, 'elapsed': current})
        display.messages(events.record(sample('epoch_1_step_2417')))
        display.progress({'label': phase, 'current': 2417, 'total': 2417, 'elapsed': 2417})
        text = terminal.getvalue()
        self.assertEqual(text.count('[Phase]'), 1)
        self.assertEqual(text.count('[Dev]'), 2)
        self.assertLess(text.index('Starting checkpoint / baseline'), text.index('epoch_1_step_2417 (intermediate Dev)'))
        self.assertNotIn('\x1b[2J', text)
        self.assertNotIn('\x1b[H', text)

    def test_log_control_sequences_are_data_and_cannot_clear_screen(self):
        value = '\x1b[2J\x1b[Hlost\x1b]2;title\x07\r\x00\x9btext'
        clean = safe_text(value)
        self.assertNotIn('\x1b', clean)
        self.assertNotIn('\x07', clean)
        self.assertNotIn('\r', clean)
        self.assertNotIn('\x9b', clean)
        output = io.StringIO()
        Display(output).messages([value])
        self.assertEqual(output.getvalue().count('\n'), 1)

    def test_once_is_read_only_and_keeps_all_completed_validation_output(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_record(run, 'baseline')
            write_record(run, 'epoch_1')
            (run / 'report.md').write_text('| baseline | x |\n| epoch_1 | x |')
            log, state = run / 'outer.log', run / 'state.json'
            log.write_text('V32_RUN=' + str(run) + '\nSTEP 1/4833 LOSS=0.1\nTEMP_DOWNLOAD_URL=https://temp.sh/result.zip\n')
            state.write_text(json.dumps({'label': 'V3.2 joint epoch 2', 'current': 20, 'total': 2417,
                                         'elapsed': 40., 'status': 'running'}))
            before = {p.name: p.read_bytes() for p in run.iterdir()}
            output = io.StringIO()
            with patch('v32_console.time.sleep', side_effect=AssertionError('must not wait')):
                watch(log, state, once=True, stream=output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in run.iterdir()})
            text = output.getvalue()
            self.assertEqual(text.count('[Dev]'), 2)
            self.assertIn('TEMP_DOWNLOAD_URL=https://temp.sh/result.zip', text)
            self.assertIn('20/2417', text)
            self.assertNotIn('STEP 1/4833', text)

    def test_tty_interrupt_retains_results_and_never_sends_training_signal(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            log, state = run / 'outer.log', run / 'state.json'
            log.write_text('previous result\n')
            state.write_text(json.dumps({'label': 'training', 'current': 1, 'total': 10, 'elapsed': 1.}))
            output = Terminal()
            with patch('v32_console.time.sleep', side_effect=KeyboardInterrupt), patch('os.kill', side_effect=AssertionError('no signals')):
                watch(log, state, stream=output)
            text = output.getvalue()
            self.assertIn('previous result\n', text)
            self.assertIn('no signal was sent', text)
            self.assertNotIn('\x1b[2J', text)

    def test_missing_fields_are_unknown_and_no_full_scan_is_needed(self):
        text = '\n'.join(format_record({'tag': 'baseline', 'dev': {}, 'decision': {}}))
        self.assertIn('Noisy=n/a', text)
        self.assertIn('real=n/a%', text)


if __name__ == '__main__':
    unittest.main()
