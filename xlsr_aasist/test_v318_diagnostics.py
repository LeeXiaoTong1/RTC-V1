import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from v318_diagnostics import StepLog, blocks, chart, exclusive_writer, follow, matching_log


def step(update, epoch=1, local=None, loss=.4, views=48, ce=24):
    return dict(update=update, epoch=epoch, step=local or update, loss=loss, views=views,
                raw_ce_sum=ce, grad_norm=.5, max_example_ce=2., risk_coefficient=0.,
                compute_seconds=2., wait_seconds=.02, unique_original_sources=32,
                lrs={'aasist':1e-4, 'lora':0., 'evidence':0.},
                cells={'online/en/1':dict(objective=.05, raw_ce_sum=2., views=4, correct=3)})


def epoch_fixture():
    group = dict(class_counts=[512,512], recall=[.9,.8], class_ce=[.2,.3],
                 macro_f1=.85, ap=.95, auc=.94, eer=.1, balanced_ce=.25)
    return dict(epoch=1, tag='epoch_1_step_2', phase='head_warmup', promoted=True,
                metrics=dict(clean_f1=.9,noisy_f1=.8,weighted_f1=.83),
                selection_metrics=dict(weighted_f1=.82), training=dict(loss=.4,raw_ce=.5),
                train={'groups':{'online/en':group}}, dev={'groups':{'online/en':group}},
                panel={'groups':{}}, fit=dict(text='Fixture: not server results'))


def make_run(root):
    run = root/'run'
    run.mkdir()
    (run/'config.json').write_text(json.dumps(dict(version='3.18',warm_epochs=2,epochs=10)))
    (run/'training_history.json').write_text(json.dumps([epoch_fixture()]))
    (run/'steps.jsonl').write_text(''.join(json.dumps(step(i))+'\n' for i in (1,2,3)))
    # A diagnostic tool must never deserialize or rewrite this file.
    (run/'last.pt').write_bytes(b'not-a-pickle-fixture')
    return run


class DiagnosticTests(unittest.TestCase):
    def test_partial_log_retry_and_truncate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'steps.jsonl'
            payload = json.dumps(step(3)).encode()
            path.write_bytes(b''.join((json.dumps(step(i))+'\n').encode() for i in (1,2))+payload[:12])
            reader = StepLog()
            reader.poll(path)
            self.assertEqual(reader.last, 2)
            self.assertFalse(reader.poll(path))
            with path.open('ab') as f:
                f.write(payload[12:]+b'\n')
            reader.poll(path)
            self.assertEqual(reader.last, 3)
            with path.open('ab') as f:
                f.write((json.dumps(step(2,loss=.123))+'\n').encode())
            reader.poll(path)
            self.assertEqual([r['update'] for r in reader.values()], [1,2])
            self.assertEqual(reader.values()[-1]['loss'], .123)
            path.write_text(json.dumps(step(1,loss=.789))+'\n')
            reader.poll(path)
            self.assertEqual(len(reader.values()), 1)
            self.assertEqual(reader.values()[0]['loss'], .789)

    def test_ce_uses_view_weights_and_epoch_blocks_do_not_mix(self):
        a, b = step(1,views=2,ce=2), step(2,views=8,ce=16)
        c = step(3,epoch=2,local=1,views=2,ce=6)
        result = blocks([a,b,c])
        self.assertAlmostEqual(result[0]['raw_ce'], 1.8)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]['raw_ce'], 3.)
        self.assertEqual(result[0]['cells']['online/en/1'], .5)

    def test_once_preserves_sources_and_exports_uncommitted_and_class_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = make_run(root)
            before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
            follow(run, 5, root=root)
            out = run/'diagnostics'/'live'
            report = json.loads((out/'observations.json').read_text(encoding='utf-8'))
            self.assertEqual((report['committed_update'],report['logged_update']), (2,3))
            with (out/'loss_steps.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r['checkpoint_committed'] for r in rows], ['True','True','False'])
            self.assertEqual(float(rows[-1]['lora_lr']), 0.)
            with (out/'groups.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['real_ce'], '0.3')
            self.assertIn('class CE', (out/'curves.html').read_text(encoding='utf-8'))
            self.assertEqual(before, {name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in before})
            self.assertEqual(list(out.glob('*.tmp')), [])

    def test_watch_drains_final_history_and_ignores_stale_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = make_run(root)
            (run/'training_history.json').write_text('[]')
            (run/'failure.log').write_text('old failed attempt; resumed')
            def finish(_):
                (run/'training_history.json').write_text(json.dumps([epoch_fixture()]))
                (run/'completed.json').write_text(json.dumps(dict(status='complete')))
            with patch('v318_diagnostics.time.sleep', side_effect=finish) as sleep:
                follow(run, 5, watch=True, root=root)
            sleep.assert_called_once()
            report = json.loads((run/'diagnostics/live/observations.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['completed_epochs'], 1)

    def test_job_exit_marker_is_only_used_for_matching_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = make_run(root)
            (root/'exp').mkdir()
            log = root/'train.log'
            (root/'exp/.latest_v318_log').write_text(str(log))
            log.write_text('V318_RUN='+str(root/'unrelated')+'\n')
            Path(str(log)+'.exit').write_text('1\n')
            self.assertIsNone(matching_log(root,run))
            log.write_text('V318_RUN='+str(run)+'\n')
            with patch('v318_diagnostics.time.sleep', side_effect=AssertionError('must not wait')):
                follow(run, 5, watch=True, root=root)
            report = json.loads((run/'diagnostics/live/observations.json').read_text())
            self.assertEqual(report['status'], 'training_exit_1')

    def test_cli_uses_only_stdlib_and_never_loads_torch(self):
        with tempfile.TemporaryDirectory() as d:
            run = make_run(Path(d))
            source = Path(__file__).with_name('v318_diagnostics.py')
            # -S removes site packages, so an accidental torch/numpy import fails.
            result = subprocess.run([sys.executable,'-S',str(source),'--run',str(run)],
                                    capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('DIAGNOSTICS=',result.stdout)

    def test_plot_boundaries_missing_values_and_html_escaping(self):
        for value in (0.,100.):
            rendered = chart('<x>', [('a<b',[(1,value),(2,value),(3,None)])],percent=True)
            self.assertIn('&lt;x&gt;',rendered)
            self.assertNotIn('nan',rendered.lower())
            self.assertNotIn('inf',rendered.lower())
        self.assertIn('Waiting',chart('zero',[('lora',[(1,0.)])],log=True))

    def test_writer_lock_rejects_second_process_and_releases(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            code = ('from pathlib import Path; from v318_diagnostics import exclusive_writer; '
                    'p=Path(__import__("sys").argv[1]); '
                    '\nwith exclusive_writer(p): pass\n')
            with exclusive_writer(out):
                result = subprocess.run([sys.executable,'-c',code,str(out)],cwd=Path(__file__).parent,
                                        capture_output=True,text=True,timeout=20)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('already owns',result.stderr)
            with exclusive_writer(out):
                pass


if __name__ == '__main__':
    unittest.main()
