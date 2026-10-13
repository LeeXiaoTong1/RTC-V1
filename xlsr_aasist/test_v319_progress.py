"""V3.19 compact metrics and artifact URLs fit terminals without scrollback."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from v319_progress import Dashboard,frame,watch
import v319_detach


def history(count=5):
    groups={condition+'/'+language:dict(recall=[.94,.86],macro_f1=.91,auc=.97,eer=.08)
            for condition in ('heldout','online','seen') for language in ('en','zh')}
    return [dict(tag=f'epoch_{i}_step_100',metrics=dict(clean_f1=.96,noisy_f1=.9,weighted_f1=.918,
            groups=groups,matched={g:dict(real_recall=.83) for g in groups}),
            promoted=['best_weighted'] if i in (1,3) else [],
            panel=dict(families={'scene'+str(j):dict(next(iter(groups.values()))) for j in range(14)}))
            for i in range(1,count+1)]


class ProgressTests(unittest.TestCase):
    def test_actual_inference_logs_replace_stale_training_progress(self):
        state=Dashboard('unused')
        state.line('[Train] V3.19 epoch 2/5 updates=100')
        state.line('  STEP 100/100 TOTAL=.4 compute_s/update=2.00 wait_s/update=.10')
        self.assertEqual(state.phase,'Training')
        state.line('V3.19 epoch_2_step_100 fixed Dev: 0/250')
        self.assertIn('Evaluation: fixed Dev | 0/250',state.progress())
        self.assertNotIn('100/100',state.progress())
        state.line('V3.19 epoch_2_step_100 fixed Dev: 200/250 elapsed=4.0m ETA=1.0m')
        self.assertIn('ETA 00:01:00',state.progress())
        state.line('V3.19 epoch_2_step_100 selection panel: 0/24')
        self.assertIn('scene selection | 0/24',state.progress())
        state.line('[Train] V3.19 epoch 3/5 updates=100')
        self.assertEqual(state.phase,'Training');self.assertIsNone(state.phase_counts)
        state.line('V3.19 parent fixed Dev: 1/250 elapsed=0.1m ETA=24.9m')
        self.assertIn('Initial evaluation: fixed Dev',state.progress())
        state.line('V319_SELECTED=best_weighted: epoch_3_step_100')
        self.assertIn('Training complete',state.progress())
        state.line('V3.19 submission inference: 12/24 elapsed=2.0m ETA=2.0m')
        self.assertIn('Submission inference | 12/24 50.0%',state.progress())

    def test_detach_reports_actual_child_and_keeps_viewer_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);log=root/'job.log';calls=[]
            def spawn(command,**kwargs):
                calls.append((command,kwargs))
                return SimpleNamespace(pid=54321)
            with patch.object(v319_detach,'os',SimpleNamespace(name='posix')):
                self.assertEqual(v319_detach.launch(log,['--epochs','5'],root,popen=spawn),54321)
            command,options=calls[0]
            self.assertEqual(command,['bash',str(root/'v319_supervise.sh'),str(log),'--epochs','5'])
            self.assertTrue(options['start_new_session'])
            self.assertTrue(options['close_fds'])
            self.assertEqual(options['cwd'],root)
            with patch.object(v319_detach,'os',SimpleNamespace(name='posix')):
                with self.assertRaises(OSError):
                    v319_detach.launch(log,[],root,popen=lambda *a,**k: (_ for _ in ()).throw(OSError('spawn failed')))

    def test_five_epochs_groups_both_links_fit_and_best_not_last(self):
        state=Dashboard('unused');state.epoch=5;state.history=history();state.exit_code=0;state.stage('Complete')
        state.links=dict(submission_url='https://temp.sh/Ab123/submission.zip',report_url='https://temp.sh/Cd456/report.zip')
        value=frame(state,80,24)
        self.assertLessEqual(len(value.splitlines()),23)
        self.assertLessEqual(max(map(len,value.splitlines())),79)
        for i in range(1,6):self.assertIn(f'{i:>5}  96.00',value)
        for name in history()[0]['metrics']['groups']:self.assertIn(name,value)
        for url in state.links.values():self.assertIn(url,value)
        self.assertIn('BestW',value);self.assertIn('R@99F',value)
        self.assertIn('    3  96.00  90.00  91.80   *',value)
        self.assertNotIn('    5  96.00  90.00  91.80   *',value)
        value.encode('ascii')

    def test_parent_best_and_small_terminal_do_not_claim_last_is_best(self):
        state=Dashboard('unused');state.history=history();state.selections={'best_weighted':'starting_parent'}
        self.assertIn('BestW: retained starting parent',frame(state))
        state.links=dict(submission_url='https://temp.sh/'+('a'*200)+'/submission.zip',
                         report_url='https://temp.sh/'+('b'*200)+'/report.zip')
        for width,height in ((40,12),(20,6),(10,3)):
            value=frame(state,width,height)
            self.assertLessEqual(len(value.splitlines()),height-1)
            self.assertLessEqual(max(map(len,value.splitlines())),width-1)

    def test_panel_previous_epoch_and_page(self):
        state=Dashboard('unused');state.history=history()
        first=frame(state,80,24,'rtc',2,1);second=frame(state,80,24,'rtc',2,2)
        self.assertIn('Held-out RTC / scenes epoch 2',first)
        self.assertIn('Page 1/',first);self.assertIn('Page 2/',second)
        self.assertIn('Epoch 9 has not completed',frame(state,epoch=9))

    def test_partial_lines_cached_history_exit_and_no_file_mutation(self):
        class Terminal(io.StringIO):
            def isatty(self):return True
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();log=root/'train.log'
            log.write_text('V319_RUN='+str(run)+'\n[Train] V3.19 epoch 2/5 updates=100')
            state=Dashboard(log);state.poll();self.assertEqual(state.epoch,0)
            with log.open('a') as stream:stream.write('\n[Model] hidden\n')
            (run/'steps.jsonl').write_text(json.dumps(dict(epoch=2,step=2,compute_seconds=3.,wait_seconds=1.))+'\n')
            (run/'training_history.json').write_text(json.dumps(history(1)))
            state.poll();self.assertEqual(state.epoch,2);self.assertIn('4.00s/step',state.progress())
            (run/'training_history.json').write_text('[');state.poll();self.assertEqual(len(state.history),1)
            (run/'training_history.json').write_text(json.dumps(history(1)))
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            for stream in (io.StringIO(),Terminal()):
                with patch('v319_progress.shutil.get_terminal_size',return_value=os.terminal_size((80,24))):
                    watch(log,once=True,stream=stream)
                self.assertEqual('\033[' in stream.getvalue(),stream.isatty())
                self.assertNotIn('[Model]',stream.getvalue())
            self.assertEqual(before,{p:p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with log.open('a') as stream:stream.write('RuntimeError: upload offline')
            Path(str(log)+'.exit').write_text('1\n');state.poll()
            self.assertIn('upload offline',frame(state))


if __name__=='__main__':unittest.main()
