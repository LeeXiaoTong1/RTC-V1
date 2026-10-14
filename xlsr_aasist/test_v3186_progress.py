"""Terminal layout and read-only incremental progress, without model imports."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from v3186_progress import Dashboard,frame,watch


def history(count=6):
    groups={condition+'/'+language:dict(recall=[.94,.86],macro_f1=.91,auc=.97,eer=.08,
        matched_95=dict(real_recall=.83)) for condition in ('heldout','online','seen') for language in ('en','zh')}
    scenes={'scene'+str(i)+'/en':dict(next(iter(groups.values()))) for i in range(12)}
    return [dict(epoch=i,metrics=dict(clean_f1=.96,noisy_f1=.9,weighted_f1=.918),
        selection_metrics=dict(weighted_f1=.91),promoted=i in (1,3),dev=dict(groups=groups),
        train=dict(groups=groups),scenes=dict(groups=scenes),panel=dict(groups=groups)) for i in range(1,count+1)]


class ProgressTests(unittest.TestCase):
    def test_parent_replay_is_shown_as_initialization_not_training(self):
        state=Dashboard('unused')
        state.line('[Parent verification] full fixed Dev inference before the first update')
        self.assertIn('verifying parent on full Dev',state.progress())
        self.assertEqual(state.epoch,0)
        state.line('[Parent verified] epoch_4 full Weighted=95.6')
        self.assertIn('parent Dev verified',state.progress())

    def test_six_epochs_groups_and_both_links_fit_without_scrolling(self):
        state=Dashboard('unused');state.epoch=6;state.history=history();state.exit_code=0;state.stage('Complete')
        state.links=dict(submission_url='https://temp.sh/Ab123/submission.zip',
                         report_url='https://temp.sh/Cd456/w2v_v3186_20261013_123456_abcd_report.zip')
        value=frame(state,80,24)
        self.assertLessEqual(len(value.splitlines()),23)
        self.assertLessEqual(max(map(len,value.splitlines())),79)
        for i in range(1,7):self.assertIn(f'{i:>5}  96.00',value)
        for name in history()[0]['dev']['groups']:self.assertIn(name,value)
        for url in state.links.values():self.assertIn(url,value)
        self.assertIn('Select80',value);self.assertIn('R@95F',value)
        self.assertIn('  91.00  *',value);value.encode('ascii')

    def test_panels_pages_and_previous_epoch(self):
        state=Dashboard('unused');state.history=history()
        first=frame(state,80,24,'scenes',2,1);second=frame(state,80,24,'scenes',2,2)
        self.assertIn('Six scenes epoch 2',first);self.assertIn('Page 1/2',first)
        self.assertIn('Page 2/2',second);self.assertNotEqual(first,second)
        self.assertIn('Fixed Train probe epoch 6',frame(state,panel='train'))
        self.assertIn('Epoch 9 has not completed',frame(state,epoch=9))

    def test_small_terminal_and_long_urls_never_force_scrolling(self):
        state=Dashboard('unused');state.history=history();state.exit_code=0
        state.links=dict(submission_url='https://temp.sh/'+('a'*200)+'/submission.zip',
                         report_url='https://temp.sh/'+('b'*200)+'/report.zip')
        for width,height in ((40,12),(20,6),(10,3)):
            value=frame(state,width,height)
            self.assertLessEqual(len(value.splitlines()),height-1)
            self.assertLessEqual(max(map(len,value.splitlines())),width-1)

    def test_incremental_partial_lines_history_cache_and_exit_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();log=root/'job.log';steps=run/'steps.jsonl'
            log.write_text('V3186_RUN='+str(run)+'\n[Model] noisy banner\n[Train] V3.18.6 epoch 2/6 phase=joint updates=100')
            state=Dashboard(log);state.poll();self.assertEqual(state.epoch,0)
            with log.open('a') as out:out.write('\n')
            row=dict(epoch=2,step=1,compute_seconds=3.,wait_seconds=1.)
            steps.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,step=2)))
            state.poll();self.assertEqual(state.steps.latest['step'],1)
            self.assertIn('4.00s/step',state.progress());self.assertIn('00:06:36',state.progress())
            with steps.open('a') as out:out.write('\n')
            (run/'training_history.json').write_text(json.dumps(history(1)))
            state.poll();self.assertEqual(state.steps.latest['step'],2)
            # An interrupted write never removes previously committed metrics.
            (run/'training_history.json').write_text('[')
            state.poll();self.assertEqual(len(state.history),1)
            with log.open('a') as out:out.write('[Complete] training saved\n[Report] uploading report ZIP\n')
            state.poll();self.assertIsNone(state.exit_code);self.assertEqual(state.phase,'Uploading report')
            with log.open('a') as out:out.write('RuntimeError: report network unavailable')
            Path(str(log)+'.exit').write_text('1\n');state.poll()
            self.assertEqual(state.exit_code,1);self.assertIn('network unavailable',frame(state))
            self.assertNotIn('[Model]',frame(state));self.assertNotIn('Traceback',frame(state))

    def test_snapshot_and_interactive_view_never_modify_training_files(self):
        class Terminal(io.StringIO):
            def isatty(self):return True
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();log=root/'job.log'
            log.write_text('V3186_RUN='+str(run)+'\n[Train] V3.18.6 epoch 3/6 updates=100\n[Model] hidden\n')
            (run/'training_history.json').write_text(json.dumps(history(2)))
            (run/'config.json').write_text('{"epochs": 6}')
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            for stream in (io.StringIO(),Terminal()):
                with patch('v3186_progress.shutil.get_terminal_size',return_value=os.terminal_size((80,24))):
                    self.assertEqual(watch(log,once=True,stream=stream),0)
                self.assertEqual('\033[' in stream.getvalue(),stream.isatty())
                self.assertNotIn('hidden',stream.getvalue())
            self.assertEqual(before,{p:p.read_bytes() for p in root.rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main()
