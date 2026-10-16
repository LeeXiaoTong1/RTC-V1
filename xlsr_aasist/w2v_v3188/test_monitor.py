"""Terminal history survives subsequent epochs and viewer restarts."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from .monitor import Dashboard, HistoryPrinter, watch


def entry(epoch):
    group=dict(recall=[.95,.85],macro_f1=.9,auc=.97,eer=.06,matched_95={'real_recall':.86})
    return dict(epoch=epoch,tag=f'epoch_{epoch}_step_1334',promoted=True,
        metrics=dict(clean_f1=.9,noisy_f1=.8,weighted_f1=.825),selection_metrics=dict(weighted_f1=.82),
        training=dict(loss=.3),dev={'groups':{'online/en':group,'online/zh':group}})


class MonitorTests(unittest.TestCase):
    def test_live_progress_keeps_all_epoch_tables_without_ansi_clear(self):
        output=io.StringIO();state=Dashboard(None)
        printer=HistoryPrinter(output,tty=True,width=160)
        state.progress_value=dict(phase='training',epoch=1,epochs=20,step=1,steps=1334,loss=.5,raw_ce=.4,pcl=.01,eta=300)
        printer.emit(state,now=0)
        state.history=[entry(1)];state.progress_value.update(epoch=2,step=1,loss=.4)
        printer.emit(state,now=1);printer.emit(state,now=2)
        state.history.append(entry(2));state.progress_value.update(epoch=3)
        printer.emit(state,now=3)
        text=output.getvalue()
        self.assertEqual(text.count('Full Dev epoch 1 (%)'),1)
        self.assertEqual(text.count('Full Dev epoch 2 (%)'),1)
        self.assertIn('loss=0.50000',text);self.assertIn('CE=0.40000',text);self.assertIn('PCL=0.01000',text)
        self.assertNotIn('\x1b',text)
        self.assertTrue(text.isascii())
        self.assertLess(text.index('Full Dev epoch 1'),text.index('Full Dev epoch 2'))

    def test_restarted_viewer_shows_every_completed_epoch_once(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);history=[entry(i) for i in range(1,21)]
            (run/'training_history.json').write_text(json.dumps(history))
            (run/'config.json').write_text(json.dumps({'epochs':20}))
            (run/'progress.json').write_text(json.dumps(dict(phase='complete',epoch=20,steps=1334,step=1334)))
            output=io.StringIO();watch(run=run,once=True,stream=output)
            text=output.getvalue()
            for i in range(1,21):self.assertEqual(text.count(f'Full Dev epoch {i} (%)'),1)
            self.assertNotIn('\x1b',text)

    def test_links_are_printed_once_and_not_truncated(self):
        output=io.StringIO();state=Dashboard(None);printer=HistoryPrinter(output,tty=True,width=40)
        state.links={'submission_url':'https://temp.sh/a/long_submission.zip','report_url':'https://temp.sh/b/long_report.zip'}
        printer.emit(state,now=0);printer.emit(state,now=1)
        text=output.getvalue()
        for url in state.links.values():self.assertEqual(text.count(url),1)

    def test_inherited_best_is_explicit_even_in_narrow_terminal(self):
        output=io.StringIO();state=Dashboard(None);printer=HistoryPrinter(output,tty=True,width=40)
        state.progress_value=dict(status='complete',best_tag='parent_v3187_epoch_5_step_1403')
        printer.emit(state,now=0);printer.emit(state,now=1)
        self.assertEqual(output.getvalue().count('BEST=parent_v3187_epoch_5_step_1403 ORIGIN=v3187_parent'),1)
        state.progress_value=dict(status='paused',best_tag='epoch_2_step_1403')
        printer.emit(state,now=2)
        self.assertIn('BEST=epoch_2_step_1403 ORIGIN=v3188_trained',output.getvalue())

    def test_log_partial_lines_and_progress_json_are_read_safely(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);log=run/'log.txt'
            log.write_text('V3188_RUN='+str(run)+'\n[Train] V3.18.8 epoch 1/20 updates=1334\n')
            state=Dashboard(log);state.poll()
            self.assertEqual(state.total,1334)
            (run/'progress.json').write_text(json.dumps(dict(epoch=1,epochs=20,step=12,steps=1334,loss=.3,raw_ce=.2)))
            state.poll();self.assertIn('12/1334',state.progress())
            (run/'progress.json').write_text('{')
            state.poll();self.assertIn('12/1334',state.progress())


if __name__=='__main__':unittest.main()
