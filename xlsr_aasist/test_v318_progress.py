import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from v318_progress import Steps, progress, watch


class ViewerTests(unittest.TestCase):
    def test_partial_append_epoch_transition_and_eta(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'steps.jsonl'
            row = dict(epoch=1, step=1, loss=.4, compute_seconds=2, wait_seconds=1)
            payload = json.dumps(row).encode()
            p.write_bytes(payload[:20])
            r = Steps(); r.poll(p)
            self.assertIsNone(r.latest)
            with p.open('ab') as f: f.write(payload[20:]+b'\n')
            r.poll(p); r.poll(p)
            self.assertEqual(r.count, 1)
            self.assertIn('ETA 00:00:27', progress(r, 1, 10, 10))
            self.assertIn('waiting', progress(r, 2, 10, 10))
            row.update(epoch=2, step=1, compute_seconds=5)
            with p.open('ab') as f: f.write(json.dumps(row).encode()+b'\n')
            r.poll(p)
            self.assertEqual((r.count, r.seconds), (1, 6))

    def test_completed_job_preserves_dev_output_without_writes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'train.log'
            content = '[Train] V3.18 epoch 1/10 phase=head_warmup updates=2417\n[Dev] Weighted=96.7\n[Exit] V3.18 exit_code=0\n'
            p.write_text(content)
            Path(str(p)+'.exit').write_text('0')
            output = io.StringIO()
            with contextlib.redirect_stdout(output): watch(p)
            self.assertIn('[Dev] Weighted=96.7', output.getvalue())
            self.assertEqual(p.read_text(), content)
            self.assertEqual(len(list(Path(d).iterdir())), 2)


if __name__ == '__main__': unittest.main()
