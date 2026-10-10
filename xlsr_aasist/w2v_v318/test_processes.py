"""Regression tests for native process failures and spawned data loading."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import Dataset

from rtc_noisy.simulator import LocalRTC
from .rtc_process import IsolatedRTC
from .data import loader,close
from .runtime import bundled_ffmpeg


RAW=['-f','f32le','-ar','16000','-ac','1','-i','pipe:0']
ENCODE=[*RAW,'-c:a','libopus','-b:a','24000','-application','voip',
        '-frame_duration','20','-vbr','on','-f','ogg','pipe:1']
DECODE=['-f','ogg','-i','pipe:0','-ar','16000','-ac','1','-f','f32le','pipe:1']


class NativeData(Dataset):
    def __len__(self):return 4
    def __getitem__(self,i):
        rtc=IsolatedRTC(bundled_ffmpeg())
        x=(.1*np.sin(np.arange(16000+160*i)*.03)).astype('<f4')
        y=rtc._run(DECODE,rtc._run(ENCODE,x.tobytes()))
        return i,y


class ExitData(Dataset):
    def __len__(self):return 1
    def __getitem__(self,i):os._exit(7)


class Tests(unittest.TestCase):
    def rtc(self):
        rtc=object.__new__(IsolatedRTC);rtc.ffmpeg='fixture-ffmpeg';return rtc

    def test_empty_stderr_has_exitcode_and_arguments(self):
        with patch('w2v_v318.rtc_process.subprocess.run',return_value=SimpleNamespace(returncode=255,stderr=b'',stdout=b'')):
            with self.assertRaisesRegex(RuntimeError,'returncode=255.*libopus.*stderr=') as caught:
                self.rtc()._run(ENCODE,b'123')
        self.assertIn('<empty>',str(caught.exception))

    def test_posix_signal_is_identified(self):
        with patch('w2v_v318.rtc_process.os.name','posix'),patch('w2v_v318.rtc_process.subprocess.run',
                return_value=SimpleNamespace(returncode=-2,stderr=b'',stdout=b'')):
            with self.assertRaisesRegex(RuntimeError,'signal=SIGINT'):
                self.rtc()._run(ENCODE,b'123')

    def test_timeout_is_identified(self):
        with patch('w2v_v318.rtc_process.subprocess.run',side_effect=subprocess.TimeoutExpired(['ffmpeg'],120,stderr=b'timeout detail')):
            with self.assertRaisesRegex(RuntimeError,'timeout_seconds=120.*timeout detail'):
                self.rtc()._run(ENCODE,b'123')

    def test_launch_failure_is_identified(self):
        with patch('w2v_v318.rtc_process.subprocess.run',side_effect=OSError('fixture unavailable')):
            with self.assertRaisesRegex(RuntimeError,'launch_error=.*fixture unavailable'):
                self.rtc()._run(ENCODE,b'123')

    def test_success_keeps_payload_and_isolates_posix_session(self):
        with patch('w2v_v318.rtc_process.os.name','posix'),patch('w2v_v318.rtc_process.subprocess.run',
                return_value=SimpleNamespace(returncode=0,stderr=b'',stdout=b'unchanged')) as run:
            self.assertEqual(self.rtc()._run(ENCODE,b'123'),b'unchanged')
            self.assertTrue(run.call_args.kwargs['start_new_session'])
            self.assertEqual(run.call_args.kwargs['input'],b'123')

    def test_real_ffmpeg_output_unchanged(self):
        old=LocalRTC(bundled_ffmpeg());new=IsolatedRTC(bundled_ffmpeg())
        wave=(.1*np.sin(np.arange(16321)*.03)).astype('<f4').tobytes()
        # Container headers can vary; compare decoded PCM bit for bit.
        self.assertEqual(old._run(DECODE,old._run(ENCODE,wave)),new._run(DECODE,new._run(ENCODE,wave)))

    def test_spawned_native_codec_matches_serial(self):
        def collect(workers):
            batches=loader(NativeData(),dict(workers=workers,seed=31801),batch_size=2)
            try:return [r for batch in batches for r in batch]
            finally:close(batches)
        self.assertEqual(collect(0),collect(2))

    def test_abrupt_worker_exit_is_reported(self):
        batches=loader(ExitData(),dict(workers=1,seed=31801),batch_size=1)
        output=io.StringIO()
        try:
            with contextlib.redirect_stdout(output),self.assertRaises(RuntimeError):
                list(batches)
        finally:close(batches)
        report=json.loads(output.getvalue().split('V318_DATA_FAILURE=',1)[1])
        self.assertEqual(report['workers'][0]['exitcode'],7)
        self.assertFalse(report['workers'][0]['alive'])

    def test_real_data_check_and_bounds(self):
        from check_v318_workers import check
        from .test_core import make_fixture,NativeBoundary
        from .common import atomic_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,train,dev=make_fixture(root)
            atomic_json(root/'config.json',cfg);atomic_json(root/'train_rows.json',train)
            with patch('w2v_v318.data.Engines',NativeBoundary),contextlib.redirect_stdout(io.StringIO()):
                report=check(root,workers=0,steps=2)
            self.assertTrue(report['passed'])
            self.assertEqual(report['views'],24)
            with self.assertRaisesRegex(ValueError,'exceed epoch length'):
                check(root,workers=0,start_step=999,steps=1)


if __name__=='__main__':unittest.main()
