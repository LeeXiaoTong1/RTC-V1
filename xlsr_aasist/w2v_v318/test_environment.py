"""Startup reporting must collect independent failures and keep full diagnostics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from .environment import run_checks,plans,check_versions
from .runtime import require_version


class TestEnvironment(unittest.TestCase):
    def test_report_collects_failures_without_hiding_later_success(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report.json'
            commands=[('one',[sys.executable,'-c','raise RuntimeError("first failure")']),
                ('two',[sys.executable,'-c','print("later stage executed")']),
                ('three',[sys.executable,'-c','raise ImportError("another dependency")'])]
            result=run_checks(commands,path)
            self.assertFalse(result['ok']);self.assertEqual([x['ok'] for x in result['checks']],[False,True,False])
            saved=json.loads(path.read_text(encoding='utf8'))
            self.assertIn('later stage executed',saved['checks'][1]['stdout'])
            self.assertIn('Traceback',saved['checks'][2]['stderr'])

    def test_timeout_is_reported_and_does_not_abort_remaining_stages(self):
        def runner(command,**kwargs):
            if command==['slow']:raise subprocess.TimeoutExpired(command,kwargs['timeout'])
            return subprocess.CompletedProcess(command,0,'success','')
        with tempfile.TemporaryDirectory() as d:
            result=run_checks([('slow',['slow']),('fast',['fast'])],Path(d)/'report.json',runner=runner,timeout=1)
            self.assertEqual([x['ok'] for x in result['checks']],[False,True])

    def test_cuda_build_pins_and_all_critical_stages(self):
        self.assertEqual(require_version('fairseq2n','0.6+cu126','0.6.0+cu126'),'0.6.0+cu126')
        for wrong in ('0.6','0.6+cpu','0.6+cu128'):
            with self.assertRaises(RuntimeError):require_version('fairseq2n','0.6+cu126',wrong)
        names={name for name,_ in plans()}
        self.assertTrue({'versions','scientific','native','cuda','omni','interface','pip-check','augment:webrtc','augment:ffmpeg'}<=names)

    def test_version_inventory_reports_multiple_bad_dependencies(self):
        def check(name,wanted):
            if name in ('pyarrow','pandas'):raise RuntimeError(name+' is incompatible')
            return wanted
        with patch('w2v_v318.runtime.require_version',side_effect=check),self.assertRaises(RuntimeError) as error:check_versions()
        self.assertIn('pyarrow',str(error.exception));self.assertIn('pandas',str(error.exception))


if __name__=='__main__':unittest.main()
