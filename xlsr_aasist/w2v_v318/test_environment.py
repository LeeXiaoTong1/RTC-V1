"""Startup reporting must collect independent failures and keep full diagnostics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
from .environment import run_checks,plans,check_versions,cuda_diagnostics
from .runtime import require_version


class TestEnvironment(unittest.TestCase):
    def test_cuda_diagnostics_preserve_driver_error(self):
        cuda=Mock();cuda.is_available.return_value=False;cuda.device_count.return_value=0
        cuda.init.side_effect=RuntimeError('CUDA driver version is insufficient')
        torch=SimpleNamespace(__version__='2.8.0+cu126',version=SimpleNamespace(cuda='12.6'),cuda=cuda)
        runner=Mock(return_value=subprocess.CompletedProcess([],0,'A100, 470.0, 40960 MiB',''))
        data=cuda_diagnostics(torch,runner)
        self.assertIn('driver version is insufficient',data['failure'])
        self.assertEqual(data['visible_device_count'],0)
        cuda.is_bf16_supported.assert_not_called()
        self.assertIn('A100',data['nvidia_smi']['stdout'])

    def test_cuda_diagnostics_distinguish_bf16_and_missing_smi(self):
        cuda=Mock();cuda.is_available.return_value=True;cuda.device_count.return_value=1
        cuda.get_device_name.return_value='fixture GPU';cuda.get_device_capability.return_value=(7,0)
        cuda.is_bf16_supported.return_value=False
        torch=SimpleNamespace(__version__='2.8.0+cu126',version=SimpleNamespace(cuda='12.6'),cuda=cuda)
        runner=Mock(side_effect=FileNotFoundError('nvidia-smi absent'))
        data=cuda_diagnostics(torch,runner)
        self.assertTrue(data['cuda_available']);self.assertIn('BF16 is unsupported',data['failure'])
        self.assertIn('nvidia-smi absent',data['nvidia_smi']['error'])
        cuda.is_bf16_supported.return_value=True
        self.assertNotIn('failure',cuda_diagnostics(torch,runner))

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
