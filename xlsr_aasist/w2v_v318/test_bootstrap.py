"""Regression guards for old-driver installation and safe environment repair."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest
from .bootstrap import active_jobs,select_profile


class TestBootstrap(unittest.TestCase):
    def test_r470_selects_cuda11_even_with_incompatible_torch_installed(self):
        for installed in ('','2.8.0+cu126','2.3.1+cu118'):
            self.assertEqual(select_profile('auto',installed,['470.82.01']),'cu118')
        self.assertEqual(select_profile('auto','',['550.54.15']),'cu126')

    def test_repair_preserves_downgrade_and_explicit_selection(self):
        for drivers in ([],['470.82.01'],['580.65.06']):
            self.assertEqual(select_profile('auto','2.6.0+cu118',drivers),'cu118')
        self.assertEqual(select_profile('11.8','2.8.0+cu126',[]),'cu118')
        self.assertEqual(select_profile('12.6','2.6.0+cu118',[]),'cu126')
        with self.assertRaisesRegex(RuntimeError,'select --cuda'):select_profile('auto','',[])
        with self.assertRaises(ValueError):select_profile('11.7','',[])

    def test_running_training_blocks_repair_without_stopping_it(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            commands={12:['python','-m','w2v_v318.bootstrap','guard'],
                13:['/env/bin/python3.11','-m','w2v_v317.workflow','--epochs','4'],
                14:['/env/bin/python','-m','w2v_v318.preflight','--weights'],
                15:['/env/bin/python','unrelated.py'],16:['bash','run_w2v_v318.sh']}
            for pid,args in commands.items():
                p=root/str(pid);p.mkdir();(p/'cmdline').write_bytes(b'\0'.join(a.encode() for a in args))
            uid=(root/'13').stat().st_uid
            jobs=active_jobs(root,current=12,uid=uid)
            self.assertEqual({x['pid'] for x in jobs},{13,14})
            self.assertTrue((root/'13'/'cmdline').exists())

    def test_repair_shell_gates_build_and_weights_on_prior_success(self):
        bash=shutil.which('bash')
        if os.name=='nt':
            candidate=Path('C:/Program Files/Git/bin/bash.exe')
            bash=str(candidate) if candidate.is_file() else None
        if not bash:self.skipTest('Bash unavailable')
        # Execute the actual entry scripts with package/build subprocesses
        # replaced. This checks ordering and failure propagation, not compilation.
        fixture=r'''
python() {
  printf '%s\n' "python $*" >> "$V318_TEST_LOG"
  case "$*" in
    *'bootstrap profile'*) printf 'cu118\n' ;;
    *'--stage cuda-torch'*) [[ "$V318_TEST_FAIL" != cuda ]] || return 42 ;;
    *'native_abi("cu118")'*) return 1 ;;
  esac
  return 0
}
bash() {
  if [[ "$1" == build_fairseq2_cu118_v318.sh ]]; then
    printf 'native-build\n' >> "$V318_TEST_LOG"
    [[ "$V318_TEST_FAIL" != build ]] || return 43
  else
    command bash "$@"
  fi
}
'''
        for failure,status in (('cuda',42),('build',43),('none',0)):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                for name in ('setup_w2v_v318.sh','repair_w2v_v318.sh'):
                    (root/name).write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes().replace(b'\r\n',b'\n'))
                script=root/'fixture.sh';script.write_text(fixture,encoding='utf8')
                log=root/'commands.txt'
                env=dict(os.environ,BASH_ENV=script.as_posix(),V318_TEST_LOG=log.as_posix(),
                    V318_TEST_FAIL=failure,CONDA_DEFAULT_ENV='sdd-v318')
                result=subprocess.run([bash,'repair_w2v_v318.sh','--cuda','11.8','--omni-size','3b'],
                    cwd=root,env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,status,result.stdout+result.stderr)
                commands=log.read_text(encoding='utf8')
                self.assertIn('torch==2.6.0+cu118 torchaudio==2.6.0+cu118',commands)
                self.assertNotIn('/whl/cu126',commands)
                if failure=='cuda':self.assertNotIn('native-build',commands)
                if failure!='none':
                    self.assertNotIn('w2v_v318.prepare',commands)
                    self.assertNotIn('V318_REPAIR_COMPLETE=True',result.stdout)
                else:
                    self.assertLess(commands.index('--stage cuda-torch'),commands.index('native-build'))
                    self.assertLess(commands.index('test_bootstrap'),commands.index('w2v_v318.prepare'))
                    self.assertIn('w2v_v318.preflight --omni-size 3b --weights',commands)
                    self.assertIn('V318_REPAIR_COMPLETE=True',result.stdout)


if __name__=='__main__':unittest.main()
