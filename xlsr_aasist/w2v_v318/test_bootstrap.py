"""Regression guards for old-driver installation and safe environment repair."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from .bootstrap import active_jobs,select_profile,prepare_runtime,storage_status,require_storage,guard


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

    def test_storage_reports_system_and_data_filesystems_before_refusing(self):
        rows=[dict(role='project',path='/data/project',filesystem=2,free_GiB=98),
            dict(role='environment',path='/system/env',filesystem=1,free_GiB=5.78),
            dict(role='temporary',path='/tmp',filesystem=1,free_GiB=5.78)]
        with self.assertRaisesRegex(RuntimeError,'environment=/system/env.*temporary=/tmp'):
            require_storage(rows,12)
        require_storage([rows[0]],12)

    def test_prepare_runtime_checks_space_without_altering_existing_files(self):
        with tempfile.TemporaryDirectory() as d,patch('w2v_v318.bootstrap.host_guard'):
            root=Path(d)/'data disk'/'runtime';old=Path(d)/'old environment';old.mkdir()
            (old/'keep').write_text('preserve')
            with patch('w2v_v318.bootstrap.shutil.disk_usage',return_value=SimpleNamespace(free=5.78*2**30)):
                with self.assertRaisesRegex(RuntimeError,'20 GiB'):prepare_runtime(str(root))
                self.assertFalse(root.exists())
            with patch('w2v_v318.bootstrap.shutil.disk_usage',return_value=SimpleNamespace(free=98*2**30)):
                self.assertEqual(prepare_runtime(str(root)),root.resolve())
                self.assertTrue((root/'tmp').is_dir());self.assertTrue((root/'pkgs').is_dir())
                self.assertFalse((root/'envs'/'sdd-v318').exists())
                self.assertEqual((old/'keep').read_text(),'preserve')
                prefix=root/'envs'/'sdd-v318';prefix.mkdir();(prefix/'keep').write_text('partial')
                with self.assertRaisesRegex(RuntimeError,'not a complete Conda'):prepare_runtime(str(root))
                self.assertEqual((prefix/'keep').read_text(),'partial')

    def test_full_prefix_activation_checks_real_python_environment(self):
        import sys
        with tempfile.TemporaryDirectory() as d,patch('w2v_v318.bootstrap.host_guard'), \
             patch('w2v_v318.bootstrap.require_storage') as check,patch.object(sys,'version_info',(3,11,0)):
            prefix=Path(d)/'envs'/'sdd-v318';prefix.mkdir(parents=True)
            with patch.object(sys,'prefix',str(prefix)),patch.dict(os.environ,{'CONDA_PREFIX':str(prefix),
                    'CONDA_DEFAULT_ENV':str(prefix),'TMPDIR':d,'CONDA_PKGS_DIRS':d}):
                guard();check.assert_called_once()
                self.assertEqual({r['role'] for r in check.call_args.args[0]},
                    {'project','environment','temporary','conda_cache'})
                with patch.object(sys,'prefix',d),self.assertRaisesRegex(RuntimeError,'full prefix path'):guard()

    def test_runtime_root_shell_keeps_cache_and_reuses_environment(self):
        bash=shutil.which('bash')
        if os.name=='nt':
            p=Path('C:/Program Files/Git/bin/bash.exe');bash=str(p) if p.is_file() else None
        if not bash:self.skipTest('Bash unavailable')
        fixture=r'''
python() {
  printf 'python %s TMP=%s CACHE=%s PREFIX=%s\n' "$*" "${TMPDIR:-}" "${CONDA_PKGS_DIRS:-}" "${CONDA_PREFIX:-}" >> "$V318_TEST_LOG"
  case "$*" in
    *'bootstrap prepare-runtime'*) printf '%s\n' "$V318_TEST_ROOT" ;;
    *'bootstrap profile'*) printf 'cu118\n' ;;
  esac
  return 0
}
conda() {
  printf 'conda %s TMP=%s CACHE=%s\n' "$*" "${TMPDIR:-}" "${CONDA_PKGS_DIRS:-}" >> "$V318_TEST_LOG"
  if [[ "$1" == info ]]; then printf '%s\n' "$V318_TEST_BASE"; fi
  if [[ "$1" == create ]]; then
    "$V318_TEST_PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); (p/"conda-meta").mkdir(parents=True); (p/"bin").mkdir(); (p/"conda-meta"/"history").touch(); (p/"bin"/"python").touch()' "$3"
  fi
  if [[ "$1" == activate ]]; then export CONDA_PREFIX="$2" CONDA_DEFAULT_ENV="$2"; fi
  return 0
}
'''
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();runtime=root/'data disk'/'runtime';runtime.mkdir(parents=True)
            base=root/'base';hook=base/'etc'/'profile.d'/'conda.sh';hook.parent.mkdir(parents=True);hook.write_text('')
            for name in ('setup_w2v_v318.sh','repair_w2v_v318.sh'):
                (root/name).write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes().replace(b'\r\n',b'\n'))
            script=root/'fixture.sh';script.write_text(fixture,encoding='utf8');log=root/'commands.txt'
            env=dict(os.environ,BASH_ENV=script.as_posix(),V318_TEST_ROOT=runtime.as_posix(),
                V318_TEST_BASE=base.as_posix(),V318_TEST_LOG=log.as_posix(),CONDA_DEFAULT_ENV='sdd-v318',
                V318_TEST_PYTHON=Path(sys.executable).as_posix())
            for _ in range(2):
                r=subprocess.run([bash,'repair_w2v_v318.sh','--cuda','11.8','--runtime-root',runtime.as_posix()],
                    cwd=root,env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=30)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                self.assertIn('Activate this environment in your terminal before training',r.stdout)
            commands=log.read_text(encoding='utf8');self.assertEqual(commands.count('conda create'),1)
            self.assertIn('conda env config vars set --prefix '+runtime.as_posix()+'/envs/sdd-v318',commands)
            self.assertNotIn('--clone',commands);self.assertNotIn('uninstall',commands)
            for line in commands.splitlines():
                if line.startswith('conda create') or 'pip install' in line or 'bootstrap guard' in line:
                    self.assertIn('TMP='+runtime.as_posix()+'/tmp',line)
                    self.assertIn('CACHE='+runtime.as_posix()+'/pkgs',line)

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
                    cwd=root,env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=30)
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
