"""Preparation failures must preserve their cause and never authorize a bad runtime."""
from contextlib import redirect_stdout,ExitStack
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
from w2v_v31810.common import atomic_json,read_json
from . import prepare_teacher as prep


class EnvironmentTests(unittest.TestCase):
    def test_child_failure_is_streamed_saved_and_recoverable(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()) as output:
            root=Path(directory);log=root/'install.log'
            command=[sys.executable,'-c','import sys; print("build started"); print("ERROR: failed source build",file=sys.stderr); sys.exit(7)']
            with self.assertRaisesRegex(RuntimeError,'failed source build') as raised:
                prep.run_logged(command,dict(os.environ),log,cwd=root)
            self.assertIn('ERROR: failed source build',log.read_text())
            self.assertIn(str(log),str(raised.exception));self.assertIn('exit 7',str(raised.exception))
            self.assertIn('ERROR: failed source build',output.getvalue())
            prep.run_logged([sys.executable,'-c','print("recovered")'],dict(os.environ),log,cwd=root)
            self.assertIn('recovered',log.read_text())
            self.assertEqual(log.read_text().count('Attempt '),2)

    def test_installer_uses_hash_pinned_archive_and_explicit_build_tools(self):
        command=prep.espnet_command('/private/bin/python')
        self.assertIn('--no-build-isolation',command);self.assertIn('--no-deps',command)
        self.assertEqual(command[-1],'espnet @ '+prep.ESPNET_ARCHIVE+'#sha256='+prep.ESPNET_SHA256)
        self.assertNotIn('git+',command[-1])
        self.assertIn('pytest-runner==6.0.1',prep.BUILD_PINS)

    def test_provenance_accepts_old_git_and_verified_archive_rejects_other_sources(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack,redirect_stdout(io.StringIO()):
            root=Path(directory);origin={}
            distribution=SimpleNamespace(locate_file=lambda _: root/'site-packages',read_text=lambda _:json.dumps(origin))
            stack.enter_context(patch('importlib.metadata.distribution',return_value=distribution))
            versions={'espnet':'202506','torch':'test','numpy':'1.23.5','scikit-learn':'1.3.2'}
            stack.enter_context(patch('importlib.metadata.version',side_effect=versions.__getitem__))
            stack.enter_context(patch.object(sys,'prefix',str(root)))
            ssl=ModuleType('espnet2.tasks.ssl');ssl.SSLTask=object
            stack.enter_context(patch.dict(sys.modules,{'espnet2.tasks.ssl':ssl,'sklearn':ModuleType('sklearn')}))
            stack.enter_context(patch('w2v_v31813.espnet_runtime.load_ssl_task',return_value=object))
            code=prep.runtime_check(root,'test')
            origin.update(url=prep.ESPNET_ARCHIVE,archive_info={'hashes':{'sha256':prep.ESPNET_SHA256}})
            exec(code,{})
            origin['archive_info']['hashes']['sha256']='incorrect'
            with self.assertRaisesRegex(AssertionError,'pinned official revision'):exec(code,{})
            origin.clear();origin.update(url='https://github.com/espnet/espnet.git',vcs_info={'commit_id':prep.ESPNET_REVISION})
            exec(code,{})
            origin['url']='https://example.invalid/espnet.git'
            with self.assertRaisesRegex(AssertionError,'pinned official revision'):exec(code,{})

    def test_retry_reuses_owned_environment_and_never_stamps_failed_imports(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);private=root/'env';cache=root/'cache';private.mkdir()
            python=private/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            python.parent.mkdir(exist_ok=True);python.touch();(private/'pyvenv.cfg').write_text('owned')
            version=lambda name:{'torch':'2.6.0','torchaudio':'2.6.0'}[name]
            owner=dict(schema='v31813_xeus_env_v1',base_python=sys.executable,torch='2.6.0',espnet=prep.ESPNET_REVISION,pins=prep.PINS)
            atomic_json(private/'owner.json',owner)
            args=prep.parser().parse_args(['--environment-root',str(private),'--cache-root',str(cache),'--output',str(root/'out')])
            stack.enter_context(patch.object(prep,'resolve_parent',return_value=root/'parent'))
            stack.enter_context(patch.object(prep.sys,'version_info',(3,11,10)))
            stack.enter_context(patch.object(prep,'snapshot',return_value={'torch':'2.6.0'}))
            stack.enter_context(patch.object(prep.metadata,'version',side_effect=version))
            create=stack.enter_context(patch.object(prep.venv.EnvBuilder,'create'))
            launch=stack.enter_context(patch.object(prep.subprocess,'run'))
            logs=stack.enter_context(patch.object(prep,'run_logged',side_effect=[None,None,None,RuntimeError('import failed')]))
            with self.assertRaisesRegex(RuntimeError,'import failed'):prep.execute(args)
            self.assertFalse((private/'installed.json').exists());launch.assert_not_called();create.assert_not_called()
            logs.reset_mock();logs.side_effect=None
            prep.execute(args)
            self.assertEqual(read_json(private/'installed.json'),owner)
            self.assertEqual(logs.call_count,4);self.assertEqual(launch.call_count,1)
            logs.reset_mock();prep.execute(args)
            self.assertEqual(logs.call_count,1)  # successful installs only recheck imports
            self.assertEqual(read_json(private/'owner.json'),owner)

if __name__=='__main__':unittest.main()
