"""Bootstrap contract tests; no real environment install or network activity."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import prepare_environment as env


class BootstrapTests(unittest.TestCase):
    def test_defaults_match_training_manifest(self):
        args = env.parser().parse_args([])
        self.assertEqual(Path(args.output_dir) / 'manifest.json', env.ROOT / 'assets/phonemes_v3186/manifest.json')
        self.assertEqual(args.model_cache, '/home/ubuntu/LXT/v318-runtime/hf-phonemes')

    def test_installer_never_resolves_protected_dependencies(self):
        command = env.pip_command('/data/private/bin/python', '/data/prep')
        self.assertIn('--no-deps', command)
        self.assertIn('--ignore-installed', command)
        self.assertIn('--only-binary=:all:', command)
        self.assertEqual(command[0], '/data/private/bin/python')
        names = {value.split('==')[0] for value in command if '==' in value}
        self.assertFalse(names.intersection(env.CORE))
        offline = env.pip_command('/data/private/bin/python', '/data/prep', True)
        self.assertIn('--no-index', offline)
        self.assertIn('--find-links', offline)

    def test_all_caches_and_temp_are_explicit(self):
        with patch.dict(os.environ, {'PIP_TARGET': '/wrong', 'PIP_USER': '1', 'PYTHONPATH': '/wrong'}):
            values = env.private_environment(Path('/data/prep'), Path('/data/models'))
        self.assertNotIn('PIP_TARGET', values)
        self.assertNotIn('PYTHONPATH', values)
        self.assertEqual(values['PIP_USER'], '0')
        self.assertEqual(values['PIP_CONFIG_FILE'], os.devnull)
        for key in ('TMPDIR', 'TMP', 'TEMP'): self.assertEqual(Path(values[key]), Path('/data/prep/tmp'))
        for key in ('HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            self.assertEqual(Path(values[key]), Path('/data/models'))

    def test_core_paths_and_venv_guard(self):
        packages = {name: {'version': 'x', 'location': '/original/site-packages'} for name in env.CORE}
        base = dict(packages=packages, core_imports={'torch': '/original/torch/__init__.py'})
        private = Path('/data/phoneme-private').resolve()
        candidate = dict(copy.deepcopy(base), prefix=str(private), base_prefix='/original')
        env.check_private_environment(base, candidate, private)
        changed = copy.deepcopy(candidate); changed['packages']['torch']['location'] = '/wrong'
        with self.assertRaises(ValueError): env.check_private_environment(base, changed, private)
        changed = copy.deepcopy(candidate); changed['prefix'] = '/original'
        with self.assertRaises(ValueError): env.check_private_environment(base, changed, private)

    def test_completed_cache_fastpath_has_no_probe_install_or_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = root / 'parent'; parent.mkdir()
            (parent / 'config.json').write_text(json.dumps({'version': '3.18.5'}))
            (parent / 'train_rows.json').write_text('[]')
            output = root / 'phones'; output.mkdir(); (output / 'manifest.json').write_text('{}')
            args = env.parser().parse_args(['--parent-run', str(parent), '--output-dir', str(output),
                '--environment-root', str(root / 'private'), '--cache-root', str(root / 'cache'),
                '--model-cache', str(root / 'models'), '--local-files-only', '--device', 'cpu'])
            with patch.object(env, 'probe', side_effect=AssertionError('No environment probe expected')), \
                    patch.object(env.venv, 'EnvBuilder', side_effect=AssertionError('No venv creation expected')), \
                    patch.object(env.subprocess, 'run') as run:
                env.execute(args)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertIn('w2v_v3186.prepare_phonemes', command)
            self.assertIn('--local-files-only', command)
            self.assertFalse((root / 'private').exists())

    def test_reject_non_v3185_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'config.json').write_text(json.dumps({'version': '3.19'}))
            with self.assertRaises(ValueError): env.resolve_parent(str(root))


if __name__ == '__main__': unittest.main()
