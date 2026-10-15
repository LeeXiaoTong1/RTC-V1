import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import prepare_environment as prep


class EnvironmentTests(unittest.TestCase):
    def test_defaults_reuse_runtime_model_and_metadata(self):
        args=prep.parser().parse_args(['--data-run','run','--local-files-only'])
        self.assertTrue(args.output_dir.endswith('phonemes_v3186'))
        self.assertTrue(args.environment_root.endswith('phoneme-env-v3186'))
        command=prep.preparation_command('/python',args,'/run')
        self.assertIn('w2v_v3187.prepare_phonemes',command)
        self.assertIn('--data-run',command);self.assertIn('--local-files-only',command)

    def test_complete_cache_does_not_probe_install_or_load_recognizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();(out/'manifest.json').write_text('{}')
            args=prep.parser().parse_args(['--data-run',str(root/'run'),'--output-dir',str(out),
                '--environment-root',str(root/'environment'),'--cache-root',str(root/'cache'),
                '--model-cache',str(root/'models'),'--local-files-only'])
            with patch.object(prep,'resolve_data_run',return_value=root/'run'),\
                    patch.object(prep,'probe',side_effect=AssertionError('No environment probe needed')),\
                    patch.object(prep.subprocess,'run') as run:
                prep.execute(args)
            command=run.call_args.args[0]
            self.assertIn('w2v_v3187.prepare_phonemes',command)
            self.assertNotIn('pip',command)
            self.assertFalse((root/'environment').exists())

    def test_runtime_isolation_and_offline_pip_policy(self):
        env=prep.private_environment('/data/cache','/data/models')
        self.assertNotIn('PYTHONPATH',env)
        self.assertIn('cache',env['TMPDIR'])
        command=prep.pip_command('/private/python','/data/cache',True)
        self.assertIn('--no-deps',command);self.assertIn('--no-index',command)
        self.assertFalse(any(item.startswith(('torch==','numpy==','fairseq')) for item in command))


if __name__=='__main__':unittest.main()
