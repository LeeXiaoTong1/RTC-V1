"""Regressions for official wheel metadata and genuine version mismatches."""
import unittest
from unittest.mock import patch
from .runtime import require_version


class TestRuntime(unittest.TestCase):
    def test_official_release_padding_and_local_build_tags(self):
        for actual in ('0.6','0.6.0','0.6.0.0','0.6+pt2.8.0.cu126'):
            with self.subTest(actual=actual),patch('w2v_v318.runtime.version',return_value=actual):
                self.assertEqual(require_version('fairseq2','0.6.0'),actual)
        self.assertEqual(require_version('torch','2.8.0','2.8.0+cu126'),'2.8.0+cu126')

    def test_wrong_release_and_unstable_builds_are_still_rejected(self):
        for actual in ('0.5.2','0.6.1','0.7','0.6rc1','0.6.dev0','0.6.post1','invalid'):
            with self.subTest(actual=actual),self.assertRaises(RuntimeError) as error:
                require_version('fairseq2','0.6.0',actual)
            self.assertIn('installed='+actual,str(error.exception))
            self.assertIn('required release=0.6.0',str(error.exception))

    def test_preflight_continues_past_equivalent_versions_and_rejects_wrong_native_release(self):
        # Stop at the SSL hub boundary: no GPU or native packages needed to test this guard.
        import sys
        from types import ModuleType
        from . import preflight
        actual={'fairseq2':'0.6','fairseq2n':'0.6','omnilingual-asr':'0.2.0',
            'webrtc-audio-processing':'0.1.3','torchaudio':'2.8.0+cu126'}
        modules={n:ModuleType(n) for n in ('omnilingual_asr','fairseq2','fairseq2.models','fairseq2.models.wav2vec2')}
        def reached_hub():raise LookupError('passed version guards')
        modules['fairseq2.models.wav2vec2'].get_wav2vec2_model_hub=reached_hub
        with patch.dict(sys.modules,modules),patch.object(sys,'argv',['preflight']),patch.object(preflight.torch,'__version__','2.8.0+cu126'),patch('w2v_v318.runtime.version',side_effect=lambda n:actual[n]):
            with self.assertRaisesRegex(LookupError,'passed version guards'):preflight.main()
            actual['fairseq2n']='0.7'
            with self.assertRaisesRegex(RuntimeError,'fairseq2n: installed=0.7'):preflight.main()


if __name__=='__main__':unittest.main()
