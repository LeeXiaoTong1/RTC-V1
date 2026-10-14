"""Network/cache failure regressions; no real model download or GPU required."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import phoneme_download as download


def fixtures():
    values={
        'preprocessor_config.json':dict(sampling_rate=16000,feature_extractor_type='Wav2Vec2FeatureExtractor',
                                       feature_size=1,do_normalize=True),
        'config.json':dict(model_type='wav2vec2',hidden_size=1024,num_hidden_layers=24,
                           num_attention_heads=16,vocab_size=392,pad_token_id=0,
                           conv_stride=[5,2,2,2,2,2,2],conv_kernel=[10,3,3,3,3,2,2]),
        'tokenizer_config.json':dict(tokenizer_class='Wav2Vec2PhonemeCTCTokenizer'),
        'special_tokens_map.json':dict(pad_token='<pad>',unk_token='<unk>',bos_token='<s>',eos_token='</s>'),
        'vocab.json':dict({'<pad>':0},**{'phone'+str(i):i for i in range(1,392)})
    }
    result={name:json.dumps(value).encode() for name,value in values.items()}
    result['pytorch_model.bin']=b'test weight bytes, never deserialized'
    return result


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.payloads=fixtures();self.calls=[]
        self.addCleanup(patch.stopall)
        patch.object(download,'WEIGHT_SHA256',hashlib.sha256(self.payloads['pytorch_model.bin']).hexdigest()).start()
        self.stdout=io.StringIO();self.output=redirect_stdout(self.stdout);self.output.__enter__()
        self.addCleanup(self.output.__exit__,None,None,None)

    def fetch(self,**kwargs):
        self.calls.append(kwargs)
        path=download.snapshot_path(kwargs['cache_dir'])/kwargs['filename']
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(self.payloads[kwargs['filename']]);return str(path)

    def complete(self):
        for name,data in self.payloads.items():
            path=download.snapshot_path(self.root)/name
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)

    def test_first_endpoint_failure_falls_back_and_reuses_working_endpoint(self):
        def fetch(**kwargs):
            if kwargs['endpoint']==download.OFFICIAL:
                self.calls.append(kwargs);raise OSError('connection failed')
            return self.fetch(**kwargs)
        result=download.ensure_model(self.root,preferred=download.OFFICIAL,download=fetch)
        self.assertEqual(result,download.snapshot_path(self.root))
        self.assertEqual(len(self.calls),len(download.FILES)+2)
        self.assertTrue(all(c['endpoint']==download.MIRROR for c in self.calls[2:]))
        for call in self.calls:
            self.assertEqual(call['revision'],download.MODEL_REVISION)
            self.assertEqual(call['repo_id'],download.MODEL_ID)
            self.assertFalse(call['token']);self.assertFalse(call['local_files_only'])
            self.assertEqual(Path(call['cache_dir']),self.root)
            self.assertNotIn('local_dir',call)
        receipt=json.loads((self.root/'v3186_model_download.json').read_text())
        self.assertTrue(receipt['local_loading_only'])
        self.assertEqual(set(receipt['files']),set(download.FILES))

    def test_complete_cache_never_calls_network_even_with_an_unrelated_endpoint_env(self):
        self.complete()
        with patch.dict('os.environ',{'HF_ENDPOINT':'https://custom-unused.invalid'}):
            download.ensure_model(self.root,True,download=lambda **kw:self.fail('network attempted'))
            download.ensure_model(self.root,False,download=lambda **kw:self.fail('network attempted'))
        self.assertIn('cached, verified',self.stdout.getvalue())

    def test_local_only_missing_file_does_not_download_or_claim_ready(self):
        with self.assertRaisesRegex(download.ModelDownloadError,'Missing cached preprocessor_config.json'):
            download.ensure_model(self.root,True,download=lambda **kw:self.fail('network attempted'))
        self.assertFalse((self.root/'v3186_model_download.json').exists())
        self.assertNotIn('PHONEME_MODEL_READY=',self.stdout.getvalue())

    def test_corrupt_weight_is_rejected_offline_and_only_that_file_is_repaired(self):
        self.complete();path=download.snapshot_path(self.root)/'pytorch_model.bin';path.write_bytes(b'wrong')
        with self.assertRaisesRegex(download.ModelDownloadError,'SHA256 mismatch'):
            download.ensure_model(self.root,True,download=self.fetch)
        before={name:(download.snapshot_path(self.root)/name).stat().st_mtime_ns for name in download.FILES[:-1]}
        download.ensure_model(self.root,preferred=download.MIRROR,download=self.fetch)
        self.assertEqual(len(self.calls),1);self.assertTrue(self.calls[0]['force_download'])
        self.assertEqual(self.calls[0]['filename'],'pytorch_model.bin')
        for name,stamp in before.items():self.assertEqual((download.snapshot_path(self.root)/name).stat().st_mtime_ns,stamp)

    def test_partial_download_survives_retry_without_forced_restart(self):
        partial=self.root/'partial-fixture.incomplete'
        def fetch(**kwargs):
            if not partial.exists():
                self.calls.append(kwargs);partial.write_bytes(b'partial data');raise OSError('interrupted')
            self.assertEqual(partial.read_bytes(),b'partial data')
            self.assertFalse(kwargs['force_download'])
            return self.fetch(**kwargs)
        download.ensure_model(self.root,preferred=download.MIRROR,download=fetch)
        self.assertEqual(len(self.calls),len(download.FILES)+1)
        self.assertTrue(partial.is_file())

    def test_all_endpoints_fail_with_bounded_attempts_and_actionable_error(self):
        def fail(**kwargs):self.calls.append(kwargs);raise OSError('offline')
        with self.assertRaisesRegex(download.ModelDownloadError,'Cache/partial files retained'):
            download.ensure_model(self.root,preferred=download.MIRROR,retries=2,download=fail)
        self.assertEqual(len(self.calls),4)
        self.assertFalse((self.root/'v3186_model_download.json').exists())

    def test_html_and_wrong_configuration_are_not_accepted_as_json(self):
        self.complete();path=download.snapshot_path(self.root)/'preprocessor_config.json'
        for data in (b'<html>503</html>',b'{"sampling_rate":8000}'):
            path.write_bytes(data)
            with self.assertRaises(download.ModelDownloadError):download.ensure_model(self.root,True,download=self.fetch)
        self.assertEqual(self.calls,[])

    def test_wrong_download_revision_is_rejected(self):
        def wrong(**kwargs):
            self.fetch(**kwargs);return str(self.root/'another-revision'/kwargs['filename'])
        with self.assertRaises(download.ModelDownloadError):
            download.ensure_model(self.root,preferred=download.MIRROR,retries=1,download=wrong)
        self.assertNotIn('PHONEME_MODEL_READY=',self.stdout.getvalue())

    def test_endpoint_selection_and_invalid_retry_budget(self):
        with patch.dict('os.environ',{'HF_ENDPOINT':download.MIRROR}):
            self.assertEqual(download.endpoint_order(),[download.MIRROR,download.OFFICIAL])
        with self.assertRaises(ValueError):download.endpoint_order('http://unsafe.invalid')
        for n in (0,6):
            with self.assertRaises(ValueError):download.ensure_model(self.root,retries=n,download=self.fetch)


class LocalLoadingTests(unittest.TestCase):
    def test_recognizer_loads_all_three_components_from_verified_local_snapshot(self):
        import torch
        from .prepare_phonemes import Recognizer
        seen=[]
        class Factory:
            def __init__(self,result):self.result=result
            def from_pretrained(self,path,**kwargs):seen.append((path,kwargs));return self.result
        class Tokenizer:
            all_special_ids=[0]
            def get_vocab(self):return {'<pad>':0,'a':1}
        model=torch.nn.Linear(1,1)
        model.config=types.SimpleNamespace(pad_token_id=0,conv_kernel=[10,3,3,3,3,2,2],conv_stride=[5,2,2,2,2,2,2])
        fake=types.ModuleType('transformers')
        fake.AutoFeatureExtractor=Factory(types.SimpleNamespace(sampling_rate=16000))
        fake.AutoTokenizer=Factory(Tokenizer());fake.AutoModelForCTC=Factory(model)
        with patch.dict('sys.modules',{'transformers':fake}), \
                patch('w2v_v3186.prepare_phonemes.ensure_model',return_value=Path('/data/verified-model')) as ensure:
            Recognizer('/data/cache','cpu',hf_endpoint=download.MIRROR,download_retries=3)
        ensure.assert_called_once_with('/data/cache',False,download.MIRROR,3)
        self.assertEqual(len(seen),3)
        for path,kwargs in seen:
            self.assertEqual(path,str(Path('/data/verified-model')))
            self.assertTrue(kwargs['local_files_only']);self.assertFalse(kwargs['trust_remote_code'])
        self.assertFalse(seen[1][1]['do_phonemize']);self.assertFalse(seen[2][1]['use_safetensors'])


if __name__=='__main__':unittest.main()
