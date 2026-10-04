"""Network-free import/reuse, corruption rejection and actionable HF failure."""
from contextlib import ExitStack,redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from huggingface_hub.utils import LocalEntryNotFoundError
from . import language,language_bundle as bundle


class BundleTests(unittest.TestCase):
    def fixture(self,root):
        contents={name:('public-fixture '+name).encode() for name in language.ASSET_FILES}
        expected={name:dict(size=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in contents.items()}
        stack=ExitStack()
        stack.enter_context(patch.object(bundle,'EXPECTED',expected))
        stack.enter_context(patch.object(language,'WEIGHT_SHA256',expected['embedding_model.ckpt']['sha256']))
        stack.enter_context(patch.object(language,'WEIGHT_BYTES',expected['embedding_model.ckpt']['size']))
        self.addCleanup(stack.close)
        entries=dict(contents,**{'bundle.json':json.dumps(bundle.manifest()).encode(),
            'LICENSE':b'Apache License Version 2.0'})
        return entries

    def archive(self,path,entries):
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for name,value in entries.items():z.writestr(name,value)
        return path

    def test_import_reuse_and_ensure_assets_never_call_hub(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);entries=self.fixture(root)
            archive=self.archive(root/'bundle.zip',entries);cache=root/'cache'
            target=bundle.import_bundle(archive,cache)
            before={p.name:p.read_bytes() for p in target.iterdir()}
            with patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('No HF request allowed')):
                for offline in (False,True):
                    assets=language.ensure_language_assets(dict(language_cache_dir=str(cache),language_offline=offline))
                    self.assertEqual(assets['files']['embedding_model.ckpt']['sha256'],bundle.EXPECTED['embedding_model.ckpt']['sha256'])
                self.assertEqual(bundle.import_bundle(archive,cache),target)
            self.assertEqual(before,{p.name:p.read_bytes() for p in target.iterdir()})
            self.assertEqual(len(list(cache.rglob('embedding_model.ckpt'))),1)

    def test_corrupt_weights_fail_before_install_and_clean_only_own_stage(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);entries=self.fixture(root)
            entries['embedding_model.ckpt']=b'x'*len(entries['embedding_model.ckpt'])
            cache=root/'cache';cache.mkdir();protected=cache/'best_model.pt';protected.write_bytes(b'user best')
            with self.assertRaisesRegex(ValueError,'pinned official bytes'):
                bundle.import_bundle(self.archive(root/'bad.zip',entries),cache)
            self.assertEqual(protected.read_bytes(),b'user best')
            self.assertFalse((cache/'v37_offline'/language.REVISION).exists())
            self.assertFalse(list(cache.glob('.v37-import-*')))

    def test_rejects_wrong_revision_traversal_duplicate_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entries=self.fixture(root)
            wrong=dict(entries);meta=json.loads(wrong['bundle.json']);meta['revision']='main'
            wrong['bundle.json']=json.dumps(meta).encode()
            traversal=dict(entries);traversal['../outside.pt']=b'bad'
            size=dict(entries);size['hyperparams.yaml']=b'oversized'*100
            for index,bad in enumerate((wrong,traversal,size)):
                with self.subTest(index=index),self.assertRaises(ValueError):
                    bundle.import_bundle(self.archive(root/f'bad{index}.zip',bad),root/f'cache{index}')
            duplicate=root/'duplicate.zip'
            with zipfile.ZipFile(duplicate,'w') as z:
                for name,value in entries.items():z.writestr(name,value)
                with self.assertWarns(UserWarning):z.writestr('README.md',entries['README.md'])
            with self.assertRaisesRegex(ValueError,'exactly'):
                bundle.import_bundle(duplicate,root/'dup_cache')
            self.assertFalse((root/'outside.pt').exists())

    def test_disk_guard_and_imported_corruption_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);entries=self.fixture(root);archive=self.archive(root/'bundle.zip',entries)
            with patch.object(bundle.shutil,'disk_usage',return_value=type('Space',(),{'free':0})()):
                with self.assertRaisesRegex(OSError,'no existing files deleted'):
                    bundle.import_bundle(archive,root/'cache')
            target=bundle.import_bundle(archive,root/'cache')
            (target/'hyperparams.yaml').write_bytes(b'wrong')
            with patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('No silent redownload')):
                with self.assertRaisesRegex(ValueError,'pinned official bytes'):
                    language.ensure_language_assets({'language_cache_dir':str(root/'cache')})

    def test_hf_failure_explains_offline_recovery_and_preserves_cause(self):
        def unavailable(**kwargs):
            if kwargs['local_files_only']:raise LocalEntryNotFoundError('Not cached')
            raise ConnectionError('network blocked')
        with tempfile.TemporaryDirectory() as directory,patch('huggingface_hub.hf_hub_download',side_effect=unavailable):
            with self.assertRaisesRegex(RuntimeError,'--import-bundle') as raised:
                language.ensure_language_assets({'language_cache_dir':directory})
            self.assertIsInstance(raised.exception.__cause__,ConnectionError)

    def test_export_allowlist_contains_only_public_files_license_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory);entries=self.fixture(root);source=root/'source';source.mkdir()
            for name in language.ASSET_FILES:(source/name).write_bytes(entries[name])
            (source/'best_model.pt').write_bytes(b'private-trained-best')
            (source/'train.wav').write_bytes(b'private-audio')
            with patch('huggingface_hub.hf_hub_download',side_effect=lambda **kw:str(source/kw['filename'])):
                archive=bundle.export_bundle(source,root/'public.zip')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),set(language.ASSET_FILES)|{'bundle.json','LICENSE'})
            with self.assertRaises(FileExistsError),patch('huggingface_hub.hf_hub_download',side_effect=lambda **kw:str(source/kw['filename'])):
                bundle.export_bundle(source,archive)
            with patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('Offline')):
                bundle.import_bundle(archive,root/'destination')


if __name__=='__main__':unittest.main()
