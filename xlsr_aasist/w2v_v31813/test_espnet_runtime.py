"""Keep optional text downloads out of audio startup without disabling G2P."""
from contextlib import ExitStack,redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
from . import espnet_runtime as runtime

SOURCE=b'''import g2p_en

class G2p_en:
    def __init__(self):
        self.g2p = None

    def __call__(self, text):
        if self.g2p is None:
            self.g2p = g2p_en.G2p()
        return self.g2p(text)
'''
PATCHED=SOURCE.replace(b'import g2p_en\n',b'',1).replace(
    b'        if self.g2p is None:\n            self.g2p = g2p_en.G2p()',
    b'        if self.g2p is None:\n            import g2p_en\n\n            self.g2p = g2p_en.G2p()',1)


class EspnetRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.private=self.root/'xeus';self.target=self.private/runtime.RELATIVE_PATH
        self.target.parent.mkdir(parents=True);self.target.write_bytes(SOURCE)
        self.owner=self.private/'owner.json'
        self.owner.write_text(json.dumps(dict(schema='v31813_xeus_env_v1',espnet=runtime.ESPNET_REVISION)))
        self.distribution=SimpleNamespace(version='202506',locate_file=lambda _:self.target)
        self.stack.enter_context(patch.object(sys,'prefix',str(self.private)))
        self.stack.enter_context(patch.object(sys,'base_prefix',str(self.root/'base')))
        self.stack.enter_context(patch.object(runtime,'ORIGINAL_SHA256',runtime.sha256(SOURCE)))
        self.stack.enter_context(patch.object(runtime,'PATCHED_SHA256',runtime.sha256(PATCHED)))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def test_import_is_lazy_and_real_converter_remains_callable(self):
        # An absent converter must not prevent importing or constructing G2p_en.
        with patch.dict(sys.modules,{'g2p_en':None}):
            with self.assertRaises(ModuleNotFoundError):exec(SOURCE,{})
            self.assertTrue(runtime.patch_text_import(self.private,self.distribution))
            namespace={};exec(self.target.read_bytes(),namespace)
            converter=namespace['G2p_en']()
            with self.assertRaises(ModuleNotFoundError):converter('hello')
        calls=[];module=ModuleType('g2p_en')
        def convert(text):calls.append(text);return ['HH','AH','L','OW']
        module.G2p=lambda:convert
        with patch.dict(sys.modules,{'g2p_en':module}):
            self.assertEqual(converter('hello'),['HH','AH','L','OW'])
        self.assertEqual(calls,['hello'])
        self.assertEqual((self.private/'v31813_phoneme_tokenizer.original.py').read_bytes(),SOURCE)
        receipt=json.loads((self.private/'v31813_text_import_patch.json').read_text())
        self.assertEqual(receipt['patched_sha256'],runtime.sha256(PATCHED))
        self.assertFalse(runtime.patch_text_import(self.private,self.distribution))
        self.assertEqual(self.target.read_bytes(),PATCHED)

    def test_unknown_source_and_modified_backup_are_rejected(self):
        self.target.write_bytes(SOURCE+b'\n# unverified revision\n')
        with self.assertRaisesRegex(RuntimeError,'unverified'):runtime.patch_text_import(self.private,self.distribution)
        self.assertFalse((self.private/'v31813_phoneme_tokenizer.original.py').exists())
        self.target.write_bytes(SOURCE)
        backup=self.private/'v31813_phoneme_tokenizer.original.py';backup.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'backup changed'):runtime.patch_text_import(self.private,self.distribution)
        self.assertEqual(self.target.read_bytes(),SOURCE)

    def test_global_or_inherited_install_is_never_patched(self):
        with patch.object(sys,'base_prefix',str(self.private)):
            with self.assertRaisesRegex(RuntimeError,'private environment'):runtime.patch_text_import(self.private,self.distribution)
        outside=self.root/'global.py';outside.write_bytes(SOURCE)
        self.distribution.locate_file=lambda _:outside
        with self.assertRaisesRegex(RuntimeError,'outside'):runtime.patch_text_import(self.private,self.distribution)
        self.assertEqual(outside.read_bytes(),SOURCE);self.assertEqual(self.target.read_bytes(),SOURCE)

    def test_wrong_owner_revision_and_version_are_rejected(self):
        self.owner.write_text(json.dumps(dict(schema='v31813_xeus_env_v1',espnet='other')))
        with self.assertRaisesRegex(RuntimeError,'ownership/revision'):runtime.patch_text_import(self.private,self.distribution)
        self.owner.write_text(json.dumps(dict(schema='v31813_xeus_env_v1',espnet=runtime.ESPNET_REVISION)))
        self.distribution.version='different'
        with self.assertRaisesRegex(RuntimeError,'pinned environment'):runtime.patch_text_import(self.private,self.distribution)
        self.assertEqual(self.target.read_bytes(),SOURCE)


if __name__=='__main__':unittest.main()
