"""Standard-library regression tests for historical cache retirement audits."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import cleanup


class RetirementMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.exp = self.root/'exp'
        self.exp.mkdir()
        self.audit = self.exp/'cache_retirement_20260930_175927_955232.json'
        self.run = self.exp/'w2v_v312_test'
        self.run.mkdir()
        (self.run/'config.json').write_text('{"version":"3.12"}')
        (self.run/'best.pt').write_bytes(b'best')
        (self.run/'last.pt').write_bytes(b'last')

    def write(self, obj):
        self.audit.write_text(json.dumps(obj, ensure_ascii=True), encoding='utf-8')

    def test_all_strings_match_json_across_tiny_chunks(self):
        obj = dict(status='complete', files=[
            dict(path='C:\\cache\\语音\\quote".wav', identity=[1, 2, 1.25e-30, -999]),
            dict(path='escaped\\backslash.npy', nested={'ref.pt': [None, True, False, 'tail']}),
        ], protected=['exp/run/last.pt'], deleted_files=2)
        self.write(obj)
        for chunk in (1, 2, 3, 7, 31, 65536):
            with self.subTest(chunk=chunk):
                self.assertEqual(list(cleanup.retirement_strings(self.audit, chunk)),
                                 list(cleanup.strings(obj)))

    def test_large_audit_protects_checkpoint_after_files_array(self):
        self.write(dict(status='partial', files=[dict(path='gone.wav')]*100,
                        protected=[str(self.run/'last.pt')]))
        with patch.object(cleanup, 'MAX_METADATA_BYTES', 64):
            plan = cleanup.inventory(self.root)
        self.assertEqual(plan['candidates'], [])
        self.assertIn(str(self.audit), plan['metadata_fingerprints'])
        self.assertIn(str(self.run/'last.pt'), plan['protected_weights'])
        cleanup.apply(plan)
        self.assertTrue((self.run/'last.pt').exists())

    def test_files_records_and_unknown_fields_also_protect_references(self):
        for obj in (
            dict(files=[dict(path=str(self.run/'last.pt'))]),
            dict(files=[], extra={str(self.run/'last.pt'): 'hash'}),
        ):
            with self.subTest(obj=obj):
                self.write(obj)
                self.assertEqual(cleanup.inventory(self.root)['candidates'], [])

    def test_empty_files_allows_unreferenced_intermediate_cleanup(self):
        self.write(dict(files=[], protected=[str(self.run/'best.pt')]))
        plan = cleanup.inventory(self.root)
        self.assertEqual([item['path'] for item in plan['candidates']], [str(self.run/'last.pt')])
        cleanup.apply(plan)
        self.assertTrue(self.audit.exists())
        self.assertTrue((self.run/'best.pt').exists())
        self.assertFalse((self.run/'last.pt').exists())

    def test_malformed_or_truncated_audits_fail_before_delete(self):
        for text in ('', '{', '{"files":[', '{"files":[{}],"protected":["last.pt"',
                     '{"files":[{},]}', '{"files":[],}', '{"files":[]}{"x":1}',
                     '{"files":{}}', '{"files":[],"n":1e}', '{1:2}',
                     '{"files":[],"x":"bad\\q"}', '{"files":[] "x":1}'):
            with self.subTest(text=text):
                self.audit.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError): cleanup.inventory(self.root)
                self.assertTrue((self.run/'last.pt').exists())

    def test_unknown_oversized_metadata_still_blocks_cleanup(self):
        path = self.exp/'unknown.json'
        path.write_text(json.dumps({'notes': 'x'*100}))
        with patch.object(cleanup, 'MAX_METADATA_BYTES', 64):
            with self.assertRaisesRegex(ValueError, 'Oversize metadata'):
                cleanup.inventory(self.root)
        self.assertTrue((self.run/'last.pt').exists())

    def test_nested_retirement_filename_does_not_bypass_size_limit(self):
        path = self.run/self.audit.name
        path.write_text(json.dumps({'files': ['x'*100]}))
        with patch.object(cleanup, 'MAX_METADATA_BYTES', 64):
            with self.assertRaisesRegex(ValueError, 'Oversize metadata'):
                cleanup.inventory(self.root)

    def test_audit_changed_after_plan_blocks_delete(self):
        self.write(dict(files=[]))
        plan = cleanup.inventory(self.root)
        self.write(dict(files=[], protected=[str(self.run/'last.pt')]))
        with self.assertRaisesRegex(ValueError, 'Metadata changed'):
            cleanup.apply(plan)
        self.assertTrue((self.run/'last.pt').exists())

    def test_change_during_stream_scan_blocks_delete(self):
        self.write(dict(files=[]))
        original = cleanup.metadata_strings
        def changed(path, exp):
            yield from original(path, exp)
            if path == self.audit:
                self.write(dict(files=[], protected=[str(self.run/'last.pt')]))
        with patch.object(cleanup, 'metadata_strings', changed):
            with self.assertRaisesRegex(ValueError, 'Metadata changed during'):
                cleanup.inventory(self.root)
        self.assertTrue((self.run/'last.pt').exists())

    def test_individual_value_memory_limit_is_enforced(self):
        reader = cleanup._JSONReader(io.StringIO('"'+'x'*100+'"'), chunk_size=4, max_value=16)
        with self.assertRaisesRegex(ValueError, 'oversized individual'):
            reader.value()


if __name__ == '__main__': unittest.main()
