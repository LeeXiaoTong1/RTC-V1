"""Read-only storage audit contracts; no real datasets/models required."""
import hashlib
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_w2v_storage as audit


class StorageAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.source=self.root/'exp'/'w2v_v316_tfcl_source'
        self.source.mkdir(parents=True)
        self.producer=self.root/'exp'/'w2v_v37_source'
        self.original=self.put('data/official/train/original.wav',b'original')
        self.dev=self.put('data/rtc_v35/dev/fixed.wav',b'fixed dev')
        self.legacy=self.put('data/rtc_v35/train/epoch_001/old.wav',b'generated')
        self.noise=self.put('data/noise/background.wav',b'noise')
        self.write(self.legacy.parent.parent/'owner.json',{'role':'train'})
        self.write(self.dev.parent/'owner.json',{'role':'dev'})
        for split,rows in (
            ('train',[{'audio':str(self.original),'condition':'online'},
                      {'audio':str(self.legacy),'condition':'seen'}]),
            ('dev',[{'audio':str(self.dev),'condition':'seen'}]),
        ):
            folder=self.producer/'features'/split
            self.write(folder/'rows.json',rows)
            signature=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.write(folder/'owner.json',dict(records_digest=signature,shape=[len(rows),128],identity={'split':split}))
            self.write(folder/'complete.json',dict(files={'rows.json':self.sha(folder/'rows.json')},owner_sha256=self.sha(folder/'owner.json')))
        self.write(self.source/'config.json',dict(version='3.16',v37_run=str(self.producer),
                   train_data_path=str(self.original.parent),noise_records={'train':[{'path':str(self.noise)}]},
                   train_caches=[str(self.legacy.parent.parent)]))

    def put(self,relative,data):
        path=self.root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        return path

    def write(self,path,obj):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(obj),encoding='utf-8')

    def sha(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()

    def collect(self):return audit.collect(self.root,str(self.source),lambda _:None)

    def snapshot(self):return {str(p):self.sha(p) for p in self.root.rglob('*') if p.is_file()}

    def test_read_only_and_current_vs_historical_data(self):
        before=self.snapshot()
        report=self.collect()
        self.assertEqual(before,self.snapshot())
        self.assertTrue(report['read_only'])
        self.assertFalse(report['deletion_authorized'])
        self.assertEqual(report['current_input_errors'],[])
        rows={r['path']:r for r in report['entries']}
        old=rows[str(self.legacy.parent.parent)]
        self.assertEqual(old['status'],'REVIEW_OLD_CACHE')
        self.assertIn(str(self.source/'config.json'),old['historical_references'])
        self.assertEqual(rows[str(self.dev.parent)]['status'],'KEEP_V318_INPUT_OR_METADATA')
        self.assertTrue(any('noise recording: train' in r['v318_reasons'] for r in rows.values()))
        self.assertTrue(any('V3.18 canonical manifest metadata' in r['v318_reasons'] for r in rows.values()))

    def test_retirement_audit_is_measured_not_parsed(self):
        path=self.put('exp/cache_retirement_20260930_175927_955232.json',b'not valid JSON')
        original=audit.load
        def guard(p):
            self.assertNotEqual(Path(p),path)
            return original(p)
        with patch.object(audit,'load',guard):report=self.collect()
        row=next(r for r in report['entries'] if r['path']==str(path))
        self.assertEqual(row['logical_bytes'],len(b'not valid JSON'))
        self.assertEqual(row['status'],'KEEP_AUDIT_RECORD')

    def test_checkpoint_references_and_combined_last_are_preserved(self):
        old=self.put('exp/w2v_v35_run/epoch_1.pt',b'old')
        last=self.put('exp/w2v_v317_run/last.pt',b'combined best and last')
        best=self.put('exp/w2v_v35_run/best.pt',b'best')
        self.write(self.source/'best_guarded.json',{'checkpoint':str(old)})
        rows={r['path']:r for r in self.collect()['entries']}
        for path in (old,last,best):self.assertEqual(rows[str(path)]['status'],'KEEP_CHECKPOINT')

    def test_unreferenced_intermediate_is_review_not_delete(self):
        path=self.put('exp/w2v_v35_run/epoch_2.pt',b'unknown utility')
        row=next(r for r in self.collect()['entries'] if r['path']==str(path))
        self.assertEqual(row['status'],'REVIEW_CHECKPOINT')
        self.assertTrue(path.exists())

    def test_changed_manifest_is_reported_as_incomplete(self):
        self.write(self.producer/'features'/'dev'/'rows.json',[])
        report=self.collect()
        self.assertTrue(report['current_input_errors'])
        self.assertFalse(report['deletion_authorized'])

    def test_cli_only_creates_two_new_reports(self):
        before=self.snapshot()
        output=io.StringIO()
        with patch('sys.argv',['audit_w2v_storage.py','--root',str(self.root),'--data-run',str(self.source)]),redirect_stdout(output):
            audit.main()
        after=self.snapshot()
        self.assertEqual({k:after[k] for k in before},before)
        self.assertEqual(len(set(after)-set(before)),2)
        self.assertIn('FILES_DELETED=0',output.getvalue())
        self.assertIn('STORAGE_AUDIT_JSON=',output.getvalue())

    def test_cli_rejects_apply(self):
        with patch('sys.argv',['audit_w2v_storage.py','--data-run',str(self.source),'--apply']),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:audit.main()
        self.assertEqual(exc.exception.code,2)

    def test_external_historical_cache_directory_is_discovered(self):
        with tempfile.TemporaryDirectory() as outside:
            cache=Path(outside)/'rtc_noisy_cache_v2'/'train_g1'
            self.write(cache/'config.json',{'role':'train'})
            (cache/'simulated.wav').write_bytes(b'cached')
            self.write(self.root/'exp'/'legacy_source_config.json',{'train_noisy_cache':str(cache)})
            rows={r['path']:r for r in self.collect()['entries']}
            self.assertEqual(rows[str(cache.resolve())]['status'],'REVIEW_OLD_CACHE')
            self.assertTrue((cache/'simulated.wav').exists())

    def test_no_symlink_following(self):
        with tempfile.TemporaryDirectory() as outside:
            target=Path(outside)/'valuable.wav'
            target.write_bytes(b'do not traverse')
            link=self.root/'data'/'linked'
            try:link.symlink_to(outside,target_is_directory=True)
            except OSError:self.skipTest('Host does not allow symlink creation')
            value=audit.measure(link,lambda _:None)
            self.assertEqual(value['logical_bytes'],0)
            self.assertEqual(value['symlinks_skipped'],1)


if __name__=='__main__':unittest.main()
