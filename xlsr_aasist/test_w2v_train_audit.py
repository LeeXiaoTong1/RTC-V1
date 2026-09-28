"""Small fixture tests for read-only inventory conclusions and report export."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

import audit_w2v_train as audit


def wave(path, seconds=1., frequency=200., silent_head=0.):
    path.parent.mkdir(parents=True,exist_ok=True)
    sr=16000
    values=(.15*np.sin(np.arange(round(seconds*sr))*frequency*2*np.pi/sr)).astype(np.float32)
    values[:round(silent_head*sr)]=0
    sf.write(path,values,sr,subtype='PCM_16')


class TrainAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.project=self.root/'project'
        self.data=self.project/'audio'
        self.data.mkdir(parents=True)

    def protocol(self, text):
        path=self.project/'train.txt'
        path.write_text(text,encoding='utf-8')
        return path

    def rows(self,text):
        issues=audit.Issues()
        rows,count=audit.protocol_rows(self.protocol(text),self.data,issues)
        return rows,count,issues

    def fixture(self):
        ids=[('offline/en/r.wav','real',8.,211.,5.),
             ('online/en/r.wav','real',8.,223.,0.),
             ('offline/en/copy.wav','real',8.,211.,5.),
             ('offline/en/f.wav','fake',1.,227.,0.),
             ('offline/zh/r.wav','real',5.,229.,0.),
             ('offline/zh/f.wav','fake',5.,233.,0.),
             ('online/zh/f.wav','fake',5.,239.,0.)]
        for name,label,seconds,freq,silent in ids:
            wave(self.data/name,seconds,freq,silent)
        shutil.copyfile(self.data/'offline/en/r.wav',self.data/'offline/en/copy.wav')
        protocol=self.protocol(''.join(f'{name} {label}\n' for name,label,*_ in ids))
        pairs=self.project/'pairs.jsonl'
        pairs.write_text(json.dumps({'offline':'offline/en/r.wav','online':'online/en/r.wav','label':1})+'\n',encoding='utf-8')
        cache=self.project/'cache'
        cache.mkdir()
        wave(cache/'a.wav')
        audit.write_json(cache/'config.json',{'role':'train','generation':0,'protocol_sha256':audit.sha256(protocol)})
        manifest=[]
        for name,label,*_ in ids:
            if not name.startswith('offline/'):
                continue
            for band in range(4):
                manifest.append({'source':name,'source_sha256':audit.sha256(self.data/name),
                                 'label':audit.LABELS[label],'role':'train','generation':0,
                                 'band':band,'snr_db':7.5+5*band,'audio':'a.wav'})
        (cache/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest),encoding='utf-8')
        run=self.project/'exp'/'run'
        (run/'stage3').mkdir(parents=True)
        sentinel=run/'stage3'/'best_model.pt'
        sentinel.write_bytes(b'checkpoint sentinel: audit must not load or modify')
        config={'train_data_path':str(self.data),'train_protocol':str(protocol),'rtc_pairs':str(pairs),
                'train_noisy_cache':str(cache),'extra_train_noisy_cache':[],
                'class_counts':[3,4],'data_fingerprints':{str(protocol.resolve()):audit.sha256(protocol)},
                'ordinary_sampling':'legacy'}
        audit.write_json(run/'stage3'/'config.json',config)
        return argparse.Namespace(run_dir=str(run),project_root=str(self.project),out=str(self.root/'reports'/'audit'),
                                  download_dir=str(self.root/'downloads'),workers=2,upload_temp=False)

    def test_protocol_formats_duplicates_and_exact_directory_tags(self):
        rows,count,issues=self.rows('offline/en/a.wav real\nspk offline/zh/b.wav x y spoof\noffline/en/a.wav real\noffline/en/a.wav fake\n')
        self.assertEqual((count,len(rows)),(4,2))
        self.assertEqual([r['label'] for r in rows],[1,0])
        self.assertEqual(issues.counts,{'duplicate_protocol_id':1,'conflicting_protocol_label':1})
        self.assertEqual(audit.tags('offline/en/a.wav')[:2],('en','offline'))
        self.assertEqual(audit.tags('offline/english/a.wav')[0],'unknown')
        self.assertEqual(audit.tags('offline/en/zh/a.wav')[0],'unknown')

    def test_protocol_rejects_traversal_absolute_and_bad_labels(self):
        for text in ('../a.wav real','/outside.wav real','C:/outside.wav real','en/a.wav unsure'):
            with self.subTest(text=text),self.assertRaises(ValueError):
                self.rows(text)

    def test_audio_reports_original_duration_and_head_middle_difference(self):
        wave(self.data/'offline/en/a.wav',12.,211.,5.)
        rows,_,_=self.rows('offline/en/a.wav real\n')
        row=audit.inspect_audio(rows[0])
        self.assertEqual(row['status'],'ok')
        self.assertEqual(row['seconds'],12.)
        self.assertEqual(row['head']['active_frame_fraction'],0.)
        self.assertGreater(row['middle']['active_frame_fraction'],.5)
        self.assertAlmostEqual(row['head_unused_seconds'],12.-audit.CUT_SECONDS)
        self.assertEqual(row['file_sha256'],audit.sha256(self.data/'offline/en/a.wav'))

    def test_missing_and_empty_audio_are_preserved_as_errors(self):
        sf.write(self.data/'empty.wav',np.zeros(0),16000)
        rows,_,_=self.rows('missing.wav real\nempty.wav fake\n')
        result=[audit.inspect_audio(r) for r in rows]
        self.assertTrue(all(r['status']=='error' for r in result))

    def test_file_copy_merge_but_conflicting_labels_not_merged(self):
        rows=[{'source_id':'a','label':1,'file_sha256':'same'},
              {'source_id':'b','label':1,'file_sha256':'same'},
              {'source_id':'c','label':1,'file_sha256':'conflict'},
              {'source_id':'d','label':0,'file_sha256':'conflict'}]
        comp=audit.Components([r['source_id'] for r in rows]); issues=audit.Issues()
        dup=audit.duplicates(rows,comp,issues)
        self.assertEqual(comp.find('a'),comp.find('b'))
        self.assertNotEqual(comp.find('c'),comp.find('d'))
        self.assertEqual(len(dup),2)
        self.assertEqual(issues.counts['same_file_bytes_conflicting_labels'],1)

    def test_full_run_preserves_inputs_counts_and_exports_only_metadata(self):
        args=self.fixture()
        before={str(p):audit.sha256(p) for p in self.project.rglob('*') if p.is_file()}
        with redirect_stdout(io.StringIO()):
            summary=audit.run(args)
        after={str(p):audit.sha256(p) for p in self.project.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(summary['status'],'complete')
        self.assertEqual(summary['class_counts'],[3,4])
        en=next(g for g in summary['groups'] if g['group']=='en-real')
        self.assertEqual((en['protocol_ids'],en['unique_file_sha256'],en['known_source_components']),(3,2,1))
        self.assertEqual(summary['all']['known_source_components'],5)
        bank=summary['noisy_caches'][0]
        self.assertEqual((bank['rows'],bank['valid_rows'],bank['covered_sources']),(20,20,5))
        self.assertEqual(bank['missing_source_ids'],[])
        out=Path(args.out)
        completed=audit.read_json(out/'completed.json')
        self.assertEqual(audit.sha256(completed['archive']),completed['archive_sha256'])
        with zipfile.ZipFile(completed['archive']) as z:
            self.assertIsNone(z.testzip())
            self.assertTrue(all(not name.endswith(('.wav','.pt','.pth')) for name in z.namelist()))
            hashes=json.loads(z.read('audit/file_hashes.json'))
            for name,digest in hashes.items():
                self.assertEqual(audit.hashlib.sha256(z.read('audit/'+name)).hexdigest(),digest)
        with self.assertRaises(FileExistsError):
            audit.run(args)

    def test_input_and_cache_directories_cannot_be_output(self):
        args=self.fixture()
        for out in (self.data/'oops',Path(args.run_dir)/'oops',self.project/'cache'/'oops'):
            args.out=str(out)
            with self.subTest(out=out),self.assertRaises(ValueError):
                audit.run(args)
            self.assertFalse(out.exists())

    def test_stale_source_hash_and_missing_band_are_reported(self):
        args=self.fixture()
        rows,_,issues=self.rows((self.project/'train.txt').read_text())
        records={r['source_id']:audit.inspect_audio(r) for r in rows}
        records['offline/en/r.wav']['file_sha256']='changed'
        cache=self.project/'cache'
        path=cache/'manifest.jsonl'
        lines=path.read_text().splitlines()
        lines=[s for s in lines if not(json.loads(s)['source']=='offline/en/f.wav' and json.loads(s)['band']==3)]
        path.write_text('\n'.join(lines)+'\n')
        with redirect_stdout(io.StringIO()):
            result=audit.cache_inventory(cache,records,audit.sha256(self.project/'train.txt'),issues)
        self.assertEqual(issues.counts['invalid_noisy_cache_row'],4)
        self.assertEqual(issues.counts['incomplete_noisy_bands'],1)
        self.assertIn('offline/en/r.wav',result['missing_source_ids'])

    def test_dev_cache_is_excluded(self):
        cache=self.root/'dev_cache'; cache.mkdir()
        audit.write_json(cache/'config.json',{'role':'dev_seen'})
        (cache/'manifest.jsonl').write_text('')
        issues=audit.Issues()
        result=audit.cache_inventory(cache,{},'hash',issues)
        self.assertEqual(issues.counts['non_train_cache_role'],1)
        self.assertIn('skipped',result)

    def test_invalid_pairs_not_used_as_independent_source_evidence(self):
        records={'offline/en/a':{'label':1,'domain':'offline','language_group':'en'},
                 'online/zh/b':{'label':1,'domain':'online','language_group':'zh'}}
        comp=audit.Components(records); issues=audit.Issues()
        path=self.root/'pairs.jsonl'
        path.write_text(json.dumps({'offline':'offline/en/a','online':'online/zh/b','label':1})+'\n')
        result=audit.pair_inventory(path,records,comp,issues)
        self.assertEqual(result['valid_pairs'],0)
        self.assertNotEqual(comp.find('offline/en/a'),comp.find('online/zh/b'))

    def test_csv_formula_escape(self):
        path=self.root/'flags.csv'
        audit.save_csv(path,[{'source':'=x','value':-4.}])
        self.assertIn("'=x",path.read_text(encoding='utf-8-sig'))
        self.assertIn('-4.0',path.read_text(encoding='utf-8-sig'))

    def test_upload_validates_url_and_does_not_invoke_a_shell(self):
        with patch.object(audit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'https://temp.sh/abc/report.zip\n','')) as call:
            self.assertEqual(audit.upload_report(Path('report.zip')),'https://temp.sh/abc/report.zip')
            self.assertIsInstance(call.call_args.args[0],list)
            self.assertNotIn('shell',call.call_args.kwargs)
        with patch.object(audit.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'https://evil.invalid/report.zip','')):
            with self.assertRaises(ValueError):
                audit.upload_report(Path('report.zip'))


if __name__=='__main__':
    unittest.main()
