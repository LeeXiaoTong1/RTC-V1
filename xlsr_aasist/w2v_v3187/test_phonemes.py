import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from w2v_v3186.phonemes import bind_manifest as legacy_bind, lineage as legacy_lineage
from .phonemes import (PhonemeCache, bind_manifest, lineage, validate_record,
                       digest_file, atomic_json, read_json, SHORT_REASON)
from .prepare_phonemes import prepare, parser


def fixture(root,sizes):
    run=root/'run';run.mkdir();rows=[]
    for i,n in enumerate(sizes):
        source=f'off-{i}';off=root/(source+'.wav')
        wave=np.random.default_rng(i).normal(0,.1,n).astype(np.float32)
        sf.write(off,wave,16000,subtype='FLOAT');sha=digest_file(off)
        base=dict(source_id=source,group_id=sha,source_sha256=sha,language='en',label=1,
                  split='train',full_length=True,output_samples=n)
        rows.append(dict(base,id=source,condition='offline',audio=str(off),audio_sha256=sha))
        on=root/(source+'-on.wav');sf.write(on,wave*.9,16000,subtype='FLOAT')
        rows.append(dict(base,id=source+'-on',condition='online',audio=str(on),audio_sha256=digest_file(on)))
    atomic_json(run/'train_rows.json',rows)
    atomic_json(run/'config.json',dict(version='3.18.7',omni_arch='3b',pooling='asp'))
    atomic_json(run/'dev_rows.json',[]);atomic_json(run/'dev_partition.json',{})
    return run,rows


class PhonemeTests(unittest.TestCase):
    def test_real_short_originals_never_invoke_recognizer_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows=fixture(root,[319,1000]);calls=[]
            def factory(*args):
                def recognize(wave):
                    calls.append(len(wave))
                    return [dict(token='a',instance=0,start_sample=0,end_sample=800,posterior=.9)],{}
                return recognize
            with patch('torch.load',side_effect=AssertionError('Do not load a detector')):
                binding=prepare(run,root/'phones',root/'models','cpu',True,factory)
            self.assertEqual(calls,[1000,1000])
            cache=PhonemeCache(binding['path'],rows,binding=binding)
            for row in rows:
                record=cache.get(row)
                if row['output_samples']<400:
                    self.assertEqual(record['segments'],[])
                    self.assertEqual(record['quality']['reason'],SHORT_REASON)
                    self.assertEqual(record['identity']['num_samples'],319)
                    bad=dict(record,segments=[dict(start_sample=0,end_sample=200)])
                    with self.assertRaises(ValueError):validate_record(bad,row)
                else:self.assertEqual(lineage(row),legacy_lineage(row))
            # Idempotent preparation rechecks bytes and records, no model load.
            repeated=prepare(run,root/'phones',root/'models','cpu',True,
                lambda *a: (_ for _ in ()).throw(AssertionError('Unexpected recognizer')))
            self.assertEqual(repeated,binding)

    def test_normal_existing_metadata_remains_legacy_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows=fixture(root,[1000])
            factory=lambda *a:lambda wave:([dict(token='a',instance=0,start_sample=0,end_sample=800,posterior=.9)],{})
            binding=prepare(run,root/'phones',root/'models','cpu',True,factory)
            self.assertEqual(binding,legacy_bind(binding['path'],rows))
            self.assertEqual(bind_manifest(binding['path'],rows),binding)

    def test_all_subframe_inventory_has_valid_empty_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,rows=fixture(root,[1,399])
            binding=prepare(run,root/'phones',root/'models','cpu',True,
                lambda *a: (_ for _ in ()).throw(AssertionError('Sub-frame audio has no phones')))
            manifest=read_json(binding['path'])
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['summary']['originals_below_receptive_field'],4)
            self.assertEqual(manifest['summary']['segments'],0)

    def test_short_lineage_still_rejects_nontrain_and_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            _,rows=fixture(Path(temp),[100])
            with self.assertRaises(ValueError):lineage(dict(rows[0],split='dev'))
            with self.assertRaises(ValueError):lineage(dict(rows[0],output_samples=0))
            args=parser().parse_args(['--data-run','run','--output-dir','out','--model-cache','models'])
            self.assertEqual(args.parent_run,'run')


if __name__=='__main__':unittest.main()
