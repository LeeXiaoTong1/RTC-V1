from contextlib import nullcontext, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from w2v_v31813.test_preparation import phone_fixture
from w2v_v31810.identity import registry_config
from .common import atomic_json, read_json, digest, fingerprint
from .test_dataset import prepared_fixture
from .dataset import transfer
from .prepare_phonemes import prepare as prepare_phones
from .reuse import phone_records
from .config import parser, validate
from .probe import paired_probe


class PreparationTests(unittest.TestCase):
    def test_same_source_gradient_probe_is_balanced_and_observational(self):
        import torch
        from w2v_v3189.data import prepared
        from w2v_v31810.test_meta import assert_state_equal
        from .common import capture_rng
        from .test_core import Tiny,cfg
        with tempfile.TemporaryDirectory() as tmp:
            rows=phone_fixture(Path(tmp));model=Tiny().train();before=copy.deepcopy(model.state_dict());rng=capture_rng()
            class Plan:
                def epoch_batches(self,epoch):
                    return [[dict(index=i,kind='offline') for i,r in enumerate(rows) if r['condition']=='offline']]
            class Data:
                def __init__(self,rows,cfg):self.rows=rows
                def original(self,row):return sf.read(row['audio'],dtype='float32')[0]
                def __getitem__(self,t):
                    r=self.rows[t['index']];return [prepared(r,self.original(r)*.6,role='noisy')]
            result=paired_probe(model,rows,cfg(paired_probe_per_cell=1),Plan(),Data)
            self.assertEqual((result['status'],result['sources'],result['optimizer_updates']),('measured',4,0))
            self.assertTrue(result['gradients']['finite']);self.assertTrue(model.training)
            assert_state_equal(self,before,model.state_dict());assert_state_equal(self,rng,capture_rng())
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_incremental_phonemes_only_recognize_new_train_originals(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);old=root/'old';old.mkdir();rows=phone_fixture(old)
            new=[];off='new-offline';group=None
            for kind in ('offline','online'):
                path=root/(kind+'.wav');sf.write(path,np.random.default_rng(len(kind)).normal(0,.1,4240),16000,subtype='PCM_16')
                if group is None:group=digest(path)
                new.append(dict(rows[0],id='new-'+kind,source_id=off,group_id=group,audio=str(path),
                    audio_sha256=digest(path),condition=kind,canonical_source='new-source'))
            data=root/'data';data.mkdir();atomic_json(data/'train_rows.json',rows+new)
            calls=[]
            def factory(*args):
                def recognize(wave):
                    calls.append(len(wave))
                    return [dict(instance=0,token='a',start_sample=0,end_sample=3000,support_start_sample=200,support_end_sample=2960,posterior=.9)],{}
                return recognize
            with patch('w2v_v41.prepare_phonemes.resolve_data_run',return_value=data):
                result=prepare_phones(data,root/'out',root/'models','cpu',recognizer_factory=factory,reuse_manifest=old/'manifest.json')
                self.assertEqual(len(calls),2)
                again=prepare_phones(data,root/'out',root/'models','cpu',recognizer_factory=factory,reuse_manifest=old/'manifest.json')
                self.assertEqual(result,again);self.assertEqual(len(calls),2)
            original=read_json(old/'manifest.json')
            for entry in original['records'].values():self.assertEqual(digest(old/entry['path']),entry['sha256'])

    def test_corrupt_phone_record_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=phone_fixture(root);data=read_json(root/'manifest.json')
            path=root/next(iter(data['records'].values()))['path'];path.write_text('{}')
            with self.assertRaises(ValueError):phone_records(rows,root/'copy',root/'manifest.json')

    def test_defaults_and_prepare_only_do_not_start_detector(self):
        args=parser().parse_args([]);validate(args)
        self.assertEqual((args.epochs,args.pcl,args.dev_real_train_fraction,args.gradient_coordination),(10,'dual',.75,'pcgrad'))
        with self.assertRaises(ValueError):validate(parser().parse_args(['--epochs','1']))
        with patch('w2v_v41.workflow.ensure_idle'),patch('w2v_v41.workflow.run_lock',return_value=nullcontext()),\
             patch('w2v_v41.workflow.prepare_missing',return_value=Path('fixture')),\
             patch('w2v_v41.workflow.initialize') as initialize,patch('sys.argv',['workflow','--prepare-only']),redirect_stdout(io.StringIO()):
            from .workflow import main
            main();initialize.assert_not_called()


if __name__=='__main__':unittest.main()
