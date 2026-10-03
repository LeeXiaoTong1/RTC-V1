"""Storage exhaustion cannot lose the previous resume point or measured Dev."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from w2v_aasist.runtime import atomic_save, sha256, storage_size
from . import SCHEMA, storage, train as training
from .test_train import fixture
from .test_cache import fixture as cache_fixture, FakeRTC
from . import cache
import recover_w2v_v35 as recovery


class StorageTests(unittest.TestCase):
    def test_compatibility_accepts_only_the_specific_storage_patch(self):
        old={'w2v_v35/train.py':storage.ORIGINAL_TRAIN_SHA256,'torch':'2.5.1',
             'w2v_v35/losses.py':'same','w2v_v35/cache.py':'same'}
        current={**old,'w2v_v35/train.py':storage.PATCHED_TRAIN_SHA256,'w2v_v35/storage.py':'new'}
        self.assertTrue(storage.compatible_code(old,current))
        self.assertTrue(storage.compatible_code(current,current))
        for changed in ({**current,'torch':'2.6.0'},{**current,'w2v_v35/losses.py':'different'},
                        {**current,'w2v_v35/cache.py':'different'},
                        {**current,'w2v_v35/train.py':'unknown'},{**current,'extra.py':'new'}):
            self.assertFalse(storage.compatible_code(old,changed))
        self.assertFalse(storage.compatible_code({**old,'w2v_v35/train.py':'unknown'},current))
        self.assertEqual(sha256(Path(training.__file__)),storage.PATCHED_TRAIN_SHA256)

    def test_peak_reserves_adam_before_its_first_step_and_two_winners(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2))
        model[0].requires_grad_(False)
        from .optim import EMA
        ema=EMA(model)
        weights=storage_size(model.state_dict())
        active=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
        self.assertEqual(storage.checkpoint_peak_bytes(model,ema),3*weights+3*active+3*storage.MARGIN)

    def test_space_fails_before_epoch_and_keeps_resume_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'last.pt').write_bytes(b'protected')
            with patch.object(storage.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(OSError,'before training'):
                    storage.check_space({},root,3,100)
            self.assertEqual((root/'last.pt').read_bytes(),b'protected')
            self.assertTrue((root/'storage_budget.json').is_file())

    def test_cache_budget_counts_full_two_views_and_only_obsolete_generation(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=cache_fixture(d)
            cache.prepare_base(cfg,records['train'],records['dev'])
            old=cache.prepare_epoch(cfg,2);current=cache.prepare_epoch(cfg,3)
            resolved,estimate=storage.generation_inventory(cfg,3)
            raw=json.loads((resolved/'sources.json').read_text())
            exact=sum(v['samples']*8 for v in raw.values() if v['domain']=='offline')
            self.assertGreater(estimate['expected_bytes'],exact)
            self.assertEqual(recovery.list_retired(cfg,3),[old])
            cache.retire_generations(cfg,[3])
            self.assertFalse(old.exists());self.assertTrue(current.exists())


class RecoveryTests(unittest.TestCase):
    def score_fixture(self,root):
        expected={};records=[]
        for condition in ('online','seen','heldout'):
            for label in (0,1):
                source=f'{condition}/{label}.wav';band=-1 if condition=='online' else 0
                expected[(condition,source)]=dict(label=label,language='en',band=band)
                z=torch.tensor([2.,-2.]) if label==0 else torch.tensor([-2.,2.])
                records.append(dict(condition=condition,source=source,label=label,language='en',band=band,
                                    logits=z.tolist(),p_fake=float(z.softmax(0)[0])))
        path=root/'epoch_3_scores.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        return path,expected,records

    def test_saved_scores_restore_metrics_and_reject_incomplete_or_mismatched_results(self):
        with tempfile.TemporaryDirectory() as d:
            path,expected,rows=self.score_fixture(Path(d))
            result=recovery.scores_to_metrics(path,expected)
            self.assertEqual(result['weighted_f1'],1.)
            self.assertEqual(result['groups']['seen/en']['recall'],[1.,1.])
            for invalid in (rows[:-1],rows+[rows[0]],[{**rows[0],'label':1}]+rows[1:]):
                path.write_text(''.join(json.dumps(r)+'\n' for r in invalid))
                with self.assertRaises(ValueError):recovery.scores_to_metrics(path,expected)

    def test_pending_ema_is_preserved_without_copy_and_survives_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg={'version':'3.5'}
            saved=dict(schema=SCHEMA,kind='weights',tag='epoch_3',epoch=3,config=cfg,model={'x':torch.tensor(3.)})
            atomic_save(root/'best_candidate.pt',saved)
            atomic_save(root/'best_candidate.pt.previous',{**saved,'epoch':2,'tag':'epoch_2'})
            state=dict(epoch=2,config=cfg)
            paths=recovery.rescue_pending_weights(root,state)
            import os
            self.assertTrue(os.path.samefile(paths[0],root/'best_candidate.pt'))
            class Controller:
                state={'best_selected':None,'best_train':{'tag':'epoch_2'}}
            training.recover_winners(root,Controller())
            self.assertEqual(training.read_state(root/'best_candidate.pt')['tag'],'epoch_2')
            self.assertEqual(training.read_state(paths[0])['tag'],'epoch_3')

    def test_recovery_command_prints_saved_dev_and_preserves_current_generation_and_last(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            root=Path(d).resolve();cfg,records=cache_fixture(root)
            cache.prepare_base(cfg,records['train'],records['dev'])
            cache.prepare_dev(cfg,[row for row in records['dev'] if row['domain']=='offline'])
            old=cache.prepare_epoch(cfg,2);current=cache.prepare_epoch(cfg,3)
            reference=root/'reference.pt';reference.write_bytes(b'protected original reference')
            cfg.update(version='3.5',reference_checkpoint=str(reference),reference_checkpoint_sha256=sha256(reference))
            run=root/'exp'/'w2v_v35_fixture';run.mkdir(parents=True)
            (run/'config.json').write_text(json.dumps(cfg))
            state=dict(schema=SCHEMA,kind='training',config=cfg,epoch=2,complete=False,
                model={'x':torch.tensor(2.)},ema={},source_hashes=training.source_fingerprints(),
                data_fingerprints={str(reference):sha256(reference)})
            atomic_save(run/'last.pt',state);last_sha=sha256(run/'last.pt')
            atomic_save(run/'best_candidate.pt',dict(schema=SCHEMA,kind='weights',config=cfg,
                tag='epoch_3',epoch=3,model={'x':torch.tensor(3.)}))
            expected=recovery.expected_scores(cfg)
            rows=[]
            for (condition,source),metadata in expected.items():
                z=torch.tensor([2.,-2.]) if metadata['label']==0 else torch.tensor([-2.,2.])
                rows.append(dict(condition=condition,source=source,label=metadata['label'],
                    language=metadata['language'],band=metadata.get('band',-1),
                    logits=z.tolist(),p_fake=float(z.softmax(0)[0])))
            (run/'epoch_3_scores.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with patch.object(recovery,'ROOT',root):
                recovery.recover(run,False)
                self.assertTrue(old.exists())
                self.assertFalse((run/'storage_recovery.json').exists())
                report=recovery.recover(run,True)
            self.assertEqual(report['recovered_dev']['weighted_f1'],1.)
            self.assertFalse(old.exists());self.assertTrue(current.exists())
            self.assertEqual(sha256(run/'last.pt'),last_sha)
            self.assertTrue((run/'epoch_3_recovered_metrics.json').is_file())
            self.assertTrue(Path(report['rescued_weights'][0]).is_file())

    def test_last_save_failure_keeps_previous_state_and_pending_metrics_then_replays_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,_,patches=fixture(root)
            cfg['joint_epochs']=1
            with patches:
                full=root/'full';training.train(cfg,full)
                expected=training.read_state(full/'last.pt')
                failed=root/'failed';real_save=training.atomic_save
                def disk_full(path,state):
                    if Path(path).name=='last.pt' and state['epoch']==2:
                        raise OSError('simulated insufficient checkpoint space')
                    real_save(path,state)
                with patch.object(training,'atomic_save',side_effect=disk_full):
                    with self.assertRaisesRegex(OSError,'insufficient checkpoint'):
                        training.train(cfg,failed)
                self.assertEqual(training.read_state(failed/'last.pt')['epoch'],1)
                measured=json.loads((failed/'epoch_2_pending.json').read_text())
                self.assertFalse(measured['checkpoint_committed'])
                self.assertEqual(measured['dev'],expected['history'][-1]['dev'])
                training.train(cfg,failed,failed/'last.pt')
                actual=training.read_state(failed/'last.pt')
                self.assertEqual(actual['history'],expected['history'])
                for name,value in expected['model'].items():
                    torch.testing.assert_close(actual['model'][name],value,rtol=0,atol=0)
                for name,value in expected['ema']['shadow'].items():
                    torch.testing.assert_close(actual['ema']['shadow'][name],value,rtol=0,atol=0)
                self.assertFalse((failed/'epoch_2_pending.json').exists())


if __name__=='__main__':unittest.main()
