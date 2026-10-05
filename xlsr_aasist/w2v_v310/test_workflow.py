"""Actual small w2v-BERT end-to-end training, recovery and submission packaging."""
from contextlib import contextmanager, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from w2v_v39.common import atomic_json, digest, read_json
from w2v_v39.test_workflow import source_fixture
from w2v_v39.model import ResidualClassifier
from w2v_v39.patch import save_patch
from .config import configuration, parser, verify_inputs
from .data import bundles, SourcePlan, loader
from .evaluate import export
from .model import load_model
from .state import SCHEMA, identity, partial_state, atomic_save, load_selected, apply_partial
from .train import run_experiment, infer
from .workflow import report
from .cleanup import cleanup


def fixture(root):
    original_source, old, model, records = source_fixture(root)
    source = root/'v39'
    source.mkdir()
    classifier = model.head.classifier[-1]
    save_patch(source/'best_patch.pt',old,'baseline',ResidualClassifier(classifier.weight,classifier.bias).spec())
    atomic_json(source/'completed.json',dict(version='3.9',status='complete',selected='baseline',
        baseline_fallback=True,language_debias_applied=False,base_checkpoint_sha256=old['base_checkpoint_sha256'],
        patch_sha256=digest(source/'best_patch.pt')))
    cfg = configuration(parser().parse_args(['--source-run',str(source),'--device','cpu','--workers','0','--epochs','1']))
    cfg.update(trainable_layers=1,adapter_hidden=8,adversary_hidden=8,probe_sources_per_language=4,
               probe_steps=8,probe_hidden=8,microbatch=4,eval_batch=4,feature_batch=4,
               disk_margin_bytes=0,min_gain=2.,patience=10,catastrophic_weighted_drop=2.)
    return source,cfg,model,records


@contextmanager
def synthetic_expanded_bundles(cfg):
    # Exercise multiple probe source groups using tiny synthetic waveforms, without
    # making the correctness suite download or process the real competition set.
    with bundles(cfg) as (train,dev):
        copies = 12
        rows = [dict(r,source_id=r['source_id']+f'_copy_{i}',group_id=r['group_id']+f'_copy_{i}')
                for i in range(copies) for r in train['rows']]
        yield dict(rows=rows,x=np.tile(train['x'],(copies,1)),logits=np.tile(train['logits'],(copies,1))),dev


class WorkflowTests(unittest.TestCase):
    def test_v39_unwrap_rejects_changed_identity_and_requires_real_finetuning(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            source,cfg,_,_ = fixture(Path(temp))
            self.assertEqual(cfg['source_version'],'3.9')
            verify_inputs(cfg)
            defaults = configuration(parser().parse_args(['--source-run',str(source),'--device','cpu']))
            self.assertEqual(defaults['trainable_layers'],8)
            self.assertEqual(defaults['epochs'],4)
            weights = source/'best_patch.pt'
            saved = torch.load(weights,map_location='cpu',weights_only=True)
            saved['config']['base_tag'] = 'wrong'
            torch.save(saved,weights)
            done = read_json(source/'completed.json')
            done['patch_sha256'] = digest(weights)
            atomic_json(source/'completed.json',done)
            with self.assertRaises(ValueError):
                configuration(parser().parse_args(['--source-run',str(source),'--device','cpu']))

    def test_real_training_commits_resume_then_exports_original_protocol_order(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source,cfg,_,records = fixture(root)
            base_hash = digest(cfg['base_checkpoint'])
            run = root/'v310'; run.mkdir(); atomic_json(run/'config.json',cfg)
            real_save = atomic_save
            first = True
            def interrupt_after_commit(path,value,margin):
                nonlocal first
                real_save(path,value,margin)
                if Path(path).name == 'last.pt' and first:
                    first = False
                    raise InterruptedError('simulated process exit just after a committed checkpoint')
            with patch('w2v_v310.train.bundles',synthetic_expanded_bundles), \
                 patch('w2v_v310.train.atomic_save',side_effect=interrupt_after_commit):
                with self.assertRaises(InterruptedError):
                    run_experiment(cfg,run)
            state = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
            self.assertEqual(state['cursor'],3)
            self.assertEqual(len(state['history']),1)
            with patch('w2v_v310.train.bundles',synthetic_expanded_bundles):
                with patch('w2v_v310.replay.check_initial',side_effect=AssertionError('Committed baseline must be reused')):
                    done = run_experiment(cfg,run)
            self.assertTrue(done['baseline_fallback'])
            self.assertEqual(done['completed_updates'],6)
            self.assertEqual(digest(cfg['base_checkpoint']),base_hash)
            self.assertFalse((run/'features').exists())
            self.assertFalse(list(run.glob('epoch_*.pt')))
            selected,_ = load_selected(run)
            self.assertIsNone(selected['model'])
            history = read_json(run/'training_history.json')
            self.assertEqual([e['cursor'] for e in history],[3,6])
            self.assertGreater(history[-1]['last_training_step']['reversal_strength'],0.)
            report(run)
            self.assertIn('baseline',(run/'report.md').read_text())
            # Exercise nonzero trained encoder/head/adapter deployment as well as fallback.
            current = load_model(cfg,training=False)
            saved = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
            apply_partial(current,saved['model'])
            selected.update(model=saved['model'],selected='test_actual_trained_weights')
            atomic_save(run/'best.pt',selected,0)
            done.update(selected='test_actual_trained_weights',baseline_fallback=False,checkpoint_sha256=digest(run/'best.pt'))
            atomic_json(run/'completed.json',done)
            rows = [records['dev'][i] for i in (9,0,6,3)]
            protocol = root/'protocol.txt'
            protocol.write_text('\n'.join(r['id'] for r in rows)+'\n',encoding='utf-8')
            expected,_ = infer(current,rows,cfg,'expected test predictions')
            archive = export(run,protocol,root/'audio',root/'submission','cpu',0)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(),['scores.txt'])
                output = [line.split() for line in z.read('scores.txt').decode().splitlines()]
            self.assertEqual([r[0] for r in output],[r['id'] for r in rows])
            np.testing.assert_allclose([float(r[1]) for r in output],torch.from_numpy(expected).softmax(-1)[:,0].numpy(),atol=1e-9,rtol=0)
            best_hash = digest(run/'best.pt')
            self.assertGreater(cleanup(run,False),0)
            self.assertTrue((run/'last.pt').exists())
            cleanup(run,True)
            self.assertFalse((run/'last.pt').exists())
            self.assertEqual(best_hash,digest(run/'best.pt'))
            self.assertEqual(base_hash,digest(cfg['base_checkpoint']))
            load_selected(run)

    def test_corrupt_dependency_and_runtime_cannot_be_silently_reused(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            _,cfg,_,_ = fixture(Path(temp))
            with patch('w2v_v310.config.runtime_versions',return_value={'torch':'changed'}):
                with self.assertRaisesRegex(ValueError,'runtime'):
                    verify_inputs(cfg)
            path = Path(cfg['v37_run'])/'features'/'train'/'x.npy'
            with path.open('r+b') as stream:
                stream.seek(-4,2); stream.write(b'bad!')
            with self.assertRaises(ValueError), bundles(cfg):
                self.fail('A corrupted original feature inventory must not be regenerated')

    def test_spawned_workers_use_numpy_transport_and_preserve_group_budget(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            _,cfg,_,records = fixture(Path(temp))
            plan = SourcePlan(records['train'],4,cfg['seed'])
            cfg['workers'] = 2
            batches = list(loader(plan,0,0,cfg))
            self.assertEqual(len(batches),plan.steps)
            for batch in batches:
                self.assertEqual(len(batch),16)
                self.assertTrue(all(isinstance(r['features'],np.ndarray) for r in batch))
                self.assertEqual({(r['language'],r['label']) for r in batch},{('en',0),('en',1),('zh',0),('zh',1)})


if __name__ == '__main__':
    unittest.main()
