"""Real AASIST CPU integration: trained parent, baseline gate, resume and export."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from w2v_v318.test_core import make_fixture,tiny_model as legacy_model,NativeBoundary
from w2v_v318.train import run_experiment as legacy_train
from .test_support import tiny_cfg,tiny_model
from .common import atomic_json,read_json,digest
from .parent import inspect_parent
from .train import run_experiment,save
from .evaluate import export,selected


def save_inputs(run,cfg,train,dev):
    run.mkdir();atomic_json(run/'train_rows.json',train);atomic_json(run/'dev_rows.json',dev)
    cfg['saved_manifests']={n:digest(run/n) for n in ('train_rows.json','dev_rows.json')}
    atomic_json(run/'config.json',cfg)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_trained_parent_reproduction_resume_and_export(self):
        with tempfile.TemporaryDirectory() as d,patch('w2v_v318.data.Engines',NativeBoundary),patch('w2v_v3183.data.Engines',NativeBoundary),redirect_stdout(io.StringIO()):
            root=Path(d);prior,train,dev=make_fixture(root/'audio');prior['epochs']=2
            parent=root/'parent';save_inputs(parent,prior,train,dev)
            legacy_train(prior,parent,train,dev,legacy_model)
            original_hash=digest(parent/'last.pt')
            old,weights,initialization=inspect_parent(parent,'epoch_2_step_3')
            del weights
            cfg=tiny_cfg()
            for key in ('dev_pairs','official_dev_protocol','official_dev_root','noise_records'):
                cfg[key]=prior[key]
            cfg['initialization']=dict(initialization,mode='CPU parent fixture',old_optimizer_loaded=False)
            full=root/'full';save_inputs(full,cfg,train,dev)
            ready=run_experiment(cfg,full,train,dev,tiny_model,baseline_only=True)
            self.assertEqual(ready['status'],'baseline_verified')
            self.assertEqual(read_json(full/'parent_baseline.json')['max_abs_logit_error'],0.)
            self.assertFalse((full/'completed.json').exists())
            with self.assertRaises(FileNotFoundError):selected(full,'best')
            run_experiment(cfg,full,train,dev,tiny_model)
            resumed=root/'resumed';save_inputs(resumed,cfg,train,dev)
            def interrupt(*args):
                save(*args)
                if args[2]['epoch']==1:raise RuntimeError('test interruption')
            with patch('w2v_v3183.train.save',interrupt),self.assertRaisesRegex(RuntimeError,'test interruption'):
                run_experiment(cfg,resumed,train,dev,tiny_model)
            with (resumed/'steps.jsonl').open('a') as f:f.write('{"update":999}\n{"broken":')
            run_experiment(cfg,resumed,train,dev,tiny_model)
            a=torch.load(full/'last.pt',weights_only=True);b=torch.load(resumed/'last.pt',weights_only=True)
            for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
            self.assertNotEqual(float(a['model']['fusion.gate']),0.)
            parent_state=torch.load(parent/'last.pt',weights_only=True)['model']
            for name,value in parent_state.items():
                if name.endswith(('running_mean','running_var','num_batches_tracked')):
                    torch.testing.assert_close(value,a['model'][name],rtol=0,atol=0)
            self.assertEqual([json.loads(s)['update'] for s in (resumed/'steps.jsonl').read_text().splitlines()],list(range(1,b['cursor']+1)))
            self.assertEqual(digest(parent/'last.pt'),original_hash)
            baseline=read_json(full/'parent_baseline.json');self.assertTrue(baseline['verified'])
            self.assertIn('parent_selection_delta',read_json(full/'training_history.json')[0])
            atomic_json(resumed/'training_history.json',[])
            run_experiment(cfg,resumed,train,dev,tiny_model)
            self.assertEqual(len(read_json(resumed/'training_history.json')),2)
            out=root/'dev-export'
            export(resumed,out,dev=True,device='cpu',workers=0,model_factory=tiny_model,verify=False)
            self.assertIn('fusion',read_json(out/'validation_meta.json'))
            self.assertIn('matched_95',read_json(out/'metrics.json')['raw']['groups']['online/en'])
            protocol=root/'eval.txt';ids=[r['id'] for r in dev if r['condition']=='online']
            protocol.write_text('\n'.join(ids)+'\n')
            export(resumed,root/'submission',protocol=protocol,audio_root=root/'audio'/'dev',
                   device='cpu',workers=0,model_factory=tiny_model,verify=False)
            self.assertEqual([s.split()[0] for s in (root/'submission'/'scores.txt').read_text().splitlines()],ids)
            # A changed base encoder cannot pass the startup reproduction gate.
            bad=root/'bad';save_inputs(bad,cfg,train,dev)
            def wrong_base(c):
                model=tiny_model(c)
                with torch.no_grad():next(model.frontend.parameters()).add_(2)
                return model
            with self.assertRaisesRegex(ValueError,'reproduction failed'):
                run_experiment(cfg,bad,train,dev,wrong_base)
            self.assertFalse((bad/'last.pt').exists())
            with self.assertRaisesRegex(ValueError,'Expected parent'):
                inspect_parent(parent,'epoch_10_step_2417')


if __name__=='__main__':unittest.main()
