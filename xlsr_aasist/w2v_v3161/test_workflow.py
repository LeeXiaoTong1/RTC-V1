"""Train actual tiny SSL, interrupt/replay, then export the actual selected weights."""
import copy
from contextlib import contextmanager,ExitStack,redirect_stdout
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from w2v_v316_tfcl.test_core import model_fixture,examples,config
from w2v_v39.common import atomic_json,read_json,digest
from w2v_v39.metrics import GROUPS
from . import train as training,evaluate as evaluation,state as checkpoint_state
from .state import partial_state,to_cpu,capture_rng,load_resume,load_selected


def score(w):
    return dict(complete=True,weighted_f1=w,clean_f1=w,noisy_f1=w,
        groups={g:dict(recall=[.99,.85],macro_f1=w,ap=.99,auc=.99,eer=.04) for g in GROUPS})


class Fixture:
    def __init__(self,root):
        self.root=Path(root);torch.manual_seed(45161);random.seed(45161);np.random.seed(45161)
        self.base=model_fixture()
        # Nonempty source Adam moments, as in a real continuation.
        opt=self.optimizer(self.base,{},None)
        sum(p.sum() for p in self.base.parameters() if p.requires_grad).backward();opt.step();opt.zero_grad()
        self.dev=[dict(id=f'{c}:{l}:{y}',source_id=f'{l}:{y}',group_id=f'{l}:{y}',
            condition=c,language=l,label=y,split='dev',audio_sha256=f'{c}:{l}:{y}',source_sha256=f'{c}:{l}:{y}')
            for c in ('online','seen','heldout') for l in ('en','zh') for y in (0,1)]
        source=self.root/'source';source.mkdir();atomic_json(source/'dev_rows.json',self.dev)
        self.cfg=dict(config(),epochs=2,seed=123,tfcl_heads=2,tfcl_bins=21,workers=0,feature_workers=0,
            source_run=str(source),source_last_tag='epoch_4_step_2',source_committed_updates=8,
            source_checkpoint_sha256='fixture',sampling_epoch_offset=4,matched_fake_recall=.99,
            disk_margin_bytes=0,free_reserve_bytes=0,maximum_new_peak_bytes=1024**3,
            lr_warmup_fraction=.025,checkpointing=False,tfcl_structure_weight=.15)
        self.source=dict(model=partial_state(self.base),optimizer=to_cpu(opt.state_dict()),rng=capture_rng(),
            last_tag='epoch_4_step_2',history=[dict(tag='epoch_4_step_2',metrics=score(.8),committed=True)])
        self.best=dict(tag='source:epoch_4_step_2',model=copy.deepcopy(self.source['model']),metrics=score(.8),
            origin=dict(run=str(source),version='3.16',tag='epoch_4_step_2'))
        self.seen=[]

    @staticmethod
    def optimizer(model,cfg,aux):
        groups=[dict(params=[p for p in model.parameters() if p.requires_grad],name='detector',lr=1e-4,initial_lr=1e-4)]
        if aux is not None:groups.append(dict(params=list(aux.parameters()),name='training_only_tfcl',lr=1e-4,initial_lr=1e-4))
        return torch.optim.AdamW(groups)

    def run(self,name):
        p=self.root/name;p.mkdir();atomic_json(p/'config.json',self.cfg);return p

    def model(self,*args,**kwargs):return copy.deepcopy(self.base)

    @contextmanager
    def patched(self,interrupt=None):
        owner=self
        @contextmanager
        def bundles(cfg):yield dict(rows=[]),dict(rows=owner.dev)
        class Plan:
            def __init__(self,*args):self.steps=2
            def coverage(self,epoch):return dict(epoch=epoch)
        class Loader:
            def __init__(self,*args):pass
            def segment(self,epoch,start,stop):
                for step in range(start,stop):
                    owner.seen.append((epoch,step))
                    if interrupt==(epoch,step):raise RuntimeError('interruption fixture')
                    rows=examples()
                    for r in rows:r['features']+=random.random()*.01+np.random.random()*.01
                    yield rows
            def close(self):pass
        def execution(model,aux,opt,plan,cfg,run):
            value=dict(selected=dict(microbatch=12,frame_budget=1000,checkpointing=False))
            atomic_json(Path(run)/'execution_plan.json',value);return value
        def infer(model,rows,cfg,label,run):
            return np.full((len(rows),2),.9 if 'epoch_5_' in label else .85,np.float32)
        replacements=dict(verify_inputs=lambda cfg:None,load_model=self.model,optimizer_for=self.optimizer,
            bundles=bundles,SourcePlan=Plan,TrainingLoader=Loader,select_execution=execution,
            source_state=lambda cfg:(copy.deepcopy(self.source),copy.deepcopy(self.best)),infer=infer,
            measure=lambda rows,z,**kwargs:score(float(z[0,0])))
        with ExitStack() as stack:
            for name,value in replacements.items():stack.enter_context(patch.object(training,name,value))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield


class WorkflowTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def assert_nested_equal(self,left,right):
        if torch.is_tensor(left):torch.testing.assert_close(left,right,atol=0,rtol=0)
        elif isinstance(left,dict):
            self.assertEqual(set(left),set(right))
            for k in left:self.assert_nested_equal(left[k],right[k])
        elif isinstance(left,(tuple,list)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.assert_nested_equal(a,b)
        else:self.assertEqual(left,right)

    def test_additional_epochs_resume_exact_and_best_is_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);full,part=f.run('full'),f.run('part')
            with f.patched():training.run_experiment(f.cfg,full)
            with f.patched(interrupt=(5,1)):
                with self.assertRaisesRegex(RuntimeError,'interruption'):training.run_experiment(f.cfg,part)
            self.assertEqual(load_resume(part,f.cfg)['cursor'],2)
            with f.patched():training.run_experiment(f.cfg,part)
            left,right=load_resume(full,f.cfg),load_resume(part,f.cfg)
            for key in ('model','auxiliary','optimizer','rng','cursor','best_tag','last_tag'):
                self.assert_nested_equal(left[key],right[key])
            self.assertEqual(right['cursor'],4);self.assertEqual(right['last_tag'],'epoch_6_step_2')
            self.assertEqual(right['best_tag'],'epoch_5_step_2')
            selected,meta=load_selected(part,'best');self.assertFalse(meta['baseline_fallback'])
            protocol=Path(folder)/'progress.txt';protocol.write_text('fixture\n')
            def infer(model,rows,cfg,label,run):
                for name,value in selected['candidate']['state'].items():
                    torch.testing.assert_close(dict(model.named_parameters())[name],value,atol=0,rtol=0)
                return np.array([[1.,0.],[0.,1.]],np.float32)
            with patch.object(evaluation,'verify_inputs',lambda cfg:None),patch.object(evaluation,'load_model',f.model),\
                 patch.object(evaluation,'read_protocol',return_value=[dict(id='a'),dict(id='b')]),\
                 patch.object(evaluation,'infer',infer),redirect_stdout(io.StringIO()):
                out=Path(folder)/'submission';archive=evaluation.export(part,protocol,folder,out,device='cpu',workers=0)
            self.assertEqual(read_json(out/'submission_meta.json')['selected'],'epoch_5_step_2')
            with zipfile.ZipFile(archive) as z:self.assertEqual(z.namelist(),['scores.txt'])

    def test_source_trained_best_excludes_historical_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt';cfg=dict(source_run=folder,source_checkpoint=str(path),
                source_checkpoint_sha256='hash',source_last_tag='epoch_4_step_2')
            saved=dict(last_tag='epoch_4_step_2',model={'x':torch.tensor([4.])},optimizer={'state':{0:{}}},
                history=[dict(tag='epoch_4_step_2',committed=True,metrics=score(.85)),
                         dict(tag='epoch_2_step_2',committed=True,metrics=score(.9))],
                selections=dict(best_weighted='epoch_2_step_2',best_guarded='starting_parent'),
                candidates={'epoch_2_step_2':dict(kind='partial',state={'x':torch.tensor([2.])}),
                            'starting_parent':dict(kind='partial',state={'x':torch.tensor([99.])})})
            torch.save(saved,path)
            with patch.object(checkpoint_state,'source_selected',return_value=({},dict(checkpoint_sha256='hash',selected='epoch_4_step_2'))):
                source,best=checkpoint_state.source_state(cfg)
            self.assertEqual(float(source['model']['x']),4.);self.assertEqual(float(best['model']['x']),2.)
            self.assertEqual(best['tag'],'source:epoch_2_step_2')


if __name__=='__main__':unittest.main()
