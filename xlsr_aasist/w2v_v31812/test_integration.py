"""Real waveform/cache parity and experiment orchestration gates."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from torch import nn
from .cache import extract
from .model import evidence, Residual, CachedDetector
from .workflow import execute
from .partition import full_sweep
from .metrics import report
from .test_experiment import data
from w2v_v31810.common import atomic_json, read_json
from w2v_v31810.test_partial_lora import detector
from w2v_v31810.data import Waves


class CacheTests(unittest.TestCase):
    def test_submission_preserves_protocol_order_short_inputs_and_one_classifier(self):
        from .evaluate import export
        from .workflow import code_identity
        from . import SCHEMA
        from w2v_v31810.common import fingerprint,digest,atomic_save
        from w2v_v3189.data import ready
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();audio=root/'audio';audio.mkdir()
            cfg=dict(schema=SCHEMA,code=code_identity(),artifacts={'parent.pt':'frozen-fixture'})
            atomic_json(run/'config.json',cfg)
            residual=Residual().eval()
            atomic_save(run/'residual.pt',dict(schema=SCHEMA,identity=fingerprint(cfg),parent_sha256='frozen-fixture',residual=residual.state_dict()))
            summary=dict(joint_eligible=True,residual_sha256=digest(run/'residual.pt'))
            atomic_json(run/'summary.json',summary)
            atomic_json(run/'completed.json',dict(identity=fingerprint(cfg),summary_sha256=digest(run/'summary.json')))
            protocol=root/'progress.txt';protocol.write_text('b.wav\na.wav\n')
            x=[np.sin(np.arange(250)*.02).astype(np.float32),np.cos(np.arange(6000)*.04).astype(np.float32)]
            for name,w in zip(('b.wav','a.wav'),x):sf.write(audio/name,w,16000,subtype='FLOAT')
            parent=detector(checkpoints=False).requires_grad_(False).eval()
            live=dict(seed=3,workers=0,eval_batch=2,microbatch=2,frame_budget=400,raw_audio_cache_mib=1)
            with torch.no_grad():expected=evidence(parent,[ready(w) for w in x],[len(w) for w in x],residual)[2].softmax(-1)[:,0].numpy()
            with patch('w2v_v31812.evaluate.load_parent',return_value=(parent,live)):
                archive=export(run,root/'out',protocol,audio,'cpu')
            self.assertTrue(archive.is_file())
            fields=[line.split() for line in (root/'out'/'scores.txt').read_text().splitlines()]
            self.assertEqual([f[0] for f in fields],['b.wav','a.wav'])
            np.testing.assert_allclose([float(f[1]) for f in fields],expected,rtol=1e-6,atol=1e-6)
            metadata=read_json(root/'out'/'submission_meta.json')
            self.assertEqual(metadata['final_classifiers'],1);self.assertFalse(metadata['ensemble'])
            summary['joint_eligible']=False;atomic_json(run/'summary.json',summary)
            atomic_json(run/'completed.json',dict(identity=fingerprint(cfg),summary_sha256=digest(run/'summary.json')))
            with self.assertRaisesRegex(ValueError,'not passed'):export(run,root/'bad',protocol,audio,'cpu')

    def test_real_decode_length_packing_cache_restart_and_identity(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[];tickets=[];rng=np.random.default_rng(19)
            for i,n in enumerate((5000,250,12000,3920)):
                p=root/f'{i}.wav';sf.write(p,rng.normal(0,.1,n),16000,subtype='PCM_16')
                rows.append(dict(id=str(i),audio=str(p),canonical_source=f'source{i}',source_id=str(i),group_id=str(i),
                    condition='online',split='train',language='en' if i%2 else 'zh',label=i%2))
                tickets.append(dict(index=i,kind='online',canonical_source=f'source{i}',occurrence=str(i)))
            cfg=dict(seed=19,workers=0,eval_batch=4,microbatch=2,frame_budget=200,raw_audio_cache_mib=1)
            parent=detector(checkpoints=False).eval().requires_grad_(False)
            result=extract(parent,rows,tickets,cfg,root/'cache','checkpoint-fixture')
            dataset=Waves(rows,cfg)
            for i,t in enumerate(tickets):
                row=dataset[t][0]
                g,l,z=evidence(parent,[row['wave']],[row['original_samples']])
                # Per-recording normalization; minor CPU GEMM batching roundoff only.
                np.testing.assert_allclose(result['global'][i],g[0].numpy(),rtol=1e-5,atol=2e-6)
                np.testing.assert_allclose(result['local'][i],l[0].numpy(),rtol=1e-5,atol=2e-5)
                np.testing.assert_allclose(result['logits'][i],z[0].numpy(),rtol=1e-5,atol=2e-5)
            self.assertEqual([r['id'] for r in result['rows']],['0','1','2','3'])
            self.assertTrue(np.all(result['global'][1]==0))
            with patch.object(parent,'encode',side_effect=AssertionError('cached shard must not encode')):
                again=extract(parent,rows,tickets,cfg,root/'cache','checkpoint-fixture')
            np.testing.assert_array_equal(result['logits'],again['logits'])
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                extract(parent,rows,tickets,cfg,root/'cache','other-checkpoint')

    def test_full_sweep_preserves_parent_variant_and_dsp_control_calls(self):
        class Plan:
            seed=7
            def view_spec(self,index,kind,view):return dict(base_variant=view,base_family='original' if view==0 else 'speed_slow')
            def _conditions(self,number,block,rank):return dict(scene='rain',family='webrtc'),91,dict(scene='rain',family='webrtc')
        rows=[dict(id='a',canonical_source='s1',condition='online'),dict(id='b',canonical_source='s1',condition='offline'),
              dict(id='duplicate',canonical_source='s1',condition='offline'),dict(id='off_only',canonical_source='s2',condition='offline')]
        tickets=full_sweep(rows,Plan(),1)
        self.assertEqual(len(tickets),3)
        for t in tickets:
            self.assertEqual(t['base_family'],'speed_slow')
            self.assertEqual(t['nuisance_seed'],91)
            self.assertEqual(t['domain_override'],None if t['kind']=='online' else dict(scene='rain',family='webrtc'))


class PipelineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_prepare_loads_real_parent_checkpoint_contract_and_independent_snapshot(self):
        from .workflow import parser,prepare,load_parent,_VERIFIED_INPUTS
        from .test_experiment import registry_rows
        from w2v_v31810.identity import build_registry,registry_config
        from w2v_v31810.common import RECIPE,schema_for,fingerprint,partial_state,atomic_save,digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'parent';source.mkdir();run=root/'new'
            model=detector(checkpoints=False).eval();rows,_=registry_rows()
            registry=build_registry(rows,[],{})
            protocol=dict(source_registry_hash=registry['registry_hash'])
            original=dict(model.cfg,version='3.18.10',recipe=RECIPE,omni_arch='3b',temporal_stats=True,
                consistency='none',pcl_noise_weight=0.,pcl_platform_weight=0.,pcl_variance_weight=0.,en_real_confidence_weight=0.,
                source_registry=registry_config(registry),validation_protocol_hash=fingerprint(protocol),source_draws_per_stream=[16]*4)
            for name,value in (('config',original),('train_rows',rows),('source_registry',registry),('validation_protocol',protocol)):
                atomic_json(source/(name+'.json'),value)
            weights=partial_state(model);tag='epoch_3'
            atomic_save(source/'last.pt',dict(schema=schema_for(original),identity=fingerprint(original),history=[dict(tag=tag)],
                best_tag=tag,last_tag=tag,best_model=weights,model=weights))
            atomic_json(source/'completed.json',dict(version='3.18.10',recipe=RECIPE,status='complete',
                checkpoint_sha256=digest(source/'last.pt'),best_tag=tag,last_tag=tag))
            args=parser().parse_args(['--parent-run',str(source),'--run',str(run),'--device','cpu','--train-device','cpu','--workers','0'])
            returned,cfg=prepare(args)
            self.assertEqual(returned,run)
            self.assertEqual(cfg['parent_tag'],tag)
            source.joinpath('last.pt').write_bytes(b'parent changed after safe snapshot')
            _VERIFIED_INPUTS.clear()
            with patch('w2v_v31812.workflow.verify_inputs') as verify,patch('w2v_v31812.workflow.load_model',side_effect=lambda cfg:detector(checkpoints=False)):
                loaded,_=load_parent(run,cfg);again,_=load_parent(run,cfg)
            self.assertEqual(verify.call_count,1)
            for k,v in weights.items():torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            resume=parser().parse_args(['--resume',str(run)])
            self.assertEqual(prepare(resume),(run,cfg))
            run.joinpath('parent.pt').write_bytes(b'corrupt')
            with self.assertRaises(Exception):prepare(resume)

    def fixture(self,root):
        fit=data();val=data();select=data();audit=data()
        for name,item in (('fit',fit),('inner',val),('select',select),('audit',audit)):
            for r in item['rows']:
                r['canonical_source']=name+'-'+r['canonical_source'];r.update(condition='online' if r['role']=='online' else 'offline',split='train')
        rows=fit['rows']+val['rows']
        split=dict(fit=sorted({r['canonical_source'] for r in fit['rows']}),validation=sorted({r['canonical_source'] for r in val['rows']}),interpretation='residual-only holdout')
        protocol={key:dict(rows=item['rows'],speech_sources=sorted({r['canonical_source'] for r in item['rows']})) for key,item in (('select',select),('audit',audit))}
        for name,value in (('train_rows',rows),('validation_protocol',protocol),('residual_split',split)):atomic_json(root/(name+'.json'),value)
        cfg=dict(schema='fixture',cache_views=1,seeds=[1,2,3],train_device='cpu',backend_precision='fp32',artifacts={'parent.pt':'fixture'})
        class Parent(nn.Module):
            def __init__(self):
                super().__init__();self.head=nn.Module();self.head.aasist=nn.Module();self.head.aasist.out_layer=nn.Linear(384,2)
        baseline=report(val['rows'],val['logits'])
        def fake_fit(kind,train,validation,classifier,cfg,directory,seed,identity,stop):
            model=CachedDetector(classifier,Residual());model.seed=seed
            return model,dict(rank=[1,1-seed/100],model=model.state_dict(),metrics=baseline,epoch=seed,guard=dict(passed=True))
        def fixed(parent,part,live,run,cfg,name,identity,stop):return {'train_inner':val,'dev_select':select,'dev_audit':audit}[name]
        def predict(model,item,device):
            z=item['logits'].copy()
            for i,r in enumerate(item['rows']):
                if r['label']==1:z[i]=[-1,1]
            return z
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch('w2v_v31812.workflow.load_parent',return_value=(Parent(),dict(seed=1,scene_noise_records={'train':[]}))))
        stack.enter_context(patch('w2v_v31810.sampling.Plan'))
        stack.enter_context(patch('w2v_v31812.workflow.full_sweep',return_value=[dict(canonical_source=split['fit'][0])]))
        stack.enter_context(patch('w2v_v31812.workflow.extract',return_value=fit))
        stack.enter_context(patch('w2v_v31812.workflow._panels',return_value={}))
        stack.enter_context(patch('w2v_v31812.workflow.fixed_cache',side_effect=fixed))
        fitter=stack.enter_context(patch('w2v_v31812.workflow.fit',side_effect=fake_fit))
        stack.enter_context(patch('w2v_v31812.workflow.predict',side_effect=predict))
        return stack,cfg,fitter

    def test_stage1_negative_stops_before_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stack,cfg,fitter=self.fixture(root)
            with stack,patch('w2v_v31812.workflow.review_global',return_value=False):execute(root,cfg)
            self.assertEqual([call.args[0] for call in fitter.call_args_list],['global','local'])
            summary=read_json(root/'summary.json')
            self.assertEqual(summary['decision'],'stop_feature_review_screen')
            self.assertFalse((root/'residual.pt').exists());self.assertTrue((root/'report.zip').exists())

    def test_stage2_three_seeds_train_only_choice_and_selected_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stack,cfg,fitter=self.fixture(root)
            with stack,patch('w2v_v31812.workflow.review_global',return_value=True):execute(root,cfg)
            self.assertEqual([call.args[0] for call in fitter.call_args_list],['global','local','residual','residual','residual'])
            summary=read_json(root/'summary.json')
            self.assertEqual(summary['chosen_seed'],1)
            self.assertTrue(summary['joint_eligible']);self.assertFalse(summary['joint_training_launched'])
            self.assertTrue((root/'audit_predictions.npz').exists())
            state=torch.load(root/'residual.pt',weights_only=True)
            self.assertTrue(all(not k.startswith('classifier') for k in state['residual']))
            with patch('w2v_v31812.workflow.load_parent',side_effect=AssertionError('completed run must not repeat')):execute(root,cfg)


if __name__=='__main__':unittest.main()
