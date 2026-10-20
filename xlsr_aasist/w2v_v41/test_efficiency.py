"""Numerical and state contracts for the execution-only V4.1 changes."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor
from w2v_v4.step import train_step as original_step
from w2v_v4.gradients import coordinate
from .features import OfficialFeatures, FeatureCache
from .gradients import GradientWorkspace
from .step import train_step
from .test_core import Tiny, fixture, cfg
from .test_model import tiny_model
from .memory import envelope, select_checkpoints


class EfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_worker_features_reconstruct_entire_official_batch_including_odd_tail(self):
        features=OfficialFeatures(SeamlessM4TFeatureExtractor(padding_value=1.))
        waves=[np.random.default_rng(n).normal(size=n).astype(np.float32) for n in (1,123,3920,4080,4500,6500,7001,16000)]
        rows=[]
        for wave in waves:
            value,length=features.one(wave);rows.append(dict(mel=value,mel_length=length))
        expected,mask,lengths=features(waves)
        actual,other,counts=features.pack(rows,'cpu')
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(mask,other,rtol=0,atol=0);self.assertEqual(lengths,counts)

    def test_static_cache_is_bounded_and_dynamic_views_never_reused(self):
        feature=OfficialFeatures(SeamlessM4TFeatureExtractor());cache=FeatureCache(feature,1)
        x=np.random.default_rng(1).normal(size=4240).astype(np.float32)
        row=dict(id='a',wave=x,original_samples=len(x))
        a=cache.attach(dict(row),True);b=cache.attach(dict(row),True)
        self.assertEqual(cache.hits,1);self.assertIs(a['mel'],b['mel'])
        c=cache.attach(dict(row,wave=x*.4),False)
        self.assertEqual(cache.hits,1);self.assertEqual(cache.misses,2)
        self.assertLessEqual(cache.bytes,cache.cap)
        zero=FeatureCache(feature,0);zero.attach(dict(row),True);self.assertFalse(zero.values)

    def test_old_and_new_adam_updates_match_with_dual_pcl_pcgrad(self):
        a=Tiny();b=copy.deepcopy(a);settings=cfg();units=fixture()
        oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
        old_storage=None
        for step in range(3):
            sa=original_step(a,oa,units,settings,dict(official=.03,noise=.02),2,step,4)
            sb=train_step(b,ob,units,settings,dict(official=.03,noise=.02),2,step,4)
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=2e-7,rtol=2e-6)
            self.assertAlmostEqual(sa['loss'],sb['loss'],places=6)
            self.assertEqual(sa['pcl_pairs'],sb['pcl_pairs'])
            self.assertEqual(sa['coordination']['applied'],sb['coordination']['applied'])
            ptr=ob._v41_workspace.storage.data_ptr()
            if old_storage is not None:self.assertEqual(old_storage,ptr)
            old_storage=ptr

    def test_flat_pcgrad_preserves_original_vectors_and_nonfinite_handling(self):
        for c,n,a in (([1.,0.],[-1.,1.],[.1,.2]),([0.,0.],[0.,1.],[0.,0.]),([float('inf'),0.],[1.,1.],[0.,0.])):
            params=[torch.nn.Parameter(torch.zeros(2))];ws=GradientWorkspace(params);ws.reset()
            for kind,values in (('online',c),('noisy',n),('official',a)):
                ws.add(kind,[torch.tensor(values)*256])
            observed=ws.coordinate(True,256)
            expected,stats=coordinate([torch.tensor(c)*256],[torch.tensor(n)*256],[torch.tensor(a)*256],scale=256)
            torch.testing.assert_close(params[0].grad,expected,rtol=2e-6,atol=1e-6,equal_nan=True)
            self.assertEqual(bool(observed[0]),stats['finite'])

    def test_native_forward_and_gradients_match_precomputed_feature_path(self):
        a,_=tiny_model();b,_=tiny_model();b.load_state_dict(a.state_dict())
        waves=[np.random.default_rng(n).normal(0,.1,n).astype(np.float32) for n in (4080,7001)]
        rows=[]
        for wave in waves:
            mel,length=b.features.one(wave);rows.append(dict(wave=wave,mel=mel,mel_length=length))
        for model in (a,b):model.set_phase(True);model.train()
        torch.manual_seed(41);za=a(waves);za.square().sum().backward()
        torch.manual_seed(41);zb=b.forward_rows(rows);zb.square().sum().backward()
        torch.testing.assert_close(za,zb,atol=0,rtol=0)
        for (_,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7)

    def test_memory_shapes_are_upper_bounds_and_cpu_never_guesses_reduction(self):
        settings=cfg(microbatch=4,frame_budget=9000,whole_max_frames=3000,checkpoint_policy='auto')
        shapes=envelope(settings);self.assertEqual(shapes,[(3,3000),(4,2250)])
        model,_=tiny_model();model.set_phase(True);optimizer=torch.optim.AdamW(model.parameters())
        state=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
        result=select_checkpoints(model,optimizer,settings)
        self.assertFalse(result['measured']);self.assertEqual(result['selected_layers'],2)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,value in state.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)

    def test_partial_recomputation_keeps_all_lora_gradients_and_rng(self):
        model,_=tiny_model(depth=3);model.set_phase(True);model.train()
        waves=[np.random.default_rng(8).normal(0,.1,6000).astype(np.float32)]
        expected=None
        for count in (3,2,1,0):
            model.zero_grad(set_to_none=True);model.set_checkpoint_count(count);torch.manual_seed(41)
            value=model(waves);value.square().sum().backward()
            gradients={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
            if expected is None:expected=(value.detach(),gradients,torch.get_rng_state().clone())
            else:
                torch.testing.assert_close(value,expected[0],rtol=0,atol=0)
                self.assertEqual(set(gradients),set(expected[1]))
                for k,v in gradients.items():torch.testing.assert_close(v,expected[1][k],rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(torch.get_rng_state(),expected[2],rtol=0,atol=0)

    def test_probe_computes_backward_without_touching_optimizer_or_parameters(self):
        from .memory import _probe
        from w2v_v31810.test_meta import assert_state_equal
        model,settings=tiny_model();model.set_phase(True);model.train()
        settings.update(consistency='dual',gradient_coordination='pcgrad',grad_clip=1.)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        params=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict())
        _probe(model,optimizer,settings,[(3,11),(4,12)])
        assert_state_equal(self,state,optimizer.state_dict())
        assert_state_equal(self,params,model.state_dict())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_spawn_workers_extract_identical_features(self):
        import soundfile as sf
        from .data import Waves
        from w2v_v31810.workers import loader
        from w2v_v3184.data import close
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);assets=root/'features';SeamlessM4TFeatureExtractor().save_pretrained(assets)
            rows=[]
            for i,n in enumerate((4080,7001,6500,9000)):
                path=root/f'{i}.wav';sf.write(path,np.random.default_rng(i).normal(0,.1,n),16000,subtype='FLOAT')
                rows.append(dict(id=str(i),audio=str(path),condition='online'))
            settings=cfg(worker_features=True,feature_cache_mib=1,bert_model_dir=str(assets),workers=0)
            direct=[Waves(rows,settings)[i][0] for i in range(len(rows))]
            settings['workers']=2;batches=loader(Waves(rows,settings),settings,batch_size=2)
            try:observed=[r for batch in batches for unit in batch for r in unit]
            finally:close(batches)
            self.assertEqual([r['id'] for r in observed],[r['id'] for r in direct])
            for a,b in zip(direct,observed):
                torch.testing.assert_close(a['mel'],b['mel'],rtol=0,atol=0)
                self.assertEqual(a['mel_length'],b['mel_length'])

    def test_base_augmentation_is_computed_once_with_identical_noisy_pcl(self):
        from .data import Waves
        from w2v_v4.data import Waves as OldWaves
        from w2v_v3189.views import augment
        row=dict(id='off',audio='unused',split='train',condition='offline',language='en',label=1,canonical_source='s',group_id='s')
        wave=np.random.default_rng(41).normal(0,.1,8000).astype(np.float32)
        pair=dict(offline='off',source='s',matches=[[600,4000,600,4000]])
        ticket=dict(index=0,kind='offline',epoch=2,occurrence='x',base_variant=1,base_view_id='x',pcl_noise=pair)
        def generate(x,*args):return x.copy(),dict(pair_eligible=True,erased_spans=[])
        outputs=[];counts=[]
        for cls,module,base_module in ((OldWaves,'w2v_v4.data','w2v_v31810.data'),(Waves,'w2v_v41.data','w2v_v41.wave_source')):
            data=cls([row],cfg());data.bank=data.engines=object()
            with patch.object(data,'original',return_value=wave),patch(base_module+'.generate',side_effect=generate),patch(base_module+'.make_recipe',return_value={}),patch(base_module+'.augment',wraps=augment) as base:
                if cls is OldWaves:
                    with patch(module+'.augment',wraps=augment) as repeat:out=data[ticket];calls=base.call_count+repeat.call_count
                else:out=data[ticket];calls=base.call_count
            outputs.append(out[0]);counts.append(calls)
        self.assertEqual(counts,[2,1]);np.testing.assert_array_equal(outputs[0]['wave'],outputs[1]['wave'])
        self.assertEqual(outputs[0]['pcl']['matches'],outputs[1]['pcl']['matches'])
        self.assertEqual(outputs[0]['pcl_coverage'],outputs[1]['pcl_coverage']);self.assertNotIn('_base_wave',outputs[1])


if __name__=='__main__':unittest.main()
