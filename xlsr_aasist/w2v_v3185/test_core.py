import copy
from collections import Counter,defaultdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_v3182.test_support import metadata
from w2v_v3183r2.test_support import tiny_model as parent_model
from w2v_v3183r2.objective import unit_loss
from w2v_v3184.test_support import scene_fixture
from w2v_v318.test_core import make_fixture,NativeBoundary
from .test_support import tiny_cfg,tiny_model
from .views import recipe,augment,FAMILIES
from .sampling import Plan
from .regions import moments
from .parent import transfer
from .model import optimizer_for
from .step import schedule
from .common import partial_state,seed_all


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_waveforms_reproducible_distinct_and_bounded(self):
        t=np.arange(64000)/16000
        x=((.1+.08*np.sin(2*np.pi*2*t))*np.sin(2*np.pi*180*t)+.02*np.sin(2*np.pi*3100*t)).astype(np.float32)
        values=[]
        for v in range(5):
            r=recipe(13,'one-source','online',v);a,meta=augment(x,r);b,_=augment(x,r)
            np.testing.assert_array_equal(a,b)
            self.assertTrue(np.isfinite(a).all());self.assertLessEqual(float(abs(a).max()),.980001)
            self.assertEqual(meta['recipe']['family'],FAMILIES[v]);self.assertFalse(meta['independent_new_recording'])
            self.assertGreaterEqual(len(a),.85*len(x));self.assertLessEqual(len(a),len(x)/.97+1)
            values.append(a.tobytes())
        self.assertEqual(len(set(values)),5);np.testing.assert_array_equal(augment(x,recipe(13,'one-source','online',0))[0],x)
        self.assertEqual(recipe(1,'same','online',1),recipe(1,'same','online',1))

    def test_short_and_low_energy_crop_fallback(self):
        x=np.sin(np.arange(8000)*.03).astype(np.float32)*.05
        y,meta=augment(x,recipe(1,'short','offline',4))
        self.assertEqual(len(y),len(x));self.assertEqual(meta['fallback'],'short_recording_envelope')
        x=np.zeros(64000,np.float32);x[:4000]=.1
        r=dict(recipe(1,'sparse','offline',4),keep=.85,start=1.)
        y,meta=augment(x,r);self.assertEqual(len(y),len(x));self.assertEqual(meta['fallback'],'energy_preserving_envelope')
        with self.assertRaises(ValueError):augment(np.array([np.nan,1]),r)

    def test_four_group_and_two_stream_loss_mass(self):
        quotas=[4,3,5,4];total=0.
        for (lang,label),q in zip((('en',0),('en',1),('zh',0),('zh',1)),quotas):
            group=[]
            for kind in ('online','offline'):
                rows=[dict(role=r,language=lang,label=label,source_id='one',occurrence='same')
                      for r in (('online',) if kind=='online' else ('noisy_a','noisy_b'))]
                loss,_,_=unit_loss(torch.zeros(len(rows),2),rows,quotas,.25,0.)
                group.append(float(loss)*q)
            self.assertAlmostEqual(sum(group),np.log(2)*.25,places=6)
            self.assertAlmostEqual(group[0],group[1],places=6);total+=sum(group)
        self.assertAlmostEqual(total,np.log(2),places=6)

    def test_asp_preserves_parent_eval_and_std_learns(self):
        cfg=tiny_cfg();seed_all(37);old=parent_model(cfg).eval()
        with torch.no_grad():old.head.contribution[-1].weight.normal_(0,.04)
        inherited=partial_state(old);seed_all(37);new=tiny_model(cfg).eval();audit=transfer(new,inherited)
        self.assertTrue(audit['initial_function_preserved'])
        waves=[np.random.default_rng(i).normal(0,.05,n).astype(np.float32) for i,n in enumerate((8500,48000,92000))]
        with torch.no_grad():torch.testing.assert_close(old(waves),new(waves),atol=2e-6,rtol=2e-5)
        new.set_phase(True);optimizer=optimizer_for(new,cfg);new.train()
        logits=new(waves);torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,0])).backward()
        self.assertGreater(new.head.aasist.out_layer.weight.grad[:,160:].abs().sum(),0)
        self.assertGreater(new.head.attention_delta[-1].weight.grad.abs().sum(),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None))
        optimizer.step();self.assertGreater(new.head.aasist.out_layer.weight[:,160:].abs().sum(),0)
        names={g['name'] for g in optimizer.param_groups}
        self.assertEqual(names,{'lora','aasist','fusion','layer_mix','evidence','pooling','classifier'})

    def test_statistics_degenerate_and_attention_not_capped(self):
        for n in (1,3):
            x=torch.ones(n,160,requires_grad=True);w=torch.ones(n)/n
            mean,std=moments(x,w);torch.testing.assert_close(std,torch.zeros_like(std),rtol=0,atol=0)
            (mean.sum()+std.sum()).backward();self.assertTrue(torch.isfinite(x.grad).all())
        head=tiny_model(tiny_cfg()).head
        h=torch.randn(4,160,requires_grad=True);mass=torch.tensor([.1,.2,.3,.4])
        with torch.no_grad():head.attention_delta[-1].weight.normal_(0,20)
        w=head.weights(h,mass);self.assertAlmostEqual(float(w.detach().sum()),1.,places=6)
        self.assertTrue(bool(((w/mass)<1/3).any() or ((w/mass)>3).any()))
        perm=torch.tensor([2,0,3,1]);wp=head.weights(h[perm],mass[perm])
        torch.testing.assert_close(wp,w[perm]);a=moments(h,w);b=moments(h[perm],wp)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb)

    def test_schedule_and_mean_ablation(self):
        cfg=tiny_cfg();model=tiny_model(cfg);opt=optimizer_for(model,cfg)
        schedule(opt,cfg,0,2,3)
        for g in opt.param_groups:
            if g['name']=='lora':self.assertEqual(g['lr'],0)
            if g['name']=='pooling':self.assertEqual(g['lr'],cfg['pool_lr'])
        schedule(opt,cfg,1,2,3)
        for g in opt.param_groups:self.assertAlmostEqual(g['lr'],g['initial_lr']*cfg['lr_floor'])
        cfg['pooling']='mean';m=tiny_model(cfg)
        self.assertEqual(m.head.aasist.out_layer.in_features,160)
        self.assertNotIn('pooling',{g['name'] for g in optimizer_for(m,cfg).param_groups})

    def test_augmentation_before_rtc_shared_base_and_fixed_probes(self):
        from .data import Waves
        from w2v_v3184.data import Waves as OldWaves
        with tempfile.TemporaryDirectory() as d,patch('w2v_v3185.data.Engines',NativeBoundary):
            root=Path(d);cfg,train,dev=make_fixture(root/'audio');cfg.update(tiny_cfg());cfg.update(scene_fixture(root/'scenes'))
            data=Waves(train,cfg);index=next(i for i,r in enumerate(train) if r['condition']=='offline')
            t=dict(index=index,kind='offline',epoch=0,step=0,occurrence='x',base_variant=1)
            seen=[]
            def generate(x,r,*args):seen.append(x.copy());return x.copy(),dict(recipe=r)
            with patch('w2v_v3185.data.generate',generate):views=data[t]
            expected,_=augment(data.original(train[index]),recipe(cfg['seed'],train[index]['id'],'offline',1))
            self.assertEqual(len(seen),2)
            for x in seen:np.testing.assert_array_equal(x,expected)
            self.assertEqual(views[0]['base_augmentation'],views[1]['base_augmentation'])
            index=next(i for i,r in enumerate(train) if r['condition']=='online')
            probe=dict(index=index,kind='online',epoch=0,step=0,occurrence='probe',probe=True)
            np.testing.assert_array_equal(data[probe][0]['wave'],OldWaves(train,cfg)[probe][0]['wave'])
            self.assertNotIn('base_augmentation',data[probe][0])

    def test_default_settings_and_resume_override_guard(self):
        from .config import parser,validate,validate_locked_cli
        p=parser();args=p.parse_args([]);validate(args)
        self.assertEqual(args.group_quotas,[4,3,5,4]);self.assertEqual((args.view_factors,args.pooling),([5,5,1,3],'asp'));self.assertFalse(args.refresh_base_variants)
        self.assertEqual((args.epochs,args.warm_epochs),(6,1))
        argv=['--resume','x','--view-factors','5','5','1','3']
        with self.assertRaisesRegex(ValueError,'saved configuration'):validate_locked_cli(p.parse_args(argv),argv)


if __name__=='__main__':unittest.main()
