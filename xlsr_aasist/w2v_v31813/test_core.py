import copy
from collections import Counter,defaultdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from w2v_v3186.test_sampling import rows_for
from w2v_v31810.sampling import Plan as BasePlan
from w2v_v31810.common import atomic_json,fingerprint,capture_rng
from w2v_v31810.test_meta import assert_state_equal
from .objectives import paired_margin,phoneme_mse,ramp
from .pairing import pair_batches,source_fields,Plan
from .prior import neighbors,histogram,originals,bind,random_neighbors
from .step import train_step,calibrate,packs


def fixture():
    units=[]
    for role in ('online','noisy'):
        for lang in ('en','zh'):
            for label in (0,1):
                i=len(units);x=np.random.default_rng(i+5).normal(0,.2+label*.3,4240).astype(np.float32)
                row=dict(wave=x,role=role,language=lang,label=label,split='train',canonical_source=str(i),
                    xeus_pair=role+'/'+lang,original_samples=len(x))
                if role=='online':
                    row['original_pcl']=dict(refs=[dict(wave=x,original_samples=len(x)),dict(wave=x*.7,original_samples=len(x))],
                        matches=[[0,3000,0,3000]])
                units.append([row])
    return units


class Tiny(nn.Module):
    def __init__(self,cfg=None):
        super().__init__();self.fusion=nn.Linear(3,5);self.head=nn.Linear(5,2)
    def encode(self,waves):
        values=[]
        for wave in waves:
            x=torch.as_tensor(wave,dtype=torch.float32)[:3840].reshape(12,320)
            f=torch.stack((x.mean(1),x.square().mean(1),x.abs().mean(1)),1)
            values.append(self.fusion(f).tanh())
        return values
    def forward(self,waves):return self.head(torch.stack([h.mean(0) for h in self.encode(waves)]))
    def set_phase(self,joint):pass


def cfg(**kw):
    value=dict(device='cpu',backend_precision='fp32',microbatch=4,frame_budget=6000,prior_margin=.5,
        label_smoothing=.02,mode='full',grad_clip=100.,seed=3181301)
    value.update(kw);return value


class ObjectiveTests(unittest.TestCase):
    def test_pair_gradient_depends_on_partner_and_separates_classes(self):
        r=torch.tensor([.2,0.],requires_grad=True);f=torch.tensor([.4,0.],requires_grad=True)
        loss=paired_margin(r,f);gr,gf=torch.autograd.grad(loss,(r,f))
        self.assertGreater(gr[0],0);self.assertLess(gf[0],0)
        f2=torch.tensor([3.,0.]);g2=torch.autograd.grad(paired_margin(r,f2),r)[0]
        self.assertFalse(torch.allclose(gr,g2))

    def test_pcl_masks_reflection_and_backpropagates_to_both_originals(self):
        a=torch.zeros(12,3,requires_grad=True);b=torch.ones(12,3,requires_grad=True)
        loss,n=phoneme_mse(a,b,[[0,1040,0,1040]],1040,1040);loss.backward()
        self.assertEqual(n,1);self.assertEqual(float(loss.detach()),1.)
        self.assertGreater(float(a.grad[:3].abs().sum()),0);self.assertGreater(float(b.grad[:3].abs().sum()),0)
        self.assertEqual(float(a.grad[3:].abs().sum()),0)
        invalid,count=phoneme_mse(a,b,[[0,300,0,300]],300,300)
        self.assertEqual(count,0);self.assertEqual(float(invalid.detach()),0.)

    def test_atomic_pairs_and_microbatch_gradient_equivalence(self):
        units=fixture();a=Tiny();b=copy.deepcopy(a)
        for batch in packs([r for u in units for r in u],cfg(microbatch=2)):
            self.assertEqual(len(batch),2);self.assertEqual(len({r['xeus_pair'] for r in batch}),1)
        oa=torch.optim.SGD(a.parameters(),lr=.05);ob=torch.optim.SGD(b.parameters(),lr=.05)
        sa=train_step(a,oa,units,cfg(microbatch=2),dict(prior=.1,pcl=.01),3,0,2)
        sb=train_step(b,ob,units,cfg(microbatch=8),dict(prior=.1,pcl=.01),3,0,2)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(sa['loss'],sb['loss'],places=6)
        self.assertEqual(sa['xeus_pairs'],4);self.assertEqual(sa['pcl_sources'],4)

    def test_calibration_has_no_updates_or_rng_side_effects(self):
        model=Tiny().train();before=copy.deepcopy(model.state_dict());rng=capture_rng()
        result=calibrate(model,[fixture(),fixture()],cfg())
        for name,x in before.items():self.assertTrue(torch.equal(x,model.state_dict()[name]))
        assert_state_equal(self,rng,capture_rng());self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertGreater(result['weights']['prior'],0);self.assertLessEqual(result['weights']['prior'],.1)
        self.assertGreater(result['weights']['pcl'],0);self.assertLessEqual(result['weights']['pcl'],.01)

    def test_ramp_and_missing_partners_fail(self):
        self.assertEqual(ramp(0,9,10),0);self.assertEqual(ramp(1,9,10),.5);self.assertEqual(ramp(2,9,10),1)
        rows=[r for u in fixture() for r in u][1:]
        with self.assertRaises(ValueError):list(packs(rows,cfg()))


class SamplingTests(unittest.TestCase):
    def test_exact_budget_source_isolation_replay_and_cap(self):
        rows=rows_for((23,23,31,23));settings=dict(source_draws_per_stream=[113,89,89,89],consistency='none',experiment='E0')
        base=BasePlan(rows,cfg=settings);before=list(base.batches(1));neighbors={}
        for row,source in zip(rows,base.canonical):
            if row['condition']=='online' and row['label']==1:
                neighbors[source]=[dict(source=s,similarity=.95) for r,s in zip(rows,base.canonical)
                    if r['condition']=='offline' and r['language']==row['language'] and r['label']==0]
        after,audit=pair_batches(before,rows,neighbors,1)
        self.assertEqual(after,pair_batches(before,rows,neighbors,1)[0]);self.assertTrue(audit['pairs'])
        def signature(batches):return Counter((t['kind'],fingerprint(source_fields(t))) for b in batches for t in b)
        self.assertEqual(signature(before),signature(after));self.assertLessEqual(audit['max_source_uses'],2)
        for old,new in zip(before,after):
            self.assertEqual(len(new),len({t['canonical_source'] for t in new}))
            self.assertEqual(Counter((t['kind'],rows[t['index']]['language'],rows[t['index']]['label']) for t in old),
                             Counter((t['kind'],rows[t['index']]['language'],rows[t['index']]['label']) for t in new))
            pairs=defaultdict(list)
            for t in new:
                if 'xeus_pair' in t:pairs[t['xeus_pair']].append(t)
            for pair in pairs.values():
                self.assertEqual(len(pair),2)
                if pair[0]['kind']=='offline':
                    for name in ('domain_override','nuisance_seed','condition_rank'):self.assertEqual(pair[0][name],pair[1][name])

    def test_offset_once_warmup_no_aux_and_original_only_pcl(self):
        rows=rows_for((23,23,23,23));settings=cfg(source_draws_per_stream=[24]*4,consistency='none',experiment='E0',
            stream_sources=24,group_quotas=[6]*4,view_factors=[5,5,1,3],refresh_base_variants=True,source_epoch_offset=15)
        phones={r['id']:dict(online=r['id']) for r in rows if r['condition']=='online'}
        plan=Plan(rows,settings,{},phones);new=plan.epoch_batches(0)
        base=list(BasePlan(rows,cfg=settings,seed=settings['seed']).batches(0))
        self.assertEqual(new,base);self.assertFalse(any('official_pcl' in t for b in new for t in b))
        new=plan.epoch_batches(1)
        for batch in new:
            selected=[t for t in batch if 'official_pcl' in t]
            self.assertEqual(len(selected),4);self.assertTrue(all(t['kind']=='online' for t in selected))


class PriorTests(unittest.TestCase):
    def test_histograms_ignore_length_scale_and_neighbors_cross_only(self):
        centers=np.eye(3,dtype=np.float32);x=np.array([[1,0,0],[1,0,0],[0,1,0]],np.float32)
        np.testing.assert_allclose(histogram(x,centers),histogram(np.tile(x,(2,1)),centers))
        rows=[];h=[]
        for lang in ('en','zh'):
            for label in (0,1):
                for i in range(3):
                    rows.append(dict(language=lang,label=label,canonical_source=f'{lang}{label}{i}',output_samples=16000,condition='offline',split='train',id=str(len(rows))))
                    h.append(np.array([1.,0,0]) if i<2 else np.array([0.,1,0]))
        pairs,quality=neighbors(rows,np.stack(h))
        identities={r['canonical_source']:(r['language'],r['label']) for r in rows}
        for real,items in pairs.items():
            for f in items:self.assertEqual(identities[f['source']],(identities[real][0],0))
        self.assertEqual(quality['en']['coverage'],1.)
        with self.assertRaises(ValueError):originals([dict(rows[0],split='dev')])

    def test_cache_rejects_wrong_inventory_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[dict(id='a',canonical_source='a',language='en',label=1,split='train',condition='offline')]
            atomic_json(root/'manifest.json',dict(complete=True,owner=dict(rows='wrong'),files=[]))
            with self.assertRaises(ValueError):bind(root/'manifest.json',rows)

    def test_random_control_retains_language_class_source_duration_rules(self):
        rows=[dict(id=f'{lang}{label}{i}',canonical_source=f'{lang}{label}{i}',language=lang,label=label,
            split='train',condition='offline',output_samples=n) for lang in ('en','zh') for label in (0,1)
            for i,n in enumerate((16000,17000,90000))]
        pairs=random_neighbors(rows,21);by={r['id']:r for r in rows}
        self.assertEqual(pairs,random_neighbors(rows,21))
        for rid,items in pairs.items():
            self.assertTrue(items)
            for item in items:
                r,f=by[rid],by[item['source']]
                self.assertEqual((r['label'],f['label']),(1,0));self.assertEqual(r['language'],f['language'])
                self.assertTrue(.5<=r['output_samples']/f['output_samples']<=2);self.assertIsNone(item['similarity'])

if __name__=='__main__':unittest.main()
