"""Synthetic CPU acceptance of all three frozen feature capture points."""
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from .probes import _extract, _fit_one, _calibration, _group_holdout, bind_rows_to_tickets, _collect


class MockFusion:
    layers=(1,2,3,4)
    def project_layer(self,hidden,index,weights=None):
        return hidden[:,:,:128] if hidden.shape[-1]>=128 else hidden


class MockHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.aasist=nn.Module()
        self.aasist.out_layer=nn.Linear(384,2)


class MockDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion=MockFusion()
        self.cfg={'encoder_dim':256}
        self.head=MockHead()
    def encode(self,waves):
        # Model's actual stream invokes project_layer final exactly once.
        n=[(len(w)-400)//320+1 for w in waves]
        z=torch.ones((len(waves),max(n),256),dtype=torch.float32)
        z[:,:,0]=torch.arange(max(n))
        self.fusion.project_layer(z,4)
        return [z[i,:k,:128] for i,k in enumerate(n)]
    def classify_sequences(self,sequences):
        result=[]
        for seq in sequences:
            v=torch.cat([seq.mean(0),seq.std(0,unbiased=False),torch.ones(128)])
            result.append(v)
        return self.head.aasist.out_layer(torch.stack(result))


class FeatureTests(unittest.TestCase):
    def test_last_fused_and_preclassifier_hooks(self):
        fake=types.ModuleType('w2v_v31810.model')
        fake.Detector=MockDetector
        parent=types.ModuleType('w2v_v31810');parent.__path__=[]
        with patch.dict(sys.modules,{'w2v_v31810':parent,'w2v_v31810.model':fake}):
            model=MockDetector().eval()
            before=model.head.aasist.out_layer.weight.detach().clone()
            result=_extract(model,[np.zeros(6400,np.float32),np.zeros(7040,np.float32)])
            self.assertEqual(result['ssl_last'].shape,(2,512))
            self.assertEqual(result['fused'].shape,(2,256))
            self.assertEqual(result['preclassifier'].shape,(2,384))
            self.assertEqual(result['logits'].shape,(2,2))
            self.assertTrue(torch.equal(before,model.head.aasist.out_layer.weight))
            self.assertEqual(len(model.head.aasist.out_layer._forward_pre_hooks),0)
            self.assertEqual(model.fusion.project_layer.__func__,MockFusion.project_layer)

    def test_train_only_calibration_and_cv_uses_group_ids(self):
        rng=np.random.default_rng(27);n=48
        label=np.array([0,1]*(n//2));language=np.repeat(np.asarray(['en','zh']*(n//4)),2)
        group=np.array([f'G{i//2}' for i in range(n)])
        # Each origin has two views in a real task. Label must be shared.
        label=np.repeat(np.arange(n//2)%2,2)
        data=dict(id=np.asarray([f'v{i}' for i in range(n)]),canonical_source=group,
            panel=np.asarray(['official_online','regular_noisy']*(n//2)),language=language,
            label=label,logits=np.column_stack(((1-2*label)*2+rng.normal(0,.3,n),np.zeros(n))),
            ssl_last=rng.normal(size=(n,16)),fused=rng.normal(size=(n,8)),
            preclassifier=rng.normal(size=(n,12)))
        for k in ('ssl_last','fused','preclassifier'):
            data[k][:,0]+=(1-2*label)*2
        bias=_calibration(data,data,None,seed=4)
        self.assertIn('train_holdout',bias)
        self.assertNotIn('historical_audit_descriptive',bias)
        fitted=_fit_one('fused',data,data,None,seed=4)
        self.assertEqual(fitted['dim'],8)
        self.assertEqual(fitted['train_samples'],n)
        self.assertEqual(len(fitted['cv_candidates']),3)

    def test_stratified_original_source_holdout(self):
        sources=[];labels=[];languages=[]
        for lang in ('en','zh'):
            for label in (0,1):
                for origin in range(15):
                    for view in range(2):
                        sources.append(f'{lang}-{label}-{origin}')
                        labels.append(label)
                        languages.append(lang)
        groups=np.asarray(sources); y=np.asarray(labels); langs=np.asarray(languages)
        left=_group_holdout(groups,y,langs,17)
        right=_group_holdout(groups,y,langs,17)
        self.assertTrue(np.array_equal(left,right))
        self.assertFalse(set(groups[left]) & set(groups[~left]))
        for lang in ('en','zh'):
            for lab in (0,1):
                stratum=(langs==lang)&(y==lab)
                self.assertEqual(len(set(groups[stratum & ~left])),3)
                self.assertEqual(len(set(groups[stratum & left])),12)

    def test_train_ticket_maps_to_actual_source_row(self):
        # Regression: production Train tickets are indices into the COMPLETE
        # training manifest, not positions within the diagnostic selection.
        rows=[dict(id=f'id-{i}',canonical_source=f'origin-{i}',
                   condition='online' if i%2 else 'offline',language='en',label=i%2)
              for i in range(12)]
        tickets=[dict(index=9,canonical_source='origin-9',kind='online'),
                 dict(index=4,canonical_source='origin-4',kind='offline'),
                 dict(index=1,canonical_source='origin-1',kind='online')]
        bound,reindexed=bind_rows_to_tickets(rows,tickets,'train')
        self.assertEqual([x['id'] for x in bound],['id-9','id-4','id-1'])
        self.assertEqual([x['index'] for x in reindexed],[0,1,2])
        self.assertEqual([x['canonical_source'] for x in bound],
                         [x['canonical_source'] for x in tickets])
        with self.assertRaisesRegex(ValueError,'identity'):
            bind_rows_to_tickets(rows,[dict(index=9,canonical_source='wrong',kind='online')],'train')
        selected=[rows[9],rows[4]]
        v,_=bind_rows_to_tickets(selected,tickets[:2],'select')
        self.assertEqual([x['id'] for x in v],['id-9','id-4'])

    def test_train_holdout_rejects_conflicting_source_labels(self):
        with self.assertRaisesRegex(ValueError,'conflicting'):
            _group_holdout(np.asarray(['source']*2),np.asarray([0,1]),
                           np.asarray(['en','en']),2)

    def test_collect_extracts_waveform_from_selected_train_index(self):
        # End-to-end contract with lightweight fake data and encoder. The old
        # implementation replaced selected Train indices with 0,1,... while
        # retaining the entire original manifest, producing wrong waveforms.
        rows=[dict(id=f'id-{i}',canonical_source=f'origin-{i}',language='en',
                   label=i%2,condition='online' if i%2 else 'offline')
              for i in range(10)]
        tickets=[dict(index=9,kind='online',canonical_source='origin-9'),
                 dict(index=4,kind='offline',canonical_source='origin-4',
                      domain_override=dict(scene='office',family='webrtc'))]
        class StubWaves:
            def __init__(self, rows, cfg, split):self.rows=rows
            def __getitem__(self,ticket):
                row=self.rows[ticket['index']]
                selfsource=ticket['canonical_source']
                if row['canonical_source']!=selfsource:
                    raise AssertionError('Audio/identity mismatch')
                return [dict(row,role='online' if ticket['kind']=='online' else 'noisy',
                             wave=np.full(6400,int(row['id'].split('-')[-1]),dtype=np.float32))]
        def stub_loader(ds,cfg,batch_sampler):
            return [[ds[t] for t in chunk] for chunk in batch_sampler]
        def stub_microbatches(units,*args):
            yield units
        def stub_extract(model,waves):
            val=np.asarray([float(w[0]) for w in waves],np.float32)
            return dict(ssl_last=val[:,None],fused=val[:,None],preclassifier=val[:,None],
                        logits=np.column_stack((val,np.zeros_like(val))))
        parent=types.ModuleType('w2v_v31810');parent.__path__=[]
        data=types.ModuleType('w2v_v31810.data')
        data.Waves=StubWaves;data.loader=stub_loader;data.close=lambda _:None
        data.microbatches=stub_microbatches
        valid=types.ModuleType('w2v_v31810.validation_data');valid.FixedWaves=StubWaves
        with patch.dict(sys.modules,{'w2v_v31810':parent,'w2v_v31810.data':data,
                                     'w2v_v31810.validation_data':valid}), \
             patch('w2v_v31811.probes._extract',side_effect=stub_extract):
            result=_collect(None,dict(eval_workers=0,eval_batch=12,microbatch=12,
                                       frame_budget=9000),rows,tickets,'train')
        self.assertEqual(set(result['id'].tolist()),{'id-9','id-4'})
        self.assertEqual(set(result['ssl_last'][:,0].tolist()),{9.0,4.0})
        self.assertEqual(set(result['canonical_source'].tolist()),{'origin-9','origin-4'})


if __name__=='__main__':unittest.main()
