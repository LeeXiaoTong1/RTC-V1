import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .training import balanced_ce, schedule, fit, predict, normalization
from .partition import split_sources
from .cache import save_npz, read_shard, extract
from .metrics import report, guard, paired_changes, bootstrap, PANELS
from w2v_v31810.common import atomic_json, digest
from w2v_v31810.identity import build_registry, registry_config


def registry_rows():
    rows=[]
    for lang in ('en','zh'):
        for label in (0,1):
            for n in range(20):
                source=f'{lang}{label}{n}'
                import hashlib
                canonical=hashlib.sha256(source.encode()).hexdigest()
                for kind in ('offline','online'):
                    rows.append(dict(id=source+kind,audio=source+kind,canonical_source=canonical,source_id=source,
                        group_id=source,language=lang,label=label,split='train',condition=kind,
                        audio_sha256=hashlib.sha256((source+kind).encode()).hexdigest(),decoded_pcm_sha256=hashlib.sha256((source+kind).encode()).hexdigest()))
    registry=build_registry(rows,[],{});return rows,dict(source_registry=registry_config(registry))


def panel_data(count=12):
    rows=[];logits=[];glob=[];local=[];rng=np.random.default_rng(77)
    for lang in ('en','zh'):
        for label in (0,1):
            for i in range(count):
                source=f'{lang}{label}{i}'
                for panel in PANELS:
                    rows.append(dict(id=source+panel,canonical_source=source,language=lang,label=label,
                        role='online' if panel=='official_online' else 'noisy',validation_panel=panel))
                    z=np.array([1.,-1.],np.float32)*(1 if label==0 else -1)
                    if lang=='en' and label==1 and i<4:z=-z
                    logits.append(z);g=rng.normal(size=256).astype(np.float32);g[0]=3*(1 if label==0 else -1)
                    glob.append(g);l=rng.normal(size=384).astype(np.float32)*.05;l[:2]=z;local.append(l)
    return dict(rows=rows,logits=np.array(logits),global_=np.array(glob),local=np.array(local))


def data(count=12):
    value=panel_data(count);value['global']=value.pop('global_');return value


class ExperimentTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_source_partition_contains_all_pcm_related_rows(self):
        rows,cfg=registry_rows();split=split_sources(rows,cfg)
        self.assertFalse(set(split['fit'])&set(split['validation']))
        self.assertEqual(set(split['fit'])|set(split['validation']),{r['canonical_source'] for r in rows})
        self.assertEqual(split,split_sources(rows,cfg))
        self.assertEqual(len(split['validation']),8)
        corrupt=copy.deepcopy(rows);corrupt[0]['decoded_pcm_sha256']='0'*64
        with self.assertRaises(ValueError):split_sources(corrupt,cfg)

    def test_eight_group_means_not_sample_means(self):
        z=torch.randn(9,2,requires_grad=True);y=torch.tensor([0,0,1,0,1,0,1,0,1]);groups=torch.tensor([0,0,1,2,3,4,5,6,7])
        loss=balanced_ce(z,y,groups)
        ce=torch.nn.functional.cross_entropy(z,y,reduction='none')
        torch.testing.assert_close(loss,torch.stack([ce[groups==i].mean() for i in range(8)]).mean())
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
        with self.assertRaises(ValueError):balanced_ce(z[:2],y[:2],groups[:2])

    def test_cached_source_exposure_budgets_exact(self):
        value=data();batches=list(schedule(value['rows'],[20,16,16,16],32,7,0));counter=Counter()
        for batch in batches:
            for i in batch:
                r=value['rows'][i];counter[r['role'],r['language'],r['label']]+=1
        for role in ('online','noisy'):
            self.assertEqual([counter[role,lang,label] for lang,label in (('en',0),('en',1),('zh',0),('zh',1))],[20,16,16,16])
        self.assertEqual(batches,list(schedule(value['rows'],[20,16,16,16],32,7,0)))

    def test_tradeoff_rejected_even_when_en_real_improves(self):
        value=data();z=value['logits'].copy();baseline=report(value['rows'],z)
        for i,r in enumerate(value['rows']):
            if r['language']=='en' and r['label']==1:z[i]=[-1,1]
        self.assertTrue(guard(baseline,report(value['rows'],z))['passed'])
        first=next(i for i,r in enumerate(value['rows']) if r['validation_panel']=='regular_noisy' and r['label']==0)
        z[first]=[-1,1]
        self.assertFalse(guard(baseline,report(value['rows'],z))['passed'])

    def test_paired_cohorts_source_bootstrap_and_ties(self):
        value=data(12);z=value['logits'].copy()
        for i,r in enumerate(value['rows']):
            if r['label']==1:z[i]=[-1,1]
        cohorts=paired_changes(value['rows'],value['logits'],z)
        self.assertEqual(cohorts['regular_noisy/en/1/both_wrong']['corrected'],4)
        b=bootstrap(value['rows'],value['logits'],z,100)
        self.assertTrue(b['supported'])
        tie=np.zeros_like(z);m=report(value['rows'],tie)
        self.assertEqual(m['panels']['official_online']['fake_recall'],1)
        self.assertEqual(m['panels']['official_online']['real_recall'],0)

    def test_cache_rejects_mismatch_nonfinite_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'000.npz';arrays={'global':np.zeros((2,256),np.float32),'local':np.zeros((2,384),np.float32),'logits':np.zeros((2,2),np.float32)}
            save_npz(path,**arrays);atomic_json(path.with_suffix('.json'),dict(identity='id',sha256=digest(path),rows=[{},{}]))
            self.assertEqual(len(read_shard(path,'id')['rows']),2)
            with self.assertRaises(ValueError):read_shard(path,'other')
            path.write_bytes(path.read_bytes()+b'x')
            with self.assertRaises(ValueError):read_shard(path,'id')
            arrays['global'][0,0]=np.nan;save_npz(path,**arrays)
            atomic_json(path.with_suffix('.json'),dict(identity='id',sha256=digest(path),rows=[{},{}]))
            with self.assertRaises(ValueError):read_shard(path,'id')

    def test_residual_training_resume_equal_and_frozen_classifier(self):
        value=data(12);classifier=torch.nn.Linear(384,2)
        with torch.no_grad():classifier.weight.zero_();classifier.bias.zero_();classifier.weight[:,:2]=torch.eye(2)
        cfg=dict(train_device='cpu',backend_precision='fp32',lr=.003,weight_decay=1e-4,epochs=2,patience=5,
                 batch_size=32,source_draws_per_stream=[16]*4,label_smoothing=.02)
        with tempfile.TemporaryDirectory() as tmp:
            continuous,best=fit('residual',value,value,classifier,cfg,Path(tmp)/'all',17,'fixture')
            calls=[0]
            def stop():
                calls[0]+=1
                return calls[0]>5  # one full four-update epoch, stop at next epoch
            with self.assertRaises(InterruptedError):fit('residual',value,value,classifier,cfg,Path(tmp)/'resume',17,'fixture',stop)
            resumed,again=fit('residual',value,value,classifier,cfg,Path(tmp)/'resume',17,'fixture')
            for k,v in continuous.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
            torch.testing.assert_close(continuous.classifier.weight,classifier.weight,rtol=0,atol=0)
            self.assertEqual(best['epoch'],again['epoch'])
            with self.assertRaises(ValueError):fit('residual',value,value,classifier,cfg,Path(tmp)/'resume',17,'changed')


if __name__=='__main__':unittest.main()
