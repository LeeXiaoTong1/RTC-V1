"""Regression coverage for the real zero-error stop and cache-only continuation."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from w2v_v31810.common import atomic_json, atomic_save, read_json, digest, fingerprint
from .cache import save_npz, read_complete
from .continuation import LEGACY_SCHEMA, LEGACY_CODE, load_classifier
from .metrics import report, guard
from .training import fit
from .workflow import parser, prepare, execute, review_global, runtime_config, save_scores
from .test_experiment import data


class ContinuationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_perfect_inner_is_preserved_but_external_improvement_is_required(self):
        value=data();z=value['logits'].copy()
        for i,r in enumerate(value['rows']):z[i]=[1,-1] if r['label']==0 else [-1,1]
        perfect=report(value['rows'],z)
        self.assertTrue(review_global(perfect,perfect,scope='inner'))
        self.assertFalse(review_global(perfect,perfect,scope='external'))
        self.assertTrue(guard(perfect,perfect,scope='inner')['passed'])
        self.assertFalse(guard(perfect,perfect)['passed'])
        i=next(i for i,r in enumerate(value['rows']) if r['language']=='en' and r['label']==1 and r['validation_panel']=='official_online')
        z[i]=[1,-1]
        self.assertFalse(guard(perfect,report(value['rows'],z),scope='inner')['passed'])

    def test_online_fake_regression_cannot_buy_a_real_recall_gain(self):
        value=data();base=report(value['rows'],value['logits']);z=value['logits'].copy()
        for i,r in enumerate(value['rows']):
            if r['label']==1:z[i]=[-1,1]
        self.assertTrue(guard(base,report(value['rows'],z))['passed'])
        i=next(i for i,r in enumerate(value['rows']) if r['language']=='en' and r['label']==0 and r['validation_panel']=='official_online')
        z[i]=[-1,1];checked=guard(base,report(value['rows'],z))
        self.assertFalse(checked['passed'])
        self.assertFalse(checked['checks']['official_online/en_fake_preserved'])

    def fixture(self,source):
        source.mkdir();fit_data=data(6);inner=data(6);select=data(6);audit=data(6)
        for name,item in (('fit',fit_data),('inner',inner),('select',select),('audit',audit)):
            for i,r in enumerate(item['rows']):
                r['canonical_source']=name+'-'+r['canonical_source'];r['id']=name+'-'+r['id']
                r.update(condition='online' if r['role']=='online' else 'offline',split='train')
                if name in ('fit','inner'):
                    item['logits'][i]=[1,-1] if r['label']==0 else [-1,1]
                    item['local'][i,:2]=item['logits'][i]
        train_rows=fit_data['rows']+inner['rows']
        tickets=lambda view:[dict(index=i,canonical_source=r['canonical_source'],visit=view) for i,r in enumerate(fit_data['rows'])]
        def panel(item):
            return dict(rows=item['rows'],tickets=[dict(index=i) for i in range(len(item['rows']))],
                        speech_sources=sorted({r['canonical_source'] for r in item['rows']}),scene_noise_records=[])
        protocol={'select':panel(select),'audit':panel(audit)}
        split=dict(fit=sorted({r['canonical_source'] for r in fit_data['rows']}),
                   validation=sorted({r['canonical_source'] for r in inner['rows']}),interpretation='Parent already saw old Train')
        original=dict(seed=7,scene_noise_records={'train':[]})
        for name,value in [('parent_config',original),('train_rows',train_rows),('validation_protocol',protocol),('source_registry',{}),('residual_split',split)]:
            atomic_json(source/(name+'.json'),value)
        classifier=torch.nn.Linear(384,2)
        with torch.no_grad():
            classifier.weight.zero_();classifier.bias.zero_();classifier.weight[:,:2]=torch.eye(2)
        atomic_save(source/'parent.pt',{'head.aasist.out_layer.'+k:v for k,v in classifier.state_dict().items()})
        artifacts={p.name:digest(p) for p in source.iterdir() if p.is_file()}
        cfg=dict(schema=LEGACY_SCHEMA,code=LEGACY_CODE,version='3.18.12',artifacts=artifacts,
                 device='cpu',train_device='cpu',workers=0,microbatch=2,frame_budget=4000,
                 cache_views=2,epochs=20,patience=5,batch_size=32,lr=.0003,weight_decay=.0001,
                 backend_precision='fp32',label_smoothing=.02,source_draws_per_stream=[16]*4,seeds=[11,12,13])
        atomic_json(source/'config.json',cfg);identity=fingerprint(cfg);live=runtime_config(source,cfg)
        def cache(name,rows,tk,item,recipe,fixed):
            root=source/'cache'/name;root.mkdir(parents=True)
            bound=dict(identity=identity,rows=fingerprint(rows),tickets=fingerprint(tk),recipe=fingerprint(recipe),fixed=fixed,dtype='float32',shard_size=64)
            atomic_json(root/'manifest.json',bound)
            for i,start in enumerate(range(0,len(tk),64)):
                path=root/f'{i:05d}.npz'
                save_npz(path,**{k:item[k][start:start+64] for k in ('global','local','logits')})
                atomic_json(path.with_suffix('.json'),dict(identity=fingerprint([bound,i]),sha256=digest(path),rows=item['rows'][start:start+64]))
        for v in range(2):cache(f'train_{v}',train_rows,tickets(v),fit_data,live,False)
        local=dict(live,scene_noise_records={'dev':[]})
        ip=panel(inner)
        cache('train_inner',ip['rows'],ip['tickets'],inner,local,True)
        sp=protocol['select'];cache('dev_select',sp['rows'],sp['tickets'],select,local,True)
        review=dict(baseline=dict(inner=report(inner['rows'],inner['logits']),select=report(select['rows'],select['logits'])),
                    probes={},coverage={},supported=False)
        for kind in ('global','local'):
            zz=np.asarray([[1,-1] if r['label']==0 else [-1,1] for r in select['rows']],dtype=np.float32)
            metrics=report(inner['rows'],inner['logits'])
            review['probes'][kind]=dict(inner=metrics,select=report(select['rows'],zz),epoch=1)
            (source/f'probe_{kind}').mkdir()
            atomic_json(source/f'probe_{kind}'/'selected.json',dict(kind=kind,identity=identity,epoch=1,metrics=metrics))
            atomic_json(source/f'probe_{kind}'/'history.json',[dict(epoch=1,best=True)])
            save_scores(source/f'probe_{kind}_select.npz',select,zz)
        atomic_json(source/'feature_review.json',review)
        decision='stop_global_hypothesis_not_supported'
        atomic_json(source/'summary.json',dict(stage='feature_review',identity=identity,decision=decision,feature_review=review))
        atomic_json(source/'completed.json',dict(identity=identity,decision=decision,summary_sha256=digest(source/'summary.json')))
        return cfg,fit_data,inner,select,tickets,ip

    def test_real_cached_continuation_skips_encoder_and_probes_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'old';cfg,fit_data,inner,select,tickets,ip=self.fixture(source)
            before={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
            args=parser().parse_args(['--continue-from',str(source),'--run',str(root/'new'),
                '--train-device','cpu','--epochs','1','--batch-size','32','--lr','1e-12'])
            run,new=prepare(args)
            self.assertNotEqual(fingerprint(new),fingerprint(cfg))
            restored=load_classifier(run,new)
            np.testing.assert_array_equal(restored(torch.from_numpy(select['local'])).numpy(),select['logits'])
            with patch('w2v_v31812.workflow.load_parent',side_effect=AssertionError('must not load the 3B parent')), \
                 patch('w2v_v31812.workflow.extract',side_effect=AssertionError('must not encode audio')), \
                 patch('w2v_v31810.sampling.Plan'), \
                 patch('w2v_v31812.workflow.full_sweep',side_effect=lambda rows,plan,v:tickets(v)), \
                 patch('w2v_v31812.workflow._panels',return_value=ip), \
                 patch('w2v_v31812.workflow.fit',wraps=fit) as fitter:
                execute(run,new)
            self.assertEqual([c.args[0] for c in fitter.call_args_list],['residual']*3)
            self.assertTrue(read_json(run/'feature_review.json')['supported'])
            self.assertTrue((run/'residual.pt').exists())
            self.assertFalse(read_json(run/'summary.json')['joint_eligible'])
            self.assertEqual(before,{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()})
            self.assertEqual(prepare(parser().parse_args(['--resume',str(run)])),(run,new))

    def test_continuation_rejects_changed_extractors_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'old';cfg,*_=self.fixture(source)
            bad=copy.deepcopy(cfg);bad['code']['model.py']='changed';atomic_json(source/'config.json',bad)
            args=parser().parse_args(['--continue-from',str(source),'--run',str(root/'new')])
            with self.assertRaisesRegex(ValueError,'audited'):prepare(args)
            self.assertFalse((root/'new').exists())

    def test_cache_only_reader_rejects_missing_shards_and_wrong_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'old';cfg,fit_data,inner,select,tickets,ip=self.fixture(source)
            rows=read_json(source/'train_rows.json');recipe=runtime_config(source,cfg);directory=source/'cache'/'train_0'
            found=read_complete(rows,tickets(0),recipe,directory,fingerprint(cfg))
            np.testing.assert_array_equal(found['global'],fit_data['global'])
            with self.assertRaisesRegex(ValueError,'configuration'):
                read_complete(rows,tickets(0),dict(recipe,frame_budget=1),directory,fingerprint(cfg))
            (directory/'00001.npz').unlink()
            with self.assertRaisesRegex(ValueError,'missing; no re-encoding'):
                read_complete(rows,tickets(0),recipe,directory,fingerprint(cfg))


if __name__=='__main__':unittest.main()
