"""CPU behavioral tests; real AASIST/WAV, small encoder, explicit native boundaries."""
import copy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from torch import nn
from .aasist import SSLAASIST
from .regions import windows,coverage_mass,RegionHead
from .model import Detector,LoRALinear,optimizer_for
from .objective import risk,unit_loss
from .sampling import Plan,probe_tickets,validate_units
from .data import Waves,loader,close,ready
from .common import GROUPS,partial_state,apply_partial,seed_all,digest,atomic_json,read_json
from .records import dev_partition
from .augment import noise_wave,recipe
from .step import train_step,schedule
from .calibration import fit as fit_calibration,apply as calibrate


class TinyLayout:
    def __init__(self,shape,seq_lens,device=None):self.seq_lens=seq_lens


class TinyFrontend(nn.Module):
    def __init__(self,dim=16):
        super().__init__();self.conv=nn.Conv1d(1,dim,400,stride=320)
    def forward(self,x,layout):
        y=self.conv(x[:,None]).transpose(1,2)
        return y,TinyLayout(y.shape,[(int(n)-400)//320+1 for n in layout.seq_lens])


class TinyLayer(nn.Module):
    def __init__(self,dim=16):
        super().__init__()
        self.q_proj,self.k_proj,self.v_proj,self.output_proj=[nn.Linear(dim,dim) for _ in range(4)]
        self.norm=nn.LayerNorm(dim)
    def forward(self,x,layout,bias_cache):
        scores=self.q_proj(x)@self.k_proj(x).transpose(1,2)/(x.shape[-1]**.5)
        mask=torch.arange(x.shape[1],device=x.device)[None,:]<torch.tensor(layout.seq_lens,device=x.device)[:,None]
        scores=scores.masked_fill(~mask[:,None,:],-torch.inf)
        return self.norm(x+self.output_proj(scores.softmax(-1)@self.v_proj(x)))


class TinyEncoder(nn.Module):
    def __init__(self):super().__init__();self.layers=nn.ModuleList([TinyLayer(),TinyLayer()])
    def forward(self,x,layout):
        for layer in self.layers:x=layer(x,layout,None)
        return x


def tiny_cfg():
    return dict(version='3.18',variant='C3',encoder_dim=16,device='cpu',lora_layers=1,lora_rank=2,lora_alpha=4.,
        lora_dropout=.05,local_evidence=True,window=128,hop=64,window_batch=16,checkpointing=True,
        lora_lr=.001,head_lr=.001,warm_lr=.001,evidence_lr=.001,weight_decay=.01,max_grad_norm=1.,
        epochs=3,warm_epochs=1,workers=0,eval_batch=8,microbatch=6,frame_budget=1000,stream_sources=4,
        train_probe_per_group=1,panel_per_group=1,seed=31801,label_smoothing=.02,risk_coefficient=.25,
        patience=0,min_joint_epochs=3,min_delta=.0002,calibration_fraction=.2,
        raw_audio_cache_mib=1,noise_cache_mib=1,ffmpeg='ffmpeg',
        free_reserve_bytes=0,disk_margin_bytes=1024**2,selection='fixture select; no fallback',
        initialization=dict(mode='tiny CPU fixture; not public 1B'),
        omni_provenance=dict(repo='tiny-fixture'))


def tiny_model(cfg):return Detector(TinyFrontend(),TinyEncoder(),cfg,TinyLayout)


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_requested_epoch_default_and_configured_audit(self):
        from .config import parser,validate
        from .audit import describe
        args=parser().parse_args([]);validate(args)
        self.assertEqual((args.epochs,args.warm_epochs,args.patience),(10,2,0))
        with tempfile.TemporaryDirectory() as d:
            cfg,train,dev=make_fixture(Path(d))
            report=describe(train,stream_sources=8)['v318']
            self.assertEqual(report['steps'],2)
            self.assertEqual(report['online_views'],16)
            self.assertEqual(report['noisy_views'],32)
    def test_full_window_coverage_and_bounded_weights(self):
        head=RegionHead(SSLAASIST(16))
        for n in (12,127,128,129,191,192,193,300,999):
            ranges=windows(n);mass=coverage_mass(n,ranges)
            self.assertAlmostEqual(float(mass.sum()),1.,places=6)
            self.assertEqual(ranges[-1][1],n)
            h=torch.randn(len(ranges),160)
            self.assertTrue(torch.allclose(head.weights(h,mass),mass))
            with torch.no_grad():head.contribution[-1].weight.fill_(10.)
            ratio=head.weights(h,mass)/mass
            self.assertTrue(bool(((ratio>=1/3-1e-6)&(ratio<=3+1e-6)).all()))
            nn.init.zeros_(head.contribution[-1].weight)
    def test_risk_both_gradients_and_ties(self):
        a=torch.tensor(2.,requires_grad=True);b=torch.tensor(.5,requires_grad=True)
        risk(a,b).backward();self.assertAlmostEqual(a.grad.item(),.625);self.assertAlmostEqual(b.grad.item(),.375)
        a=torch.tensor(1.,requires_grad=True);b=torch.tensor(1.,requires_grad=True)
        risk(a,b).backward();self.assertAlmostEqual(a.grad.item(),.5);self.assertAlmostEqual(b.grad.item(),.5)
    def test_cyclic_coverage_balancing_and_missing_online(self):
        rows=[]
        for kind in ('online','offline'):
            for g in GROUPS:
                for i in range(3+(g[1]==0)*4):
                    identity=f'{kind}/{g}/{i}'
                    rows.append(dict(id=identity,source_id=identity,condition=kind,language=g[0],label=g[1],audio_sha256=identity))
        plan=Plan(rows,8,1);coverage=plan.coverage(0)
        self.assertEqual(coverage['class_view_counts']['fake'],coverage['class_view_counts']['real'])
        self.assertEqual(coverage['stream_loss_mass'],{'online':.5,'noisy':.5})
        for kind in plan.pools:
            for g,pool in plan.pools[kind].items():
                stream=plan.stream(kind,g,0)+plan.stream(kind,g,1)
                self.assertEqual(set(stream[:len(pool)]),set(pool))
        self.assertEqual(list(plan.batches(2)),list(plan.batches(2)))
        self.assertEqual(len(probe_tickets(plan,2)),16)
    def test_objective_keeps_class_mass(self):
        values=[]
        for g in GROUPS:
            for role in ('online','noisy'):
                rows=[dict(role='online' if role=='online' else 'noisy_'+v,label=g[1],language=g[0],source_id=str(g),occurrence='one') for v in (('a',) if role=='online' else ('a','b'))]
                z=torch.zeros(len(rows),2,requires_grad=True)
                loss,_,_=unit_loss(z,rows,4,0.,0.);values.append(loss)
        self.assertAlmostEqual(float(sum(values)),np.log(2),places=6)
    def test_bn_state_in_partial_and_frozen_joint(self):
        cfg=tiny_cfg();model=tiny_model(cfg);model.train();x=np.sin(np.arange(9000)*.02).astype(np.float32)
        before=partial_state(model);model([x,x]);after=partial_state(model)
        self.assertTrue(any('running_mean' in n and not torch.equal(v,after[n]) for n,v in before.items()))
        model.set_phase(True);frozen=partial_state(model);model([x,x])
        self.assertTrue(all(torch.equal(v,partial_state(model)[n]) for n,v in frozen.items() if 'running_' in n or 'num_batches' in n))
        clone=tiny_model(cfg);apply_partial(clone,partial_state(model))
        self.assertTrue(all(torch.equal(v,partial_state(clone)[n]) for n,v in partial_state(model).items()))
    def test_lora_gradient_and_frozen_original(self):
        cfg=tiny_cfg();model=tiny_model(cfg);model.set_phase(True)
        base={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        z=model([np.random.default_rng(1).normal(size=9000).astype(np.float32),np.random.default_rng(2).normal(size=12000).astype(np.float32)])
        torch.nn.functional.cross_entropy(z,torch.tensor([0,1])).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('.lora_b')))
        self.assertTrue(all(p.grad is None and torch.equal(p,base[n]) for n,p in model.named_parameters() if not p.requires_grad))
    def test_lora_zero_equivalence_and_lengths(self):
        cfg=tiny_cfg();cfg['lora_dropout']=0.;model=tiny_model(cfg).eval()
        x=np.random.default_rng(1).normal(size=18000).astype(np.float32);y=x[:10000]
        model.set_phase(False)
        with torch.no_grad():a=model([x,y]);solo=model([x])
        model.set_phase(True)
        with torch.no_grad():b=model([x,y])
        self.assertTrue(torch.allclose(a,b,atol=1e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(a[0],solo[0],atol=1e-5,rtol=1e-5))
    def test_microbatch_gradient_budget_with_frozen_bn(self):
        cfg=tiny_cfg();seed_all(12);one=tiny_model(cfg);two=copy.deepcopy(one)
        units=[]
        for lang,label in GROUPS:
            for kind in ('online','offline'):
                name=f'{kind}/{lang}/{label}'
                unit=[]
                for v in range(1 if kind=='online' else 2):
                    wave=np.random.default_rng(100+label+v).normal(0,.05,7000+v*320).astype(np.float32)
                    unit.append(dict(wave=wave,label=label,language=lang,source_id=name,group_id=name,
                        occurrence=name,split='train',role='online' if kind=='online' else ('noisy_a','noisy_b')[v]))
                units.append(unit)
        gradients=[]
        for model,size in ((one,2),(two,12)):
            model.set_phase(True)
            for m in model.modules():
                if isinstance(m,nn.Dropout):m.p=0.
            optimizer=optimizer_for(model,cfg)
            captured={}
            optimizer.step=lambda:captured.update({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
            train_step(model,optimizer,units,dict(cfg,microbatch=size,frame_budget=10000),.25)
            gradients.append(captured)
        self.assertEqual(gradients[0].keys(),gradients[1].keys())
        for name,a in gradients[0].items():
            self.assertTrue(torch.allclose(a,gradients[1][name],atol=3e-6,rtol=3e-4),name)
    def test_natural_noise_and_distinct_families(self):
        class Bank:
            def sample(self,rng):return np.arange(10000,dtype=np.float32),'recording'
        out,keys=noise_wave(Bank(),8000,'continuous',np.random.default_rng(3))
        self.assertTrue(np.array_equal(np.diff(out),np.ones(7999)))
        for i in range(50):
            a=recipe(1,str(i),'source',0,0);b=recipe(1,str(i),'source',1,0)
            self.assertNotEqual(a['family'],b['family']);self.assertGreaterEqual(a['snr_db'],10.)
    def test_global_calibration_preserves_ranking(self):
        rows=[];z=[]
        for condition in ('online','seen','heldout'):
            for lang,label in GROUPS:
                for i in range(3):
                    rows.append(dict(condition=condition,language=lang,label=label));z.append([2.-3*label+i*.1,0.])
        cal=fit_calibration(rows,z,list(range(len(rows))));new=calibrate(z,cal)
        self.assertGreater(cal['scale'],0.)
        self.assertTrue(np.array_equal(np.argsort(np.array(z)[:,0]),np.argsort(new[:,0]-new[:,1])))


def make_fixture(root):
    cfg=tiny_cfg();train=[];dev=[];official=[];pairs=[]
    for split in ('train','dev'):
        for lang,label in GROUPS:
            for index in range(3):
                source=f'{split}_{lang}_{label}_{index}.wav';group=f'{split}:{lang}:{label}:{index}'
                wave=(.06*np.sin(np.arange(8500+index*320)*(.02+.007*label))+.005*np.random.default_rng(index+label*10).normal(size=8500+index*320)).astype(np.float32)
                for condition in (('offline','online') if split=='train' else ('offline','online','seen','heldout')):
                    name=f'{condition}/{lang}/{source}' if condition!='online' else f'online/{lang}/online_{source}'
                    p=root/split/name;p.parent.mkdir(parents=True,exist_ok=True);sf.write(p,wave,16000,subtype='FLOAT')
                    st=p.stat();r=dict(id=name,audio=str(p),audio_size=st.st_size,audio_mtime_ns=st.st_mtime_ns,
                        audio_sha256=digest(p),condition=condition,language=lang,label=label,split=split,
                        source_id=f'offline/{lang}/{source}' if condition!='online' else f'online/{lang}/online_{source}',
                        group_id=group if condition!='online' else 'online:'+group,full_length=True)
                    if split=='train':
                        # Unique fixture groups are distinct draws even when tonal wave bytes repeat.
                        r['audio_sha256']=group+condition;train.append(r)
                    elif condition!='offline':
                        if condition in ('seen','heldout'):r['source_id']=f'offline/{lang}/{source}'
                        dev.append(r)
                    if split=='dev' and condition in ('offline','online'):official.append(name+' '+('spoof' if label==0 else 'bonafide'))
                if split=='dev':pairs.append([f'dev/offline/{lang}/{source}',f'dev/online/{lang}/online_{source}'])
    protocol=root/'dev.txt';protocol.write_text('\n'.join(official)+'\n',encoding='utf-8')
    pair=root/'dev_pairs.csv'
    with pair.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['offline_id','online_id']);w.writerows(pairs)
    noise=root/'noise.wav';sf.write(noise,np.random.default_rng(9).normal(0,.03,32000).astype(np.float32),16000,subtype='FLOAT');s=noise.stat()
    record=dict(path=str(noise),sha256=digest(noise),size=s.st_size,mtime_ns=s.st_mtime_ns)
    cfg.update(dev_pairs=str(pair),official_dev_protocol=str(protocol),official_dev_root=str(root/'dev'),noise_records={'train':[record],'dev':[record]})
    return cfg,train,dev


class NativeBoundary:
    def __init__(self,*args):pass
    def __call__(self,x,r):return x.copy()


class TestIntegration(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_partition_official_pairs_and_noise_siblings(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,train,dev=make_fixture(Path(d));p=dev_partition(dev,cfg['dev_pairs'],cfg['seed'])
            self.assertFalse(set(p['source_keys'][i] for i in p['select'])&set(p['source_keys'][i] for i in p['calibration']))
            self.assertEqual(len(p['select'])+len(p['calibration']),len(dev))
    def test_spawned_inference_preserves_order_and_scores(self):
        from .inference import infer
        with tempfile.TemporaryDirectory() as d:
            cfg,train,dev=make_fixture(Path(d));seed_all(22);model=tiny_model(cfg)
            a,ra=infer(model,dev,cfg)
            b,rb=infer(model,dev,dict(cfg,workers=2))
            self.assertEqual([r['id'] for r in ra],[r['id'] for r in rb])
            self.assertTrue(np.array_equal(a,b))
    def test_resume_atomic_bn_optimizer_and_export(self):
        from .train import run_experiment,save
        from .evaluate import export,selected
        with tempfile.TemporaryDirectory() as d,patch('w2v_v318.data.Engines',NativeBoundary):
            root=Path(d);cfg,train,dev=make_fixture(root)
            runs=[root/'full',root/'resumed']
            for run in runs:
                run.mkdir();atomic_json(run/'train_rows.json',train);atomic_json(run/'dev_rows.json',dev)
            cfg['saved_manifests']={n:digest(runs[0]/n) for n in ('train_rows.json','dev_rows.json')}
            for run in runs:atomic_json(run/'config.json',cfg)
            run_experiment(cfg,runs[0],train,dev,tiny_model)
            def interrupted(*args):
                save(*args)
                if args[2]['epoch']==1:raise RuntimeError('fixture interrupt after commit')
            with patch('w2v_v318.train.save',interrupted),self.assertRaisesRegex(RuntimeError,'fixture interrupt'):
                run_experiment(cfg,runs[1],train,dev,tiny_model)
            run_experiment(cfg,runs[1],train,dev,tiny_model)
            a=torch.load(runs[0]/'last.pt',weights_only=True);b=torch.load(runs[1]/'last.pt',weights_only=True)
            self.assertEqual(a['best_tag'],b['best_tag']);self.assertEqual(a['last_tag'],b['last_tag'])
            for name,value in a['model'].items():self.assertTrue(torch.equal(value,b['model'][name]),name)
            for k,values in a['optimizer']['state'].items():
                for name,value in values.items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,b['optimizer']['state'][k][name]))
            _,best,meta=selected(runs[1],'best');self.assertFalse(meta['baseline_fallback'])
            export(runs[1],root/'dev-export',dev=True,device='cpu',workers=0,raw=False,model_factory=tiny_model,verify=False)
            self.assertTrue((root/'dev-export'/'validation_meta.json').is_file())
            protocol=root/'progress.txt'
            ids=[r['id'] for r in dev if r['condition']=='online']
            protocol.write_text('\n'.join(ids)+'\n',encoding='utf-8')
            export(runs[1],root/'submission',kind='last',protocol=protocol,audio_root=root/'dev',
                device='cpu',workers=0,model_factory=tiny_model,verify=False)
            metadata=read_json(root/'submission'/'submission_meta.json')
            self.assertEqual(metadata['selected'],b['last_tag']);self.assertFalse(metadata['baseline_fallback'])
            lines=(root/'submission'/'scores.txt').read_text().splitlines()
            self.assertEqual([s.split()[0] for s in lines],ids)
            self.assertTrue(all(0<=float(s.split()[1])<=1 for s in lines))
            self.assertTrue((root/'resumed'/'diagnostics'/'curves.html').is_file())


if __name__=='__main__':unittest.main()
