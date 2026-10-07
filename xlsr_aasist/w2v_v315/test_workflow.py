"""Real tiny detector: matched-wave training, exact resume, export and compaction.

Only the unavailable native RTC engines are replaced by deterministic nonlinear
test DSP. Production setup separately requires actual FFmpeg AND WebRTC smoke tests.
"""
from contextlib import contextmanager, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch

from w2v_v39.common import atomic_json, digest, read_json
from w2v_v312.test_workflow import fixture, synthetic_expanded_bundles
from w2v_v312.train import run_experiment as run_v312
from w2v_v313.test_workflow import synthetic_paired_bundles, assert_nested_equal
from .config import configuration, parser, verify_inputs
from .model import load_model
from .state import atomic_save, load_selected, apply_candidate
from .train import run_experiment, infer
from .data import Triplets

torch.set_num_threads(1)


class TestEngines:
    def __init__(self,ffmpeg=None):pass
    def __call__(self,wave,r):
        return (.8*wave+.1*np.tanh(2*wave)).astype(np.float32)


class WorkerTriplets(Triplets):
    """Picklable CPU DSP fixture; all worker transport/feature logic is production."""
    def _ready(self):
        if self.bank is None:
            from .augment import NoiseBank
            from .cache import RollingCache
            from .state import identity
            self.bank=NoiseBank(self.cfg['noise_records'][self.split])
            self.engines=TestEngines()
            self.cache=RollingCache(Path(self.run)/'rolling_pairs',identity(self.cfg),self.cfg['rolling_cache_bytes'])


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='v315_integration_')
        cls.addClassCleanup(cls.temporary.cleanup);cls.root=Path(cls.temporary.name)
        with redirect_stdout(io.StringIO()):
            _,old,_,cls.records=fixture(cls.root)
            cls.source=cls.root/'v312';cls.source.mkdir();atomic_json(cls.source/'config.json',old)
            with patch('w2v_v312.train.bundles',synthetic_expanded_bundles):run_v312(old,cls.source)
            noises={}
            files={}
            for i,split in enumerate(('train','dev')):
                path=cls.root/(split+'_noise.wav')
                sf.write(path,np.random.default_rng(i).normal(0,.1,1234+i).astype(np.float32),16000,subtype='FLOAT')
                s=path.stat();sha=digest(path);files[str(path)]=sha
                noises[split]=[dict(path=str(path),sha256=sha,size=s.st_size,mtime_ns=s.st_mtime_ns,
                                   split=split,original_recording=split)]
            binding=dict(noise_records=noises,noise_catalogs={s:{'hash':v[0]['sha256']} for s,v in noises.items()},
                         augmentation_files=files,augmentation_runtime={'test_DSP':True})
            with patch('w2v_v315.augment.bind_augmentation',return_value=binding):
                cls.cfg=configuration(parser().parse_args(['--source-run',str(cls.source),'--device','cpu','--workers','0','--epochs','1','--no-autotune']))
            cls.cfg.update(trainable_layers=1,microbatch=8,frame_budget=1200,feature_batch=4,
                tfcl_bins=21,tfcl_heads=8,rolling_cache_bytes=2*1024**2,free_reserve_bytes=0,
                disk_margin_bytes=0,min_gain=2.,patience=9,catastrophic_weighted_drop=2.)
            with synthetic_paired_bundles(cls.cfg) as (train,dev):
                cls.train_rows=copy.deepcopy(train['rows'])
                cls.dev_rows=[dict(r,source_sha256=r['audio_sha256']) for r in dev['rows']]
            cls.off=[dict(r,condition='offline',group_id=r['audio_sha256']) for r in cls.dev_rows if r['condition']=='seen']
            cls.protected={str(p):digest(p) for p in cls.root.rglob('*') if p.is_file()}

    @contextmanager
    def data(self,cfg):
        yield dict(rows=self.train_rows),dict(rows=self.dev_rows)

    def new_run(self,name):
        run=self.root/name;run.mkdir();atomic_json(run/'config.json',self.cfg)
        return run

    def run_model(self,run):
        with patch('w2v_v315.train.bundles',self.data), \
             patch('w2v_v315.train.offline_rows',return_value=(self.off,'test actual source waveforms')), \
             patch('w2v_v315.data.Engines',TestEngines), \
             patch('w2v_v315.augment.runtime',return_value={'test_DSP':True}):
            return run_experiment(self.cfg,run)

    def test_actual_training_updates_detector_and_auxiliary_and_exports_all_three_choices(self):
        from .evaluate import export
        from .cleanup import cleanup
        from .workflow import report
        with redirect_stdout(io.StringIO()):
            run=self.new_run('complete')
            original=load_model(self.cfg,training=False)
            before={k:v.detach().clone() for k,v in original.named_parameters() if v.requires_grad}
            done=self.run_model(run);report(run)
            self.assertEqual(done['committed_updates'],done['planned_updates'])
            self.assertEqual(done['selections']['best_guarded'],'starting_last')
            saved=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
            changed=[k for k,v in saved['model'].items() if not torch.equal(v,before[k])]
            self.assertTrue(any(k.startswith('head.classifier.2.') for k in changed))
            self.assertTrue(any(k.startswith('backbone.encoder.layers.1.') for k in changed))
            self.assertTrue(any(k.startswith('head.blocks.') for k in changed))
            self.assertTrue(any(g['name']=='training_only_tfcl' for g in saved['optimizer']['param_groups']))
            self.assertTrue((run/'dev_scores_offline_baseline.npz').exists())
            self.assertTrue((run/'treatment_baseline.json').exists())
            self.assertFalse((run/'features').exists());self.assertFalse(list(run.glob('epoch_*.pt')))
            self.assertLessEqual(sum(p.stat().st_size for p in (run/'rolling_pairs').glob('*.npz')),self.cfg['rolling_cache_bytes'])
            rows=[self.records['dev'][i] for i in (9,0,6,3)]
            protocol=run/'test_protocol.txt';protocol.write_text('\n'.join(r['id'] for r in rows)+'\n')
            expected={}
            for kind in ('best_guarded','best_weighted','last'):
                selected,meta=load_selected(run,kind)
                model=load_model(self.cfg,training=False);apply_candidate(model,selected['candidate'])
                z,_=infer(model,rows,self.cfg,'test export '+kind);expected[kind]=z
                archive=export(run,protocol,self.root/'audio',self.root/('submission_'+kind),'cpu',0,kind)
                with zipfile.ZipFile(archive) as stream:
                    values=[line.split() for line in stream.read('scores.txt').decode().splitlines()]
                self.assertEqual([v[0] for v in values],[r['id'] for r in rows])
                np.testing.assert_allclose([float(v[1]) for v in values],torch.from_numpy(z).softmax(-1)[:,0],atol=1e-9,rtol=0)
            cleanup(run,apply=True,remove_cache=True)
            self.assertFalse((run/'last.pt').exists());self.assertFalse((run/'rolling_pairs').exists())
            for kind,z in expected.items():
                selected,_=load_selected(run,kind)
                model=load_model(self.cfg,training=False);apply_candidate(model,selected['candidate'])
                after,_=infer(model,rows,self.cfg,'test compacted '+kind)
                np.testing.assert_array_equal(z,after)
            self.assertEqual(self.protected,{p:digest(p) for p in self.protected})

    def test_half_epoch_resume_reproduces_detector_auxiliary_moments_rng_and_recipes(self):
        with redirect_stdout(io.StringIO()):
            full=self.new_run('full');stopped=self.new_run('resume');self.run_model(full)
            interrupted=False
            def stop_after_commit(path,value,margin):
                nonlocal interrupted
                atomic_save(path,value,margin)
                if Path(path).name=='last.pt' and value['cursor']>0 and not interrupted:
                    interrupted=True;raise InterruptedError('fixture stop after half epoch')
            with patch('w2v_v315.train.atomic_save',side_effect=stop_after_commit):
                with self.assertRaises(InterruptedError):self.run_model(stopped)
            self.run_model(stopped)
            a=torch.load(full/'last.pt',map_location='cpu',weights_only=True)
            b=torch.load(stopped/'last.pt',map_location='cpu',weights_only=True)
            for key in ('model','auxiliary','optimizer','rng','selections','candidates'):
                assert_nested_equal(self,a[key],b[key])
            self.assertEqual(read_json(full/'panel_recipes.json'),read_json(stopped/'panel_recipes.json'))

    def test_defaults_and_source_replay_guards(self):
        self.assertEqual(parser().parse_args([]).epochs,4)
        with self.assertRaises(ValueError):load_model(dict(self.cfg,starting_checkpoint_sha256='0'*64))
        with patch('w2v_v315.augment.runtime',return_value={'test_DSP':True}):verify_inputs(self.cfg)
        with patch('w2v_v315.augment.runtime',return_value={'changed':True}):
            with self.assertRaisesRegex(ValueError,'runtime changed'):verify_inputs(self.cfg)

    def test_canonical_metadata_does_not_require_obsolete_vector_payloads(self):
        from .data import metadata
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            source=Path(self.cfg['v37_run'])/'features'/'train'
            for name in ('owner.json','complete.json','rows.json'):
                shutil.copy2(source/name,Path(directory)/name)
            result=metadata(directory,'train',self.cfg)
            self.assertGreater(len(result['rows']),0)
            self.assertFalse((Path(directory)/'x.npy').exists())
            (Path(directory)/'rows.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'row manifest changed'):metadata(directory,'train',self.cfg)

    def test_spawn_workers_transport_numpy_and_replay_identical_generated_views(self):
        from .data import _loader, SourcePlan, validate_batch
        with redirect_stdout(io.StringIO()):
            run=self.new_run('workers');cfg=dict(self.cfg,workers=2)
            plan=SourcePlan(self.train_rows,cfg['source_batch'],cfg['seed'])
            tickets=plan.batches(0)[:1]
            first=list(_loader(WorkerTriplets(self.train_rows,cfg,run),cfg,batch_sampler=tickets))[0]
            second=list(_loader(WorkerTriplets(self.train_rows,cfg,run),cfg,batch_sampler=tickets))[0]
            validate_batch(first,cfg['source_batch'])
            for a,b in zip(first,second):
                self.assertIsInstance(a['features'],np.ndarray);self.assertIsInstance(a['mask'],np.ndarray)
                np.testing.assert_array_equal(a['features'],b['features'])
                self.assertEqual(a['recipe'],b['recipe'])


if __name__=='__main__':unittest.main()
