"""Release boundary tests: real old pins, WAV I/O, SSL, Adam, Dev and export.

Only the unavailable native Linux APM/runtime boundary is replaced in the CPU
integration. Fingerprint checks, checkpoint loaders, features, model, optimizer,
metrics, training orchestration and submission inference are not mocked.
"""
import ast
import copy
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch

from w2v_v39.common import ROOT, atomic_json, digest, read_json
from w2v_v39.config import runtime_versions
from w2v_v313.state import identity, partial_state, to_cpu, capture_rng
from w2v_v316_tfcl.config import code_fingerprints, pretrained_assets
from w2v_v316_tfcl.state import SCHEMA as SOURCE_SCHEMA
from w2v_v316_tfcl.model import load_model as old_model, optimizer_for as old_optimizer
from w2v_v316_tfcl.data import SourcePlan, TrainingLoader, tensors
from w2v_v316_tfcl.metrics import measure
from w2v_v3.test_model_step import tiny_detector, small_head
from w2v_v36.features import MODE, FORMAT, _digest
from .arguments import parser
from .compatibility import OLD_CONFIG_HASHES, verify_source
from .config import configuration, verify_inputs
from .cursor import source_cursor, segments, completed_tag
from .output import infer
from .train import run_experiment
from .evaluate import export
from .state import load_resume, load_selected


FIXTURE = Path(__file__).parent/'fixtures'/'v316_config_22e4252.txt'


class NativeFixtureEngine:
    """The native codec boundary only; actual noise mixing and feature I/O run."""
    def __init__(self, *args): pass
    def __call__(self, waveform, recipe): return waveform.copy()


@contextmanager
def native_fixture():
    with patch('w2v_v316_tfcl.config.runtime', return_value={'fixture_native_apm': True}), \
         patch('w2v_v316_tfcl.data.Engines', NativeFixtureEngine):
        yield


def metadata_cache(path, rows, split, cfg):
    path.mkdir(parents=True)
    code = {}
    for folder, name in (('w2v_v36','features.py'),('w2v_v3','model.py'),('w2v_v3','data.py'),
                         ('w2v_v3','step.py'),('w2v_aasist','model.py'),('w2v_aasist','data.py')):
        p = ROOT/folder/name
        code[str(p)] = digest(p)
    owner_identity = dict(split=split, base_checkpoint_sha256=cfg['base_checkpoint_sha256'],
        data_fingerprints=cfg['data_fingerprints'], code_fingerprints=code)
    bound = dict(identity=owner_identity, records_digest=_digest(rows), mode=MODE,
                 preprocessing_sha256=digest(Path(cfg['ssl_path'])/'preprocessor_config.json'))
    atomic_json(path/'owner.json',dict(bound, format=FORMAT, identity_digest=_digest(bound),shape=[len(rows),32]))
    atomic_json(path/'rows.json',rows)
    atomic_json(path/'complete.json',dict(format=FORMAT,owner_sha256=digest(path/'owner.json'),
                                         files={'rows.json':digest(path/'rows.json')}))


class ReleaseFixture:
    def __init__(self, folder):
        from transformers import SeamlessM4TFeatureExtractor
        self.root = Path(folder)
        torch.manual_seed(613161)
        ssl = self.root/'ssl'
        tiny_detector().backbone.save_pretrained(ssl)
        SeamlessM4TFeatureExtractor().save_pretrained(ssl)
        parent = self.root/'parent'; parent.mkdir()
        atomic_json(parent/'config.json',dict(version='fixture_previous_parent'))
        (parent/'weights.bin').write_bytes(b'immutable historical parent fixture')
        parent_hash=digest(parent/'weights.bin')
        atomic_json(parent/'completed.json',dict(checkpoint_sha256=parent_hash))
        self.source=self.root/'source'; self.source.mkdir()
        head=asdict(small_head()); head.pop('input_dim')
        self.cfg=dict(version='3.16',variant='offline_reference_tfcl_v1',init_mode='pretrained',
            seed=31601,device='cpu',amp='none',eval_amp='none',ssl_path=str(ssl),pretrained_path=str(ssl),
            pretrained_fingerprints=pretrained_assets(ssl),new_head_config=head,
            adapter_hidden=4,adapter_max_ratio=.1,trainable_layers=1,checkpointing=True,
            parent_run=str(parent),parent_checkpoint=str(parent/'weights.bin'),parent_checkpoint_sha256=parent_hash,
            parent_config_identity=identity(read_json(parent/'config.json')),parent_selected_tag='fixture',
            parent_selector='best_guarded',base_checkpoint_sha256='fixture_base',starting_checkpoint_sha256='fixture_start',
            runtime_versions=runtime_versions(),augmentation_runtime={'fixture_native_apm':True},
            source_fingerprints={str(parent/'config.json'):digest(parent/'config.json')},
            data_fingerprints={},augmentation_files={},code_fingerprints=code_fingerprints(),
            source_batch=16,microbatch=24,frame_budget=14400,feature_batch=16,eval_batch=16,workers=0,feature_workers=0,
            tfcl_heads=8,tfcl_bins=201,tfcl_time_weight=.15,tfcl_structure_weight=.045,
            encoder_lr=2e-5,head_lr=1e-4,adapter_lr=1e-4,output_lr=1e-4,tfcl_lr=1e-4,layer_decay=.8,
            weight_decay=.01,max_grad_norm=1.,rolling_cache_bytes=0,disk_margin_bytes=0,free_reserve_bytes=0,
            maximum_new_peak_bytes=1024**3,gpu_reserve_bytes=0,matched_fake_recall=.99,
            v37_run=str(self.root/'metadata'),source_config={'data_source_marker':'must_survive_continuation'})
        code=self.cfg['code_fingerprints']
        code[str((ROOT/'w2v_v316_tfcl'/'config.py').resolve())]=hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        for name in ('arguments.py','validate.py'):
            code.pop(str((ROOT/'w2v_v316_tfcl'/name).resolve()),None)
        self.train,self.dev=[],[]
        for language in ('en','zh'):
            for label in (0,1):
                for index in range(4):
                    key=f'train_{language}_{label}_{index}'
                    original=self.wave(key,'train','offline',language,label)
                    self.train.append(original)
                    online=self.wave(key+'_online','train','online',language,label)
                    online.update(source_id=original['id'],source_sha256=original['audio_sha256'],group_id=original['group_id'])
                    self.train.append(online)
                    for condition in ('noisy_a','noisy_b'):
                        self.train.append(dict(original,id=key+'_'+condition,condition=condition))
                for condition in ('online','seen','heldout'):
                    self.dev.append(self.wave(f'dev_{language}_{label}_{condition}','dev',condition,language,label))
        noise=self.root/'noise.wav'
        sf.write(noise,np.random.default_rng(711).normal(0,.05,17000).astype('float32'),16000)
        st=noise.stat(); record=dict(path=str(noise),sha256=digest(noise),size=st.st_size,mtime_ns=st.st_mtime_ns)
        self.cfg['noise_records']={'train':[record]}
        self.cfg['augmentation_files']={str(noise):digest(noise)}
        for split,rows in (('train',self.train),('dev',self.dev)):
            metadata_cache(Path(self.cfg['v37_run'])/'features'/split,rows,split,self.cfg)
        atomic_json(self.source/'config.json',self.cfg)
        atomic_json(self.source/'dev_rows.json',self.dev)
        atomic_json(self.source/'execution_plan.json',{'source_fixture':True})

    def wave(self,name,split,condition,language,label):
        path=self.root/(name+'.wav')
        frequency=120+int(hashlib.sha256(name.encode()).hexdigest()[:4],16)%1000
        x=.2*np.sin(2*np.pi*frequency*np.arange(6720)/16000)
        x+=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:16],16)).normal(0,.005,len(x))
        sf.write(path,x.astype('float32'),16000)
        st=path.stat(); sha=digest(path)
        self.cfg['data_fingerprints'][str(path)]=sha
        return dict(id=name,source_id=name,group_id=sha,source_sha256=sha,audio_sha256=sha,
            audio=str(path),audio_size=st.st_size,audio_mtime_ns=st.st_mtime_ns,output_samples=len(x),
            split=split,condition=condition,domain=condition,language=language,label=label,
            full_length=True,view='full',noisy=False)

    def commit_source(self):
        # Genuine tiny trained weights and nonempty Adam, stored in the original
        # V3.16 schema. No replacement of source_selected or verify_inputs.
        model=old_model(self.cfg); optimizer=old_optimizer(model,self.cfg)
        plan=SourcePlan(self.train,16,self.cfg['seed'],self.cfg)
        loader=TrainingLoader(plan,self.cfg,self.source)
        try:
            examples=tensors(next(loader.segment(3,0,1)))
            from w2v_v316_tfcl.step import train_step
            from w2v_v316_tfcl.objectives import TFCL
            for _ in range(4):
                train_step(model,TFCL(8,2,21),optimizer,examples,self.cfg,0.)
        finally:
            loader.close()
        logits=infer(model,self.dev,self.cfg,'source fixture Dev',self.source)
        metrics=measure(self.dev,logits)
        tag='epoch_4_step_1'; history=[dict(tag=tag,cursor=4,committed=True,metrics=metrics)]
        state=dict(schema=SOURCE_SCHEMA,identity=identity(self.cfg),model=partial_state(model),
            optimizer=to_cpu(optimizer.state_dict()),rng=capture_rng(),cursor=4,history=history,last_tag=tag,
            selections={'best_weighted':'starting_parent','best_guarded':'starting_parent'},candidates={},
            execution_plan_sha256=digest(self.source/'execution_plan.json'))
        torch.save(state,self.source/'last.pt')
        done=dict(version='3.16',variant='offline_reference_tfcl_v1',status='complete',state_file='last.pt',
            checkpoint_sha256=digest(self.source/'last.pt'),committed_updates=4,last_tag=tag,
            selections=state['selections'],execution_plan_sha256=state['execution_plan_sha256'])
        for key in ('base_checkpoint_sha256','starting_checkpoint_sha256','parent_checkpoint_sha256',
                    'parent_selected_tag','parent_selector'):done[key]=self.cfg[key]
        atomic_json(self.source/'completed.json',done)


class ReleaseTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_allowlist_is_exact_published_argument_refactor(self):
        raw=FIXTURE.read_bytes()
        self.assertIn(hashlib.sha256(raw).hexdigest(),OLD_CONFIG_HASHES)
        self.assertIn(hashlib.sha256(raw.replace(b'\n',b'\r\n')).hexdigest(),OLD_CONFIG_HASHES)
        old=ast.parse(raw);new=ast.parse((ROOT/'w2v_v316_tfcl'/'config.py').read_text())
        def functions(tree):return {x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
        a,b=functions(old),functions(new);a.pop('parser')
        a['configuration'].body.pop(0);b['configuration'].body.pop(0)
        self.assertEqual({k:ast.dump(v,include_attributes=False) for k,v in a.items()},
                         {k:ast.dump(v,include_attributes=False) for k,v in b.items()})

    def test_real_legacy_manifest_to_training_validation_and_submission(self):
        with tempfile.TemporaryDirectory() as directory,native_fixture(),redirect_stdout(io.StringIO()):
            f=ReleaseFixture(directory);f.commit_source()
            old_config=read_json(f.source/'config.json'); original_identity=identity(old_config)
            cfg=configuration(parser().parse_args(['--source-run',str(f.source),'--epochs','1',
                '--workers','0','--device','cpu','--no-autotune']))
            self.assertEqual(identity(read_json(f.source/'config.json')),original_identity)
            self.assertEqual(cfg['continuation_source_config'],old_config)
            self.assertEqual(cfg['source_config'],old_config['source_config'])
            self.assertEqual(len(cfg['source_code_migrations']),1)
            self.assertTrue(any(x.endswith('arguments.py') for x in cfg['code_fingerprints']))
            verify_inputs(cfg)
            # Real data/weight/runtime mutations must still fail despite migration.
            data=Path(f.dev[0]['audio']); saved=data.read_bytes();data.write_bytes(saved+b'changed')
            with self.assertRaisesRegex(ValueError,'Pinned input'):verify_inputs(cfg)
            data.write_bytes(saved)
            # Restore recorded mtime as well as bytes for verified audio loading.
            import os
            os.utime(data,ns=(data.stat().st_atime_ns,f.dev[0]['audio_mtime_ns']))
            unknown=copy.deepcopy(old_config)
            unknown['code_fingerprints'][str(ROOT/'w2v_v316_tfcl'/'model.py')]='0'*64
            with self.assertRaisesRegex(ValueError,'Unreviewed source-code'):verify_source(unknown)
            changed=copy.deepcopy(old_config);changed['runtime_versions']={'wrong':True}
            with self.assertRaisesRegex(ValueError,'runtime'):verify_source(changed)
            run=f.root/'continuation';run.mkdir();atomic_json(run/'config.json',cfg)
            run_experiment(cfg,run)
            state=load_resume(run,cfg)
            self.assertEqual(state['last_tag'],'epoch_5_step_1')
            self.assertEqual(state['cursor'],1)
            self.assertTrue(read_json(run/'initialization.json')['detector_parameters_with_moments']>0)
            self.assertEqual(len(state['history']),1)
            self.assertTrue(state['history'][0]['metrics']['complete'])
            for kind in ('best','last'):
                _,meta=load_selected(run,kind)
                self.assertFalse(meta['baseline_fallback'])
                self.assertIn(meta['origin']['version'],('3.16','3.16.1'))
            checked=export(run,None,None,f.root/'validation',device='cpu',workers=0,checkpoint_kind='last',dev=True)
            self.assertEqual(read_json(checked),state['history'][-1]['metrics'])
            protocol=f.root/'progress.txt'
            protocol.write_text('\n'.join(Path(r['audio']).name for r in f.dev)+'\n',encoding='utf-8')
            archive=export(run,protocol,f.root,f.root/'submission',device='cpu',workers=0,checkpoint_kind='last')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(),['scores.txt'])
                lines=z.read('scores.txt').decode().strip().splitlines()
                self.assertEqual(len(lines),len(f.dev))
            self.assertEqual(read_json(f.root/'submission'/'submission_meta.json')['selected'],'epoch_5_step_1')

    def test_source_mid_epoch_cursor_never_skips_or_repeats_remaining_batches(self):
        cfg=dict(sampling_epoch_offset=4,source_last_step=2,source_committed_updates=14)
        start=source_cursor(cfg,4)
        self.assertEqual(list(segments(start,4,4)),[(3,2,4),(4,0,2)])
        self.assertEqual(completed_tag(start+16,4),'epoch_8_step_2')
        self.assertEqual(completed_tag(16,4),'epoch_4_step_4')
        cfg['source_committed_updates']=15
        with self.assertRaisesRegex(ValueError,'cursor'):source_cursor(cfg,4)


if __name__=='__main__':unittest.main()
