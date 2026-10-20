"""Registered-source preparation and actual dual-PCL pipeline integration."""
from contextlib import ExitStack,redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .common import atomic_json,read_json,digest,fingerprint,capture_rng
from .test_core import Tiny,fixture,cfg
from .calibration import calibrate
from .phonemes import build_pairs
from .data import Waves
from .sampling import Plan
from .source import verify_source
from .config import parser,prepare_missing,initialize,verify
from .dataset import transfer
from .test_dataset import prepared_fixture
from w2v_v31810.identity import registry_config
from w2v_v31810.test_meta import assert_state_equal
from w2v_v31813.test_preparation import phone_fixture
from .interface_smoke import main as native_smoke


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_actual_native_hf_update_has_two_pcl_gradients(self):
        with redirect_stdout(io.StringIO()):result=native_smoke('cpu','fp32')
        self.assertEqual(result['joint']['pcl_pairs'],{'official':4,'noise':4})
        self.assertTrue(all(v>0 for v in result['joint']['pcl_gradient_norms'].values()))
    def test_calibration_is_observational_and_reports_actual_scale(self):
        model=Tiny().train();state=copy.deepcopy(model.state_dict());rng=capture_rng()
        with redirect_stdout(io.StringIO()):result=calibrate(model,[fixture(),fixture()],cfg())
        self.assertEqual(set(result['weights']),{'official','noise'})
        self.assertTrue(all(0<v<=1 for v in result['weights'].values()))
        self.assertTrue(all(v>0 for v in result['achieved_ratios'].values()))
        assert_state_equal(self,state,model.state_dict());assert_state_equal(self,rng,capture_rng())
        self.assertTrue(model.training);self.assertTrue(all(p.grad is None for p in model.parameters()))
    def test_calibration_distinguishes_missing_from_connected_zero(self):
        units=fixture()
        for u in units:u[0]['pcl']['reference']['wave']=u[0]['wave'].copy()
        with redirect_stdout(io.StringIO()):result=calibrate(Tiny(),[units],cfg())
        self.assertEqual(result['weights'],{'official':1.,'noise':1.})
        for u in units:u[0].pop('pcl')
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'No valid official'):
            calibrate(Tiny(),[units],cfg())
    def test_registered_originals_bind_both_branches_on_new_frame_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=phone_fixture(root)
            phones=build_pairs(rows,root/'manifest.json',root/'pairs.json')
            self.assertEqual(phones['owner']['geometry']['receptive'],560)
            self.assertEqual(set(phones['official']),{r['id'] for r in rows if r['condition']=='online'})
            self.assertEqual(set(phones['noise']),{r['id'] for r in rows if r['condition']=='offline'})
            self.assertEqual(phones,build_pairs(rows,root/'manifest.json',root/'pairs.json'))
    def test_simulated_pcl_reuses_exact_ce_waveform_and_offline_only_source(self):
        from w2v_v31810.data import Waves as BaseWaves
        from w2v_v3189.data import prepared
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=phone_fixture(root)
            phones=build_pairs(rows,root/'manifest.json',root/'pairs.json')
            off=next(r for r in rows if r['condition']=='offline')
            wave=np.random.default_rng(5).normal(0,.1,off['output_samples']).astype(np.float32)
            noisy=wave+.002*np.random.default_rng(6).normal(size=len(wave)).astype(np.float32)
            main=prepared(off,noisy,role='noisy',augmentation=dict(pair_eligible=True,erased_spans=[]))
            dataset=Waves([off],cfg())
            ticket=dict(index=0,pcl_noise=phones['noise'][off['id']])
            with patch.object(BaseWaves,'__getitem__',return_value=[main]) as ce,patch.object(dataset,'original',return_value=wave):
                result=dataset[ticket][0]
            self.assertEqual(ce.call_count,1);self.assertIs(result['wave'],main['wave'])
            self.assertTrue(result['pcl']['reuse_received']);self.assertIsNone(result['pcl']['received'])
            self.assertEqual(result['pcl']['kind'],'noise');self.assertTrue(result['pcl']['matches'])
    def test_real_sampler_preserves_exposure_with_both_pcl_branches(self):
        from collections import Counter
        with tempfile.TemporaryDirectory() as tmp:
            settings,train,dev,protocol,registry=prepared_fixture(tmp,extra_train=6)
            rows,_,protocol,registry,_=transfer(train,dev,protocol,registry,settings)
            settings.update(source_registry=registry_config(registry),source_draws_per_stream=[32]*4,
                stream_sources=8,group_quotas=[2]*4,view_factors=[1,5,1,5],refresh_base_variants=True,
                source_epoch_offset=0,experiment='E0',warm_epochs=1,consistency='dual')
            phones={b:{r['id']:dict(source=r['canonical_source']) for r in rows if r['condition']==c}
                    for b,c in (('official','online'),('noise','offline'))}
            plan=Plan(rows,settings,phones);counts=Counter()
            for batch in plan.epoch_batches(1):
                self.assertEqual(len({t['canonical_source'] for t in batch}),len(batch))
                self.assertEqual(sum('pcl_official' in t for t in batch),4)
                self.assertEqual(sum('pcl_noise' in t for t in batch),4)
                self.assertFalse(any(any('xeus' in k for k in t) for t in batch))
                for t in batch:
                    r=rows[t['index']];counts[t['kind'],r['language'],r['label']]+=1
            self.assertEqual(set(counts.values()),{32})
    def test_preparation_never_requires_checkpoint_or_xeus(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);data=root/'data';data.mkdir();parent=root/'source';parent.mkdir()
            atomic_json(data/'bert_assets.json',dict(bert_model_dir=str(root/'bert'),bert_revision=None))
            (data/'phonemes').mkdir();atomic_json(data/'phonemes/manifest.json',{})
            args=parser().parse_args(['--data-run',str(parent),'--bert-model-dir',str(root/'bert')])
            stack.enter_context(patch('w2v_v4.config.prepare_data',return_value=(data,{})))
            stack.enter_context(patch('w2v_v4.config.resolve_parent',return_value=parent))
            stack.enter_context(patch('w2v_v4.config.verify_assets'))
            sub=stack.enter_context(patch('w2v_v4.config.subprocess.run'))
            self.assertEqual(prepare_missing(args),data);sub.assert_not_called()
    def test_data_source_verification_without_model_or_teacher_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings,train,dev,protocol,registry=prepared_fixture(root/'audio')
            settings.update(version='3.18.13',source_registry=registry_config(registry),teacher_binding=dict(path='missing'))
            for name,value in (('config.json',settings),('train_rows.json',train),('validation_protocol.json',protocol),('source_registry.json',registry)):
                atomic_json(root/name,value)
            self.assertEqual(verify_source(root),settings)
            changed=read_json(root/'train_rows.json');changed[0]['canonical_source']='bad';atomic_json(root/'train_rows.json',changed)
            with self.assertRaises(ValueError):verify_source(root)

    def test_fresh_config_is_independent_of_parent_weights_and_verifies_for_export(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            root=Path(tmp);settings,train,dev,protocol,registry=prepared_fixture(root/'audio',extra_train=6)
            settings.update(version='3.18.13',source_registry=registry_config(registry),source_draws_per_stream=[32]*4,
                stream_sources=8,group_quotas=[2]*4,view_factors=[1,5,1,5],refresh_base_variants=True,
                source_epoch_offset=0,experiment='E0',warm_epochs=1,consistency='none',seed=4001,
                validation_input_files={},teacher_binding=dict(path='absent'),omni_checkpoint='absent',
                initialization=dict(mode='old'),eval_batch=8)
            parent=root/'parent';parent.mkdir();atomic_json(parent/'config.json',settings)
            rows,remaining,protocol,registry,transfer_report=transfer(train,dev,protocol,registry,settings)
            data=root/'prepared';data.mkdir()
            names=('train_rows.json','dev_rows.json','validation_protocol.json','source_registry.json','source_transfer.json')
            for name,value in zip(names,(rows,remaining,protocol,registry,transfer_report)):atomic_json(data/name,value)
            staging=dict(settings,data_recipe='v4_source_transfer_v2',data_owner=dict(parent_config=digest(parent/'config.json')),
                source_registry=registry_config(registry),validation_protocol_hash=fingerprint(protocol),validation_input_files={},
                prepared_files={name:digest(data/name) for name in names})
            atomic_json(data/'config.json',staging)
            atomic_json(data/'bert_assets.json',dict(bert_model_dir='official-fixture',transformers_version='4.38.2',bert_files={}))
            stack.enter_context(patch('w2v_v4.config.resolve_parent',return_value=parent))
            stack.enter_context(patch('w2v_v4.config.verify_source',return_value=settings))
            stack.enter_context(patch('w2v_v4.config.verify_assets'))
            stack.enter_context(patch('torch.cuda.is_available',return_value=True))
            stack.enter_context(patch('torch.cuda.is_bf16_supported',return_value=True))
            stack.enter_context(patch('transformers.__version__','4.38.2'))
            args=parser().parse_args(['--pcl','none','--data-run',str(parent),'--epochs','2'])
            config,observed,_=initialize(args,root/'new',data)
            self.assertEqual((config['encoder_dim'],config['lora_layers'],config['pooling_heads']),(1024,24,4))
            self.assertFalse(config['initialization']['detector_weights_loaded'])
            self.assertNotIn('teacher_binding',config);self.assertNotIn('omni_checkpoint',config)
            self.assertFalse((root/'new/init.pt').exists());self.assertEqual(rows,observed)
            self.assertEqual(verify(root/'new',training=False),config)


if __name__=='__main__':unittest.main()
