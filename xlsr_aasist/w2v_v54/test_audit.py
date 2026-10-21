"""Regression checks for the design-versus-implementation audit."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .test_integration import fixture
from .test_data import row
from .data import pair_info
from .alignment import align_pair
from .diagnostics import encode_bounded, probe
from .objective import paired_losses
from .side import _layout
from .model import optimizer_for
from .step import train_step
from .schedule import schedule, auxiliary_coefficients
from .common import partial_state,apply_partial,atomic_save,fingerprint


def fixed_pair_file(model,cfg,units,path):
    selected = next(u[0] for u in units if u[0]['role']=='noisy')
    model.eval()
    with torch.no_grad():
        c,n = model.encode([selected['pcl_refs']['clean']['wave'],selected['wave']])
        logits = model.classify_sequences([c,n]).cpu()
    info = selected['repair_pair']
    item = dict(identity=dict(source=selected['source_id'],group=f'{selected["language"]}/{selected["label"]}',
                    origin='dev',condition='test',seconds=len(selected['wave'])/16000.,snr_db=0.),
        label=selected['label'],clean_wave=torch.as_tensor(selected['pcl_refs']['clean']['wave']),
        noisy_wave=torch.as_tensor(selected['wave']),clean_feature=c.cpu(),noisy_feature=n.cpu(),
        scale_valid=torch.ones(len(c),dtype=torch.bool),initial_logits=logits,
        info={k:torch.as_tensor(v) if isinstance(v,np.ndarray) else v for k,v in info.items()})
    atomic_save(path,dict(config=fingerprint(cfg),pairs=[item]),0)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_shared_alignment_matches_inherited_rules_and_resamples_only_twice(self):
        from w2v_v3186.augment import estimate_delay,reliable_spans
        from scipy.signal import resample_poly
        a=np.random.default_rng(54).normal(0,.1,9600).astype(np.float32)
        spans=[(0,2560),(2560,5120),(5120,7680)]
        for shift in (-320,0,320):
            b=np.roll(a,shift)*.8
            expected=estimate_delay(a,b)
            with patch('w2v_v54.alignment.resample_poly',wraps=resample_poly) as convert:
                delay,valid=align_pair(a,b,spans)
                self.assertEqual(convert.call_count,2)
            self.assertEqual(delay,expected)
            self.assertEqual(valid,reliable_spans(a,b,spans,0.,delay['offset']))

    def test_pair_provenance_rejects_different_rtc_or_base(self):
        a=row(np.ones(6400,np.float32));b=copy.deepcopy(a)
        a['augmentation']=dict(recipe=dict(codec='opus'),common_gain=1.)
        b['augmentation']=dict(recipe=dict(codec='opus'),common_gain=.8)
        with self.assertRaisesRegex(ValueError,'settings'):pair_info(a,b)
        b['augmentation']=copy.deepcopy(a['augmentation'])
        a['base_augmentation']=dict(recipe='one')
        with self.assertRaisesRegex(ValueError,'base'):pair_info(a,b)

    def test_invalid_auxiliary_is_skipped_and_unqualified_teacher_has_no_student_forward(self):
        cfg,units,m,_=fixture();m.eval()
        unit=next(u for u in units if u[0]['role']=='noisy' and u[0]['meta_side']=='A')
        rows,slots=_layout(unit,dict(noise=0,platform=0))
        seq=m.encode([r['wave'] for r in rows]);processed=m.processed_sequences(seq)
        settings=dict(cfg,current_update=5,_repair_warm_steps=1,repair_teacher_confidence=1.)
        args=(m,[unit],[(0,slots)],seq,processed,[0],[1],[1]*4,settings)
        with patch.object(m,'classify_processed',wraps=m.classify_processed) as head:
            f,d,records=paired_losses(*args)
            self.assertGreater(float(f),0);self.assertEqual(float(d),0)
            self.assertEqual(head.call_count,1)  # Teacher only.
            self.assertFalse(records[0]['kd_student_evaluated'])
        unit[0]['repair_pair']['noisy_valid'][:]=False
        with patch.object(m.repair,'auxiliary_prediction',side_effect=AssertionError('invalid LF graph')),patch.object(m,'classify_processed',side_effect=AssertionError('invalid KD backend')):
            f,d,records=paired_losses(*args)
            self.assertEqual(float(f+d),0);self.assertFalse(records[0]['valid'])

    def test_small_budget_kd_ramp_reaches_configured_global_weight(self):
        cfg,_,m,_=fixture();cfg['repair_kd_ramp']=200
        opt=optimizer_for(m,cfg);schedule(opt,cfg,0,7,8)
        f,d=auxiliary_coefficients(dict(cfg,current_update=8,_repair_warm_steps=opt._v54_warm_steps,
                                       _repair_total_updates=opt._v54_total_updates))
        self.assertEqual(f,cfg['repair_feature_weight']);self.assertEqual(d,cfg['repair_kd_weight'])

    def test_gradient_audit_has_correct_boundary_and_does_not_change_update(self):
        from w2v_v3191.meta import rng_state,restore_rng
        cfg,units,m,_=fixture();other=copy.deepcopy(m)
        for unit in units:
            if unit[0]['role']=='noisy':
                unit[0]['augmentation']=dict(noise_hashes=['fixed-noise-hash'],noise_original_sources=['fixed-noise-source'])
        opt=optimizer_for(m,cfg);other_opt=optimizer_for(other,cfg)
        schedule(opt,cfg,0,4,20);schedule(other_opt,cfg,0,4,20)
        rng=rng_state();result=train_step(m,opt,units,cfg,audit=True)
        restore_rng(rng);train_step(other,other_opt,units,cfg,audit=False)
        for k,v in partial_state(m).items():torch.testing.assert_close(v,partial_state(other)[k],rtol=0,atol=0)
        norms=result['repair_gradient_audit']['norms']
        self.assertGreater(norms['feature']['repair_output'],0)
        for name,value in norms['feature'].items():
            if name!='repair_output':self.assertEqual(value,0)
        self.assertGreater(norms['ce']['classifier'],0)
        self.assertGreater(result['backend_windows'],0)
        self.assertEqual(result['repair_losses']['coefficients']['inner']*result['repair_losses']['coefficients']['outer'],1.)
        noisy_examples=[r for r in result['hard_examples']['examples'] if r['view']=='noisy']
        self.assertTrue(noisy_examples)
        self.assertTrue(all(r['augmentation']['noise_hashes']==['fixed-noise-hash'] for r in noisy_examples))

    def test_fixed_probe_measures_real_output_and_restores_rng_modes(self):
        from w2v_v3191.meta import rng_state
        from w2v_v3191.test_step import assert_rng_equal
        cfg,units,m,_=fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            run=Path(directory);fixed_pair_file(m,cfg,units,run/'fixed_pairs.pt')
            m.train();rng=rng_state();modes=[x.training for x in m.modules()]
            first=probe(m,cfg,run,'initial',cached_initial=True)
            self.assertEqual(first['records'][0]['correction_ratio'],0)
            m.repair.out.bias.data.add_(torch.linspace(-.3,.5,128))
            result=probe(m,cfg,run,'changed')
            row=result['records'][0]
            self.assertGreater(row['correction_ratio'],0)
            self.assertGreater(row['max_bypass_probability_change'],0)
            self.assertEqual(row['reference_drift'],0)
            self.assertEqual([x.training for x in m.modules()],modes)
            assert_rng_equal(self,rng,rng_state())

    def test_diagnostic_frame_budget_and_invalid_fixed_scale(self):
        cfg,units,m,_=fixture()
        waves=[u[0]['wave'] for u in units[:4]]
        budget=max((len(w)-400)//320+1 for w in waves)
        with patch.object(m,'encode',wraps=m.encode) as encode:
            encode_bounded(m,waves,dict(cfg,frame_budget=budget))
            self.assertEqual(encode.call_count,4)
        values=partial_state(m);values['repair.frame_scale'][0]=0
        with self.assertRaisesRegex(ValueError,'strictly positive'):apply_partial(m,values)

    def test_declared_architecture_has_not_been_reduced(self):
        from .repair import FeatureRepair
        model=FeatureRepair()
        self.assertEqual(sum(p.numel() for p in model.parameters()),197184)
        self.assertEqual(len(model.blocks),2)
        for block in model.blocks:
            self.assertEqual([m.kernel_size for m in block.filters],[(3,),(7,),(15,)])
            self.assertEqual([m.groups for m in block.filters],[256]*3)
        self.assertEqual(model.gate[1].in_features,384)
        self.assertFalse(any(isinstance(m,(torch.nn.Dropout,torch.nn.modules.batchnorm._BatchNorm)) for m in model.modules()))

    def test_intermediate_stage_replay_matches_original_endpoints_and_shared_gain(self):
        from .inspect_receiver import stage_waves
        from w2v_v5.processing import generate_matched_context
        rec=dict(seed=54,category='echo',echo=dict(delay_samples=160,gain=.2),input_gain_db=-1.,
                 family='bypass',codec='none')
        a=np.random.default_rng(7).normal(0,.1,6400).astype(np.float32)
        engine=lambda wave,rec:(wave*.8).astype(np.float32)
        expected_c,expected_n,meta=generate_matched_context(a,rec,None,engine,context_seconds=.1)
        stages,new=stage_waves(a,rec,None,engine,.1)
        np.testing.assert_array_equal(stages['clean_rtc'],expected_c)
        np.testing.assert_array_equal(stages['noisy_rtc'],expected_n)
        np.testing.assert_array_equal(stages['noisy_rtc_echo_off'],expected_c)
        np.testing.assert_array_equal(stages['scene_before_rtc']*.8,expected_n)
        self.assertEqual(meta['common_gain'],new['common_gain'])

    def test_runtime_check_restores_weights_optimizer_and_rng_before_actual_training(self):
        from .smoke import check
        from w2v_v3191.meta import rng_state
        from w2v_v3191.test_step import assert_rng_equal
        cfg,units,m,_=fixture();opt=optimizer_for(m,cfg)
        saved=partial_state(m);rng=rng_state()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            result=check(m,opt,units,cfg,20,Path(directory)/'check.json')
        self.assertEqual(result['committed_training_updates'],0)
        self.assertEqual([x['stage'] for x in result['checks']],['repair_warmup','joint'])
        self.assertEqual(opt.state_dict()['state'],{})
        for key,value in saved.items():torch.testing.assert_close(partial_state(m)[key],value,rtol=0,atol=0)
        assert_rng_equal(self,rng,rng_state())

    def test_observation_failure_is_detected_before_optimizer_commit(self):
        from w2v_v3191.meta import rng_state
        from w2v_v3191.test_step import assert_rng_equal
        cfg,units,m,_=fixture();opt=optimizer_for(m,cfg)
        before=partial_state(m);rng=rng_state()
        with patch('w2v_v54.telemetry.collect',side_effect=RuntimeError('bad attribution')):
            with self.assertRaisesRegex(RuntimeError,'attribution'):train_step(m,opt,units,cfg)
        self.assertEqual(opt.state_dict()['state'],{})
        for key,value in before.items():torch.testing.assert_close(partial_state(m)[key],value,rtol=0,atol=0)
        assert_rng_equal(self,rng,rng_state())


if __name__=='__main__':unittest.main()
