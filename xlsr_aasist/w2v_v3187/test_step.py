"""Training contract: references never classify, no repeat encoder passes."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from w2v_v3183r2.common import GROUPS
from .step import train_step, schedule, _update_snapshot
from .pcl_state import PCLScale


class CountingDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Module()
        self.fusion.projection = nn.Linear(4, 128)
        self.classifier = nn.Linear(128, 2)
        self.pcl_scale = PCLScale(128)
        self.pcl_scale.observe(torch.stack((torch.zeros(128),torch.ones(128))), 'calibration-a')
        self.pcl_scale.observe(torch.stack((torch.zeros(128),torch.ones(128))), 'calibration-b')
        self.pcl_scale.freeze()
        self.encoded, self.classified, self.lookup = [], [], {}

    def encode(self, waves):
        sequences = []
        for wave in waves:
            self.encoded.append(id(wave))
            x = torch.as_tensor(wave[:5120]).reshape(16, 320)
            x = torch.stack((x.mean(-1), x.std(-1), x[:, 0], x[:, -1]), -1)
            value = self.fusion.projection(x)
            self.lookup[id(value)] = id(wave)
            sequences.append(value)
        return sequences

    def classify_sequences(self, sequences):
        self.classified.extend(self.lookup[id(value)] for value in sequences)
        return self.classifier(torch.stack([value.mean(0) for value in sequences]))


def fixture():
    cfg = dict(group_quotas=[1, 1, 1, 1], platform_sources_per_group=1, microbatch=8, frame_budget=1000,
        label_smoothing=.02, max_grad_norm=1., consistency='dual', pcl_ramp_updates=300,
        pcl_noise_weight=.15, pcl_platform_weight=.15, pcl_min_frames=1,
        epochs=6, warm_epochs=0, lr_floor=.1, lr_warmup_updates=2, risk_coefficient=.25,
        en_real_confidence_weight=0.,en_real_confidence_margin=3.5)
    rng = np.random.default_rng(36)
    units = []
    for language, label in GROUPS:
        for stream in ('online', 'offline'):
            source = f'{stream}-{language}-{label}'
            row = dict(id=source, source_id=source, group_id=source, language=language, label=label,
                       occurrence=source, split='train')
            make = lambda role: dict(row, role=role, wave=rng.normal(0, .2, 5200).astype(np.float32))
            if stream == 'online':
                units.append([make('online')])
            else:
                main = [make('noisy_a'), make('noisy_b')]
                main[0].update(pcl_refs=dict(clean=make('clean_ref'), official=make('official_ref')),
                    pcl_noisy_index=0, pcl_pairs={kind: [[0, 2000, 0, 2000], [2000, 5000, 2000, 5000]]
                                                for kind in ('noise', 'platform')},
                    pcl_coverage=dict(noise_candidates=3, platform_candidates=3))
                units.append(main)
    return cfg, units


def optimizer(model):
    return torch.optim.SGD([dict(params=list(model.parameters()), name='fixture', lr=.01, initial_lr=.01)], lr=.01)


class TestStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_one_encoding_each_and_reference_never_classified(self):
        cfg, units = fixture()
        model = CountingDetector()
        opt = optimizer(model)
        opt._v3187_joint_update = 300
        stats = train_step(model, opt, units, cfg, .25, audit=True)
        main = {id(row['wave']) for unit in units for row in unit}
        refs = {id(row['wave']) for unit in units for row in unit[0].get('pcl_refs', {}).values()}
        self.assertEqual(len(model.encoded), len(set(model.encoded)))
        self.assertEqual(set(model.encoded), main | refs)
        self.assertEqual(set(model.classified), main)
        self.assertEqual((stats['views'], stats['encoded_views'], stats['auxiliary_reference_views']), (12, 20, 8))
        self.assertGreater(stats['pcl_losses']['noise']['weighted_loss'], 0)
        self.assertGreater(stats['pcl_losses']['platform']['weighted_loss'], 0)
        self.assertIn('classification/noise', stats['pcl_gradient_audit']['cosines'])
        self.assertEqual(sum(v['candidate_phones'] for v in stats['pcl_coverage'].values()), 24)

    def test_classification_only_omits_all_auxiliary_compute(self):
        cfg, units = fixture()
        cfg['consistency'] = 'none'
        model = CountingDetector()
        stats = train_step(model, optimizer(model), units, cfg, .25)
        self.assertEqual(len(model.encoded), 12)
        self.assertEqual(stats['auxiliary_reference_views'], 0)
        self.assertEqual(stats['loss'], stats['classification_loss'])
        self.assertEqual(stats['pcl_coverage'], {})

    def test_invalid_pairs_skip_reference_forward_but_report_coverage(self):
        cfg, units = fixture()
        for unit in units:
            if len(unit) == 2:
                unit[0]['pcl_pairs'] = dict(noise=[], platform=[])
        model = CountingDetector()
        stats = train_step(model, optimizer(model), units, cfg, .25, audit=True)
        self.assertEqual(len(model.encoded), 12)
        self.assertIsNone(stats['pcl_gradient_audit'])
        self.assertEqual(sum(v['scheduled_sources'] for v in stats['pcl_coverage'].values()), 8)
        self.assertEqual(sum(v['valid_sources'] for v in stats['pcl_coverage'].values()), 0)

    def test_cross_source_reference_is_rejected(self):
        cfg, units = fixture()
        units[1][0]['pcl_refs']['official']['source_id'] = 'another-source'
        model = CountingDetector()
        with self.assertRaisesRegex(ValueError, 'match the source'):
            train_step(model, optimizer(model), units, cfg, .25)

    def test_coverage_accumulates_multiple_sources_per_group(self):
        cfg, units = fixture()
        cfg.update(group_quotas=[2, 2, 2, 2], platform_sources_per_group=2)
        extra = copy.deepcopy(units)
        for unit in extra:
            for row in list(unit) + list(unit[0].get('pcl_refs', {}).values()):
                for key in ('id', 'source_id', 'group_id', 'occurrence'):
                    row[key] += '-second'
        model = CountingDetector()
        stats = train_step(model, optimizer(model), units + extra, cfg, .25)
        for value in stats['pcl_coverage'].values():
            self.assertEqual(value['scheduled_sources'], 2)
            self.assertEqual(value['valid_sources'], 2)
            self.assertEqual(value['valid_phones'], 4)
            self.assertEqual(value['candidate_phones'], 6)

    def test_real_checkpointed_reader_ce_pcl_backward_and_audit(self):
        from .test_model import tiny_cfg
        from .model import Detector, optimizer_for
        from w2v_v318.test_core import TinyFrontend, TinyEncoder, TinyLayout
        cfg, units = fixture()
        settings = tiny_cfg()
        settings.update(cfg)
        model = Detector(TinyFrontend(), TinyEncoder(), settings, TinyLayout)
        model.pcl_scale.observe(torch.stack((torch.zeros(128),torch.ones(128))), 'calibration-a')
        model.pcl_scale.observe(torch.stack((torch.zeros(128),torch.ones(128))), 'calibration-b')
        model.pcl_scale.freeze()
        stats = train_step(model, optimizer_for(model, settings), units, settings, .25, audit=True)
        self.assertTrue(np.isfinite(stats['loss']))
        self.assertEqual(stats['encoded_views'], 20)
        self.assertGreater(sum(v for k,v in stats['module_grad_norms_before_clip'].items() if k.startswith('lora')), 0.)
        self.assertIn('classification/noise', stats['pcl_gradient_audit']['cosines'])
        self.assertFalse(any(layer._forward_hooks for layer in model.encoder.layers))

    def test_schedule_never_mutates_config_and_reaches_floor(self):
        cfg, _ = fixture()
        snapshot = copy.deepcopy(cfg)
        opt = optimizer(CountingDetector())
        schedule(opt, cfg, 0, 0, 10)
        self.assertEqual(opt._v3187_update, 1)
        schedule(opt, cfg, 5, 9, 10)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], .001)
        self.assertEqual(cfg, snapshot)

    def test_head_warmup_has_no_reference_encoding_or_auxiliary_gradient(self):
        cfg, units = fixture()
        cfg.update(warm_epochs=1, en_real_confidence_weight=.01)
        model = CountingDetector()
        model.pcl_scale=PCLScale(128)
        for unit in units:
            if len(unit)==1:
                unit[0]['pcl_calibration_spans']=[[0,2000],[2000,5000]]
        opt = optimizer(model)
        schedule(opt,cfg,0,0,10)
        stats = train_step(model,opt,units,cfg,.25)
        self.assertEqual(stats['phase'],'head_warmup')
        self.assertEqual(stats['encoded_views'],12)
        self.assertEqual(stats['auxiliary_reference_views'],0)
        self.assertEqual(stats['loss'],stats['classification_loss'])
        self.assertFalse(stats['en_real_confidence']['enabled'])
        self.assertEqual(stats['pcl_scale_calibration'],dict(source_visits=4,phones=8))
        self.assertEqual(int(model.pcl_scale.source_visits),4)
        self.assertFalse(model.pcl_scale.is_frozen)
        schedule(opt,cfg,1,0,10)
        self.assertEqual(opt._v3187_joint_update,1)
        self.assertTrue(opt._v3187_joint)

    def test_joint_rejects_missing_frozen_scale_before_encoding(self):
        cfg,units=fixture();model=CountingDetector();model.pcl_scale=PCLScale(128)
        with self.assertRaisesRegex(RuntimeError,'frozen Train phone scales'):
            train_step(model,optimizer(model),units,cfg,.25)
        self.assertEqual(model.encoded,[])

    def test_variance_has_explicit_sparse_batch_coverage(self):
        cfg,units=fixture();model=CountingDetector()
        cfg.update(pcl_variance_weight=.01,pcl_ramp_updates=1)
        stats=train_step(model,optimizer(model),units,cfg,.25)
        self.assertEqual(stats['pcl_variance']['weighted_loss'],0.)
        self.assertGreater(sum(v.get('skipped_insufficient_sources',0) for v in stats['pcl_variance']['coverage'].values()),0)

    def test_variance_valid_batch_reuses_features_and_has_gradient_audit(self):
        cfg,units=fixture();model=CountingDetector()
        cfg.update(microbatch=20,frame_budget=10000,pcl_variance_weight=.01,
                   pcl_variance_min_phones=8,pcl_variance_target=10.,pcl_ramp_updates=1)
        stats=train_step(model,optimizer(model),units,cfg,.25,audit=True)
        self.assertEqual(stats['encoded_views'],20)
        self.assertEqual(len(model.encoded),len(set(model.encoded)))
        self.assertGreater(stats['pcl_variance']['weighted_loss'],0.)
        self.assertEqual(stats['pcl_variance']['coverage']['noise']['applied'],1)
        self.assertIn('classification/variance',stats['pcl_gradient_audit']['cosines'])
        total=stats['classification_loss']+sum(v['weighted_loss'] for v in stats['pcl_losses'].values())
        self.assertAlmostEqual(stats['loss'],total+stats['pcl_variance']['weighted_loss'],places=6)

    def test_all_depth_lora_roles_receive_warmup_and_joint_ramp(self):
        cfg,_=fixture();cfg.update(warm_epochs=1)
        parameters=[nn.Parameter(torch.ones(1)) for _ in range(4)]
        groups=[dict(params=[p],name='lora_'+name,role='lora',lr=.01,initial_lr=.01)
                for p,name in zip(parameters[:3],('early','middle','late'))]
        groups.append(dict(params=[parameters[-1]],name='head',lr=.02,initial_lr=.02))
        opt=torch.optim.SGD(groups,lr=.01)
        schedule(opt,cfg,0,0,10)
        self.assertEqual([g['lr'] for g in opt.param_groups],[0.,0.,0.,.02])
        schedule(opt,cfg,1,0,10)
        self.assertEqual([g['lr'] for g in opt.param_groups],[.005,.005,.005,.02])

    def test_update_audit_samples_lora_b_even_when_all_a_gradients_are_zero(self):
        parameters=[nn.Parameter(torch.ones(2000)) for _ in range(6)]
        for index,value in enumerate(parameters):
            value.grad=torch.zeros_like(value) if index%2==0 else torch.ones_like(value)
        opt=torch.optim.SGD([dict(params=parameters,name='lora_early',lr=.1)],lr=.1)
        snapshots=_update_snapshot(opt)
        self.assertEqual(len(snapshots),6)
        self.assertEqual(sum(item[2] for item in snapshots),1024)
        opt.step()
        changes=[float((parameter.detach().flatten()[:count]-before).square().sum())
                 for _,parameter,count,before in snapshots]
        self.assertTrue(all(changes[i]==0 for i in (0,2,4)))
        self.assertTrue(all(changes[i]>0 for i in (1,3,5)))

    def test_variable_actual_quota_metadata_overrides_static_configuration(self):
        cfg, units = fixture()
        cfg['group_quotas']=[6,6,6,6]
        for unit in units:
            for row in unit:
                row['batch_group_quotas']=(1,1,1,1)
            unit[0]['pcl_platform_budget']=1
        model=CountingDetector()
        stats=train_step(model,optimizer(model),units,cfg,.25)
        self.assertEqual(stats['actual_group_quotas'],[1,1,1,1])

    def test_physical_partition_preserves_classification_pcl_and_cap_update(self):
        cfg,units=fixture()
        cfg.update(en_real_confidence_weight=.01,pcl_ramp_updates=1,frame_budget=10000)
        torch.manual_seed(33)
        first=CountingDetector()
        with torch.no_grad():first.classifier.bias[1].fill_(6.)
        second=copy.deepcopy(first)
        a=train_step(first,optimizer(first),units,dict(cfg,microbatch=4),.25)
        b=train_step(second,optimizer(second),units,dict(cfg,microbatch=20),.25)
        self.assertGreater(a['en_real_confidence']['weighted_loss'],0)
        self.assertAlmostEqual(a['loss'],b['loss'],places=6)
        for left,right in zip(first.parameters(),second.parameters()):
            torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-7)


if __name__ == '__main__':
    unittest.main()
