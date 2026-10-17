"""CPU acceptance of the fixed fresh-model, schedule and resource contract."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from .common import RECIPE, schema_for, fingerprint
from .config import parser, validate, validate_locked_cli, training_configuration, code_inventory, configuration
from .model import optimizer_for
from .resources import audit_training, audit_inference
from .sampling import quota_schedule
from .schedule import schedule
from .workers import settings


def inherited():
    return dict(encoder_layers=60, omni_arch='3b', encoder_dim=2048, pooling='asp',
        local_evidence=True, window=128, hop=64, window_batch=8, whole_max_frames=3000,
        initialization={'files': {'old/best.pt': 'hash'}}, parent_init={'run': 'old'},
        saved_manifests={'old/last.pt': 'hash'}, warm_start={'run': 'old'},
        learning_rate_origin='parent-best', warm_lr=.1, consistency='dual',
        pcl_noise_weight=.15, pcl_platform_weight=.045, pcl_variance_weight=.01,
        en_real_confidence_weight=.1, phoneme_provenance={'old': 'hash'},
        data_files={'old/last.pt': 'hash'}, workers=4, raw_audio_cache_mib=128,
        noise_cache_mib=128, scene_cache_mib=32)


def fresh():
    old = inherited(); args = parser().parse_args([])
    source = dict(run='data/run', files={'data/run/train_rows.json': 'hash'})
    host = dict(usable_cpu_cores=16, available_memory_bytes=None)
    with patch('w2v_v31810.config.runtime_versions', return_value={}), \
         patch('w2v_v31810.config.settings', side_effect=lambda a, o: settings(a, o, host)):
        result = training_configuration(old, args, source)
    return result


class Adapter(nn.Module):
    def __init__(self):
        super().__init__(); self.lora_a = nn.Parameter(torch.ones(2, 3)); self.lora_b = nn.Parameter(torch.zeros(3, 2))


class OptimizerFixture(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = nn.Module(); self.encoder.layers = nn.ModuleList(Adapter() for _ in range(60))
        self.encoder.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
        self.head = nn.Module(); self.head.aasist = nn.Module()
        self.head.aasist.hidden = nn.Linear(3, 3); self.head.aasist.out_layer = nn.Linear(3, 2)
        self.head.contribution = nn.Linear(3, 1); self.head.attention_delta = nn.Linear(3, 1)
        self.fusion = nn.Module(); self.fusion.projection = nn.Linear(3, 3)
        self.fusion.layer_logits = nn.Parameter(torch.zeros(60))


class ContractTests(unittest.TestCase):
    def test_fresh_official_contract_overwrites_parent_training_state(self):
        cfg = fresh()
        self.assertEqual((cfg['version'], cfg['recipe'], cfg['experiment']), ('3.18.10', RECIPE, 'E2'))
        self.assertEqual((cfg['epochs'], cfg['warm_epochs']), (15, 1))
        self.assertEqual(cfg['initialization']['mode'], 'pretrained')
        self.assertFalse(cfg['initialization']['old_detector_loaded']); self.assertFalse(cfg['initialization']['old_optimizer_loaded'])
        self.assertEqual(cfg['initialization'].get('files', {}), {})
        self.assertEqual(cfg['data_files'], {'data/run/train_rows.json': 'hash'})
        for name in ('parent_init', 'warm_start', 'saved_manifests', 'learning_rate_origin', 'warm_lr', 'phoneme_provenance'):
            self.assertNotIn(name, cfg)
        self.assertEqual(cfg['consistency'], 'none'); self.assertIsNone(cfg['phoneme_manifest'])
        for name in ('pcl_noise_weight', 'pcl_platform_weight', 'pcl_variance_weight', 'en_real_confidence_weight'):
            self.assertEqual(cfg[name], 0.)
        self.assertEqual((cfg['lora_layers'], cfg['lora_rank'], cfg['lora_alpha']), (60, 16, 32.))
        self.assertEqual(cfg['fusion_layers'], list(range(1, 61)))
        self.assertEqual((cfg['window'], cfg['hop'], cfg['fusion_dim']), (128, 64, 128))
        self.assertEqual(cfg['normalization'], 'group_layer'); self.assertTrue(cfg['temporal_stats'])
        self.assertEqual(cfg['label_smoothing'], .02); self.assertEqual(cfg['max_grad_norm'], 1.)
        self.assertEqual(cfg['source_epoch_offset'], 0); self.assertTrue(cfg['refresh_base_variants'])
        quotas = quota_schedule(tuple(cfg['source_draws_per_stream']), cfg['stream_sources'])
        self.assertEqual((len(quotas), sum(quotas[-1]), len(quotas) * cfg['epochs']), (1403, 20, 21045))

    def test_all_sixty_lora_layers_and_backend_groups_have_prescribed_rates_and_decay(self):
        cfg = fresh(); model = OptimizerFixture(); optimizer = optimizer_for(model, cfg)
        groups = optimizer.param_groups
        for name, lr in (('lora_early', 5e-6), ('lora_middle', 1e-5), ('lora_late', 2e-5)):
            selected = [g for g in groups if g['name'] == name]
            self.assertEqual(sum(len(g['params']) for g in selected), 40)
            self.assertTrue(all(g['initial_lr'] == lr and g['role'] == 'lora' for g in selected))
        for group in groups:
            if group['role'] != 'lora': self.assertEqual(group['initial_lr'], 1e-4)
            for parameter in group['params']:
                self.assertEqual(group['weight_decay'], 1e-4 if parameter.ndim >= 2 else 0.)
        actual = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(actual), len(set(actual))); self.assertNotIn(id(model.encoder.frozen), actual)

    def test_head_warmup_then_lora_two_hundred_updates_beta_ramp_and_cosine_floor(self):
        cfg = fresh(); optimizer = optimizer_for(OptimizerFixture(), cfg); steps = 1403
        joint, beta = schedule(optimizer, cfg, 0, steps - 1, steps)
        self.assertFalse(joint); self.assertEqual(beta, 0.)
        self.assertTrue(all(g['lr'] == (0. if g['role'] == 'lora' else g['initial_lr']) for g in optimizer.param_groups))
        joint, beta = schedule(optimizer, cfg, 1, 0, steps)
        self.assertTrue(joint); self.assertEqual(beta, 0.)
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group['lr'] / group['initial_lr'], 1/200 if group['role'] == 'lora' else 1.)
        schedule(optimizer, cfg, 1, 199, steps)
        rates = {g['role']: g['lr'] / g['initial_lr'] for g in optimizer.param_groups}
        self.assertAlmostEqual(rates['lora'], rates['head']); self.assertLess(rates['head'], 1.)
        self.assertAlmostEqual(schedule(optimizer, cfg, 2, steps - 1, steps)[1], .5)
        self.assertEqual(schedule(optimizer, cfg, 3, 0, steps)[1], .5)
        schedule(optimizer, cfg, 14, steps - 1, steps)
        self.assertTrue(all(abs(g['lr'] / g['initial_lr'] - .1) < 1e-12 for g in optimizer.param_groups))

    def test_schema_rejects_old_versions_and_any_reintroduced_auxiliary_objective(self):
        cfg = fresh(); self.assertIn('metadg', schema_for(cfg))
        for change in ({'version': '3.18.9'}, {'recipe': 'old'}, {'consistency': 'dual'},
                       {'pcl_noise_weight': .1}, {'pcl_platform_weight': .1},
                       {'pcl_variance_weight': .1}, {'en_real_confidence_weight': .1},
                       {'noisy_views_per_source': 2}, {'hop': 129}):
            with self.subTest(change=change), self.assertRaises(ValueError): schema_for(dict(cfg, **change))

    def test_worker_queue_estimate_is_forty_eight_classification_views_not_legacy_refs(self):
        args = parser().parse_args([]); old = inherited(); old['stream_sources'] = 999
        result = settings(args, old, dict(usable_cpu_cores=32, available_memory_bytes=None))
        self.assertEqual((result['workers'], result['eval_workers'], result['prefetch_factor']), (8, 4, 1))
        cache = sum(result[k] for k in ('raw_audio_cache_mib', 'noise_cache_mib', 'scene_cache_mib'))
        queue = math.ceil(48 * (3000 * 320 + 400) * 4 / 1024**2)
        self.assertEqual(result['worker_resource_audit']['estimated_pool_bytes'], 12 * (512 + cache + queue) * 1024**2)
        self.assertLessEqual(8 * cache, result['worker_resource_audit']['parent_aggregate_cache_mib'])
        self.assertEqual(result['worker_resource_audit']['native_threads_per_worker'], 1)

    def test_resource_guards_require_full_content_and_only_one_classification_view(self):
        cfg = fresh()
        train = [dict(id='off', source_id='origin', condition='offline', split='train', output_samples=96000),
                 dict(id='on', source_id='origin', condition='online', split='train', output_samples=1000)]
        dev = [dict(id='dev', output_samples=2000)]
        report = audit_training(train, dev, cfg)
        self.assertTrue(report['valid']); self.assertEqual(report['maxima']['atomic_views'], 1)
        self.assertEqual(report['maxima']['auxiliary_frames'], 0)
        self.assertEqual(report['dev']['maxima']['classification_frames'], 12)
        with self.assertRaisesRegex(ValueError, 'no audio truncation|not decoded|cropped|no silent|preflight failed'):
            audit_inference([dict(id='too_long', output_samples=2_000_000)], cfg)

    def test_resume_does_not_accept_recipe_changes(self):
        args = parser().parse_args(['--resume', 'run'])
        validate_locked_cli(args, ['--resume', 'run'])
        validate_locked_cli(args, ['--resume', 'run', '--submit-best', '--upload-submission'])
        for option in ('--epochs', '--experiment', '--workers', '--backend-precision'):
            with self.assertRaisesRegex(ValueError, 'Resume uses saved'): validate_locked_cli(args, ['--resume', 'run', option, 'changed'])
        with self.assertRaises(ValueError): validate(parser().parse_args(['--epochs', '3']))

    def test_configuration_audits_new_protocol_audio_not_obsolete_cached_dev(self):
        source = Path('prepared-data-run')
        train = [dict(id='train', audio='/actual/train.wav')]
        historical = [dict(id='legacy-noisy', audio='/deleted/old-dev-cache.wav')]
        selected = [dict(id='select-online', audio='/actual/dev-online.wav'),
                    dict(id='select-noisy', audio='/actual/dev-offline.wav')]
        audited = [dict(id='audit-online', audio='/actual/audit-online.wav'),
                   dict(id='audit-noisy', audio='/actual/audit-offline.wav')]
        protocol = dict(select={'rows': selected}, audit={'rows': audited},
            input_files={'/actual/dev-online.wav': 'verified-hash', '/actual/dev-pairs.csv': 'pair-hash'})
        provenance = dict(run=str(source), files={'prepared-data-run/train_rows.json': 'manifest-hash'})
        events = []
        def build(cfg, observed_train, observed_dev, observed_source):
            self.assertIs(observed_train, train); self.assertIs(observed_dev, historical)
            self.assertEqual(observed_source, source); events.append('protocol')
            return protocol
        def resources(observed_train, observed_dev, cfg):
            self.assertEqual(events, ['protocol']); events.append('resources')
            self.assertIs(observed_train, train); self.assertEqual(observed_dev, selected + audited)
            self.assertNotIn('/deleted/old-dev-cache.wav', [row['audio'] for row in observed_dev])
            return {'new_protocol_only': True}
        def read(path):
            return train if Path(path).name == 'train_rows.json' else historical
        host = dict(usable_cpu_cores=16, available_memory_bytes=None)
        with patch('w2v_v31810.config.resolve_data_run', return_value=source), \
             patch('w2v_v31810.config.metadata', return_value=(inherited(), provenance)), \
             patch('w2v_v31810.config.read_json', side_effect=read), \
             patch('w2v_v31810.config.torch.cuda.is_available', return_value=True), \
             patch('w2v_v31810.config.torch.cuda.is_bf16_supported', return_value=True), \
             patch('w2v_v31810.config.runtime_versions', return_value={}), \
             patch('w2v_v31810.config.settings', side_effect=lambda a, o: settings(a, o, host)), \
             patch('w2v_v31810.validation.build_protocol', side_effect=build) as build_mock, \
             patch('w2v_v31810.resources.audit_training', side_effect=resources) as resources_mock, \
             patch('w2v_v31810.sampling.Plan') as plan, \
             patch('w2v_v31810.config.code_inventory', return_value={'new-code.py': 'code-hash'}), \
             patch('w2v_v31810.config.verify_inputs') as verify:
            plan.return_value.audit_budget.return_value = {'exposure_budget': 'unchanged'}
            output = configuration(parser().parse_args([]))
        self.assertEqual(len(output), 4)
        cfg, actual_train, actual_dev, actual_protocol = output
        self.assertIs(actual_train, train); self.assertIs(actual_dev, historical); self.assertIs(actual_protocol, protocol)
        self.assertEqual(events, ['protocol', 'resources'])
        self.assertEqual(build_mock.call_count, 1); self.assertEqual(resources_mock.call_count, 1)
        plan.return_value.audit_budget.assert_called_once_with(15)
        verify.assert_called_once_with(cfg)
        self.assertEqual(cfg['window_resource_audit'], {'new_protocol_only': True})
        self.assertEqual(cfg['validation_protocol_hash'], fingerprint(protocol))
        self.assertEqual(cfg['validation_input_files'], protocol['input_files'])
        self.assertEqual(cfg['data_files'], provenance['files'])

    def test_inventory_binds_validation_meta_and_relative_shared_implementations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'w2v_v31810').mkdir(); (root/'shared').mkdir()
            (root/'w2v_v31810'/'a.py').write_text('from . import b\nfrom shared.ops import go\n', encoding='utf-8')
            (root/'w2v_v31810'/'b.py').write_text('import math\n', encoding='utf-8')
            (root/'shared'/'ops.py').write_text('def go(): pass\n', encoding='utf-8')
            (root/'w2v_v31810'/'test_skip.py').write_text('raise RuntimeError\n', encoding='utf-8')
            self.assertEqual({Path(p).name for p in code_inventory(root)}, {'a.py', 'b.py', 'ops.py'})
        names = {Path(p).name for p in code_inventory()}
        self.assertTrue({'meta.py', 'calibration.py', 'sampling.py', 'identity.py', 'validation.py', 'validation_data.py'} <= names)
        self.assertFalse(any(name.startswith('test_') for name in names))


if __name__ == '__main__': unittest.main()
