"""Gradient correctness, production-loss parity and no-update report integration."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

import audit_w2v_pair_gradients as audit
from w2v_rebuild.core import objective, pair_loss, sha256
from w2v_rebuild.pair_gradient import loss_terms, gradient_gram, comparison_rows, pair_diagnostics, dot
from w2v_rebuild.pair_gradient_data import AuditTrainData, recorded_args, metadata_fingerprints


class GradientMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)

    def test_ce_and_combined_gradient_match_production_with_all_weights(self):
        labels = torch.tensor([0, 1]*12 + [0, 0, 1, 1]*4)
        logits = torch.randn(40, 2, requires_grad=True)
        features = torch.randn(40, 8, requires_grad=True)
        weights = torch.tensor([.63, 2.7])
        language = torch.linspace(.3, 2.5, 40)
        for cost in (1., 1.25):
            terms = loss_terms(logits, features, labels, (24, 4, 4), weights, cost, language)
            expected, _ = objective(logits, features, labels, 24, 4, 4, weights, .1, .05,
                                     real_ce_weight=cost, language_weights=language)
            combined = terms['ce'] + .1*terms['rtc'] + .05*terms['noisy']
            self.assertTrue(torch.equal(combined, expected))
            a = torch.autograd.grad(combined, (logits, features), retain_graph=True)
            b = torch.autograd.grad(expected, (logits, features), retain_graph=True)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_error_probe_preserves_actual_weighted_ce_contribution(self):
        y = torch.tensor([0, 1]*12 + [0, 0, 1, 1]*4)
        z = torch.randn(40, 2, requires_grad=True)
        h = torch.randn(40, 8, requires_grad=True)
        lang = torch.ones(40); lang[38] = 2.3
        terms = loss_terms(z, h, y, (24, 4, 4), torch.ones(2), 1.25, lang,
                           {'noisy_error_ce': [False, False, True, False], 'rtc_error_ce': [False]*4})
        expected = .3 * F.cross_entropy(z[38:39], y[38:39]) * 1.25 * 2.3 / 4.5
        torch.testing.assert_close(terms['noisy_error_ce'], expected)
        self.assertNotIn('rtc_error_ce', terms)

    def test_known_opposing_gradients_scaling_and_unused_classifier(self):
        p = nn.Parameter(torch.tensor([1., 2.]))
        c = nn.Parameter(torch.tensor([3.]))
        terms = {'ce': p.sum()+c.sum(), 'rtc': -2*p.sum(), 'noisy': p.sum()*3,
                 'noisy_processed_ce': p.sum()+c.sum()}
        result = gradient_gram(terms, [('backbone.encoder.layers.0.weight', p), ('head.classifier.bias', c)])
        self.assertEqual(dot(result, 'rtc', 'rtc', 'classifier'), 0.)
        self.assertEqual(dot(result, 'ce', 'rtc', 'encoder'), -4.)
        rows = comparison_rows(result)
        row = next(r for r in rows if r['target'] == 'ce' and r['auxiliary'] == 'rtc' and r['group'] == 'shared')
        self.assertAlmostEqual(row['cosine'], -1.)
        self.assertAlmostEqual(row['norm_ratio'], .2)
        self.assertAlmostEqual(row['opposition_fraction'], .2)
        self.assertIsNone(p.grad); self.assertIsNone(c.grad)
        self.assertEqual(p.tolist(), [1., 2.])

    def test_rounded_zero_loss_does_not_hide_nonzero_gradient(self):
        ref = torch.tensor([[1., .01], [1., .01], [-1., -.01], [-1., -.01]])
        processed = ref.clone(); processed[2, 1] = .01
        row = pair_diagnostics(ref, processed, torch.tensor([0, 0, 1, 1]))
        self.assertEqual(row['loss_fp32'], 0.)
        self.assertGreater(row['loss_fp64_reference'], 0.)
        self.assertGreater(row['feature_gradient_norm'], 0.)
        self.assertEqual(row['opposite_class_negatives'], [2]*4)

    def test_real_hf_checkpointed_backprop_preserves_every_parameter_and_buffer(self):
        from w2v_rebuild.partial_freeze_tests import small_detector
        from w2v_rebuild.model import forward_chunks
        model = small_detector(checkpointing=True)
        model.configure_trainable_layers(1)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        named = list(model.named_parameters())
        # Small real encoder, same five-branch logical layout.
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        for train in (False, True):
            model.train(train)
            z, h = forward_chunks(model, torch.randn(10, 20, 160), torch.ones(10, 20, dtype=torch.long), 2)
            terms = loss_terms(z, h, labels, (2, 2, 2), torch.ones(2),
                               error_masks={'noisy_error_ce': [True, False]})
            result = gradient_gram(terms, named)
            self.assertGreater(dot(result, 'ce', 'ce', 'encoder'), 0.)
            self.assertGreater(dot(result, 'noisy', 'noisy', 'head_shared'), 0.)
            self.assertEqual(dot(result, 'noisy', 'noisy', 'classifier'), 0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, original[k], rtol=0, atol=0)

    def test_random_cohort_not_reclassified_when_errors_are_found(self):
        rows = [dict(step=s, branch=b, processed_correct=(s != 1), correct_to_wrong=(s == 1))
                for s in range(3) for b in ('rtc', 'noisy') for _ in range(4)]
        selected = audit.select_batches(rows, [1], 2, 123)
        self.assertEqual(selected[0]['cohort'], 'random')
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['fixed_error_masks']['noisy_error_ce'], [True]*4)

    def test_reject_unmeasured_extra_objectives(self):
        for field in ('local_structure_weight', 'consistency_weight'):
            with self.assertRaisesRegex(ValueError, 'extra objectives'):
                recorded_args({'stage': 3, field: .02}, 'cpu')


class TinyAuditHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        h = self.dropout(torch.tanh(self.shared(x.mean(1))))
        return self.classifier(h), h


class TinyAuditDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = TinyAuditHead()

    def configure_trainable_layers(self, count):
        self.requires_grad_(True)

    def forward(self, features, mask, validated_mask=False):
        return self.head(self.backbone(features[..., :4]))


class TinyExtractor:
    sampling_rate = 16000


class TinyReadOnlyCollator:
    def __init__(self, *args, **kwargs):
        from audit_w2v_dev import ReadOnlyCollator
        self.inner = ReadOnlyCollator(*args, **kwargs)
        self.inner.extractor = TinyExtractor()
        self.inner.extract = self.extract

    def __call__(self, rows):
        return self.inner(rows)

    def extract(self, waves):
        x = torch.stack(waves)
        signal = F.adaptive_avg_pool1d(x[:, None], 16).transpose(1, 2)
        features = torch.cat((signal, signal.square(), torch.sin(signal*10), torch.cos(signal*10)), -1)
        return features.repeat(1, 1, 40), torch.ones(len(waves), 16, dtype=torch.long)


def training_fixture(root):
    # Reuse actual complete cache format fixture, adding official RTC and noise.
    from test_w2v_structure_audit import fixture
    source, config, baseline = fixture(root)
    protocol = Path(config['train_protocol'])
    originals = protocol.read_text(encoding='utf-8').splitlines()
    extra, pairs = [], []
    for line in originals:
        sid, label = line.split()
        online = sid.replace('/offline/', '/online/')
        wave, sr = sf.read(Path(config['train_data_path'])/sid)
        path = Path(config['train_data_path'])/online; path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wave*.95, sr)
        extra.append(online+' '+label)
        pairs.append(dict(offline=sid, online=online, label=0 if label == 'fake' else 1))
    protocol.write_text('\n'.join(originals+extra)+'\n', encoding='utf-8')
    manifest = root/'pairs.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in pairs), encoding='utf-8')
    noise = root/'noise.wav'
    sf.write(noise, np.random.default_rng(7).normal(0, .03, 70000), 16000)
    noise_manifest = root/'noise.jsonl'
    noise_manifest.write_text(json.dumps({'path': str(noise), 'split': 'train'})+'\n', encoding='utf-8')
    for bank in [root/'old', root/'diverse']:
        path = bank/'config.json'; value = json.loads(path.read_text())
        value['protocol_sha256'] = sha256(protocol)
        value['noise']['manifest_sha256'] = sha256(noise_manifest)
        path.write_text(json.dumps(value), encoding='utf-8')
    config.update(rtc_pairs=str(manifest), train_noise_manifest=str(noise_manifest), stage=3,
                  noise_environment={'RTC_B_NOISE_MANIFEST': str(noise_manifest), 'RTC_B_NOISE_PROB': '0.5'},
                  amp='none', no_checkpointing=False, trainable_encoder_layers=4, algo=0,
                  language_weighting=True, real_ce_weight=1.25, extra_train_noisy_cache=[str(root/'diverse')],
                  noisy_extra_fraction=.2, noisy_mix_warmup_epochs=1., eval_microbatch=4)
    paths = [protocol, manifest, noise_manifest, root/'ssl'/'config.json', root/'ssl'/'preprocessor_config.json']
    paths += [bank/file for bank in [root/'old', root/'diverse'] for file in ('config.json', 'manifest.jsonl')]
    config['data_fingerprints'] = {str(p.resolve()): sha256(p) for p in paths}
    checkpoint = torch.load(baseline, weights_only=True)
    checkpoint.update(model=TinyAuditDetector().state_dict(), data_fingerprints=config['data_fingerprints'])
    torch.save(checkpoint, baseline)
    config['init_sha256'] = sha256(baseline)
    (source/'stage3'/'config.json').write_text(json.dumps(config), encoding='utf-8')
    return source, config, baseline


class ReportIntegrationTests(unittest.TestCase):
    def test_train_only_real_waves_cache_sampler_and_export_no_updates(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, config, baseline = training_fixture(root)
            config['feature_cache'] = str(root/'readonly_feature_cache')
            # Fingerprinted but NOT read/iterated by the Train-only audit.
            config['dev_protocol'] = str(root/'absent_dev.txt')
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            args = argparse.Namespace(device='cpu', seed=1729, microbatch=4, train_repeats=2,
                                      screen_batches=3, random_batches=2, hard_batches=1,
                                      download_dir=str(root/'download'), upload_temp=True)
            model = TinyAuditDetector()
            model.load_state_dict(torch.load(baseline, weights_only=True)['model'])
            expected = copy.deepcopy(model.state_dict())
            out = root/'audit'
            with patch.dict(os.environ, config['noise_environment']), \
                 patch('w2v_rebuild.pair_gradient_data.ReadOnlyCollator', TinyReadOnlyCollator), \
                 patch.object(audit.Detector, 'load', return_value=model), \
                 patch.object(audit, 'upload_report', return_value='https://temp.sh/test/report.zip') as upload, \
                 patch('torch.optim.AdamW', side_effect=AssertionError('No optimizer allowed')), \
                 patch('torch.save', side_effect=AssertionError('No checkpoint writing allowed')):
                self.assertEqual(audit.run(args, source, config, out), 0)
                upload.assert_called_once_with((root/'download'/'audit.zip').resolve())
            self.assertFalse(Path(config['feature_cache']).exists())
            self.assertIn('https://temp.sh/test/report.zip', (out/'temp_download_url.txt').read_text())
            for path, digest in before.items():
                self.assertEqual(sha256(path), digest, path)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            summary = json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['original_best_preserved'])
            self.assertGreater(len(summary['aggregates']), 0)
            self.assertEqual(summary['selected_cohorts']['random'], 2)
            records = [json.loads(line) for line in (out/'gradient_records.jsonl').read_text().splitlines()]
            self.assertEqual({r['mode'] for r in records}, {'eval_fp32', 'train_fp32_1', 'train_fp32_2'})
            for r in records:
                self.assertEqual(len(r['source_ids']), sum([r['layout'][0], 2*r['layout'][1], 2*r['layout'][2]]))
            with zipfile.ZipFile(root/'download'/'audit.zip') as archive:
                self.assertEqual(set(archive.namelist()), set(audit.EXPORT_FILES))
                self.assertFalse(any(n.endswith(('.pt', '.wav', '.npy')) for n in archive.namelist()))

    def test_changed_train_metadata_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            source, config, _ = training_fixture(Path(folder))
            with Path(config['train_protocol']).open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                metadata_fingerprints(config, source)

    def test_sampling_budgets_and_mixture_match_production_data_bundle(self):
        from w2v_rebuild.data import DataBundle
        from audit_w2v_structure import index_bank
        from audit_w2v_train import protocol_rows, Issues
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, config, _ = training_fixture(root)
            def metadata_only_loader(folder, role, protocol, audio_root):
                rows, _ = protocol_rows(protocol, audio_root, Issues())
                return index_bank(folder, role, protocol, audio_root, {r['source_id']: r for r in rows})
            cases = [dict(language_weighting=True, coverage_training=False, ordinary_sampling='legacy'),
                     dict(language_weighting=False, coverage_training=False, ordinary_sampling='balanced'),
                     dict(language_weighting=True, coverage_training=True, ordinary_sampling='legacy')]
            for case in cases:
                with self.subTest(case=case), patch.dict(os.environ, config['noise_environment']), \
                     patch('rtc_noisy_v2.cache.load_v2_cache', side_effect=metadata_only_loader):
                    args = recorded_args(dict(config, **case, num_workers=0), 'cpu')
                    actual = AuditTrainData(args)
                    expected = DataBundle(args)
                    expected.rotation.configure_mixture(args.noisy_extra_fraction,
                                                         round(expected.steps*args.noisy_mix_warmup_epochs))
                    expected.begin(1)
                    self.assertEqual(actual.ordinary_plan, list(expected.train[0].batch_sampler))
                    self.assertEqual(actual.rtc_plan, list(expected.train[1].batch_sampler))
                    self.assertEqual(actual.noisy_plan, list(expected.train[2].batch_sampler))
                    self.assertEqual(actual.language_budgets, expected.language_budgets)
                    self.assertEqual(actual.rotation_summary, expected.rotation.plan_summary())
                    torch.testing.assert_close(actual.weights, expected.weights)


if __name__ == '__main__':
    unittest.main()
