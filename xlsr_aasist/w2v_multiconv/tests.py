"""CPU verification; no real datasets, network, checkpoints or GPU are required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from torch import nn
from .data import (AudioDataset, EpochPlan, FeatureCollator, cache_index, check_processing,
                   read_protocol, safe_audio_path)
from .evaluate import package_scores
from .model import BlockStatisticsPool, Detector, HeadConfig, MultiConvHead, diversity_cka
from .runtime import (Metrics, atomic_json, atomic_save, load_checkpoint,
                      loss_function, replay_step, seed_all, sha256)

torch.set_num_threads(1)


def small_head():
    return HeadConfig(input_dim=16, projection=8, expansion=32,
                      kernels=(3, 5, 7, 9), merge_kernel=5, dropout=.15)


def tiny_detector():
    from transformers import Wav2Vec2BertConfig
    cfg = Wav2Vec2BertConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=32, feature_projection_input_dim=160,
                           conv_depthwise_kernel_size=7, hidden_dropout=.1,
                           attention_dropout=.1, activation_dropout=.1,
                           feat_proj_dropout=0., layerdrop=0., apply_spec_augment=False,
                           num_conv_pos_embedding_groups=2)
    from dataclasses import asdict
    return Detector.from_config(cfg.to_dict(), asdict(small_head()))


class ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(5, 16) for _ in range(3)])
        self.dropout = nn.Dropout(.25)
        self.head = MultiConvHead(small_head())

    def forward(self, features, mask):
        return self.head([self.dropout(layer(features)) for layer in self.layers], mask)


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_all(39)

    def test_cka_is_differentiable_and_identical_is_one(self):
        x = torch.randn(8, 4, 6, requires_grad=True)
        loss = diversity_cka(x)
        loss.backward()
        self.assertGreater(x.grad.abs().sum().item(), .001)
        repeated = torch.randn(8, 1, 6).expand(-1, 4, -1)
        self.assertAlmostEqual(diversity_cka(repeated).item(), 1., places=6)
        zero = torch.zeros(8, 4, 6, requires_grad=True)
        diversity_cka(zero).backward()
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_pool_preserves_time_and_block_axes(self):
        pool = BlockStatisticsPool(4, 2)
        pool.attention.weight.data.zero_()
        frames = [torch.arange(6).float().reshape(1, 3, 2) + 10 * i for i in range(4)]
        result, means = pool(frames, torch.ones(1, 3, dtype=torch.long))
        expected = torch.stack([x.mean(1) for x in frames], 1)
        torch.testing.assert_close(means, expected)
        torch.testing.assert_close(result[:, :8], expected.flatten(1))

    def test_padding_does_not_change_head_output(self):
        model = MultiConvHead(small_head()).eval()
        hidden = [torch.randn(1, 13, 16) for _ in range(3)]
        original = model(hidden, torch.ones(1, 13, dtype=torch.long))
        padded = [torch.cat([x, torch.randn(1, 9, 16) * 50], 1) for x in hidden]
        mask = torch.cat([torch.ones(1, 13), torch.zeros(1, 9)], 1).long()
        actual = model(padded, mask)
        for a, b in zip(original, actual):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_replay_matches_full_graph_with_dropout_and_cka(self):
        model = ToyDetector().train()
        replay = copy.deepcopy(model)
        examples = [dict(features=torch.randn(1, 11 + i, 5), mask=torch.ones(1, 11 + i).long(),
                         label=i % 2, noisy=i >= 4) for i in range(6)]
        weights = torch.tensor([.7, 1.5])
        direct_opt = torch.optim.SGD(model.parameters(), lr=.04)
        replay_opt = torch.optim.SGD(replay.parameters(), lr=.04)
        seed_all(333)
        zs, hs = zip(*(model(e['features'], e['mask']) for e in examples))
        loss, _ = loss_function(torch.cat(zs), torch.cat(hs), torch.tensor([e['label'] for e in examples]),
                                torch.tensor([e['noisy'] for e in examples]), weights, .3, .2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        direct_opt.step()
        expected_rng = torch.get_rng_state()
        seed_all(333)
        stats, _ = replay_step(replay, examples, replay_opt, weights, torch.device('cpu'),
                               amp='none', cka_weight=.2, check_replay=True)
        self.assertAlmostEqual(stats['loss'], loss.item(), places=5)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng)
        for a, b in zip(model.parameters(), replay.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_weight_does_not_cancel_in_microbatch(self):
        logits = torch.zeros(4, 2, requires_grad=True)
        blocks = torch.randn(4, 4, 8, requires_grad=True)
        loss, _ = loss_function(logits, blocks, torch.tensor([0, 0, 1, 1]),
                                torch.zeros(4, dtype=torch.bool), torch.tensor([.5, 2.]), cka_weight=0.)
        grad, = torch.autograd.grad(loss, logits)
        self.assertAlmostEqual(abs(grad[2, 1] / grad[0, 0]).item(), 4.)

    def test_hf_frontend_checkpointing_replay_and_freezing(self):
        model = tiny_detector()
        model.configure_trainable_layers(1)
        self.assertFalse(model.backbone.encoder.layers[0].training)
        self.assertTrue(model.backbone.encoder.layers[-1].training)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.feature_projection.parameters()))
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        examples = [dict(features=torch.randn(1, 13 + i, 160), mask=torch.ones(1, 13 + i).long(),
                         label=i % 2, noisy=False) for i in range(4)]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        stats, _ = replay_step(model, examples, opt, torch.ones(2), torch.device('cpu'),
                               amp='none', check_replay=True)
        self.assertTrue(np.isfinite(stats['grad_norm']))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.backbone.encoder.layers[-1].parameters()))
        for n, p in model.named_parameters():
            if n in frozen:
                torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)


class DataTests(unittest.TestCase):
    def test_real_cache_metadata_complete_and_incomplete(self):
        from rtc_noisy_v2.plan import SCHEMA, PLAN_ID, BANDS, plan_definition, settings_for
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records = []
            for i in range(2):
                audio = root / f'{i}.wav'
                audio.touch()
                records.append(dict(id=f'offline/en/{i}.wav', audio=str(audio), label=i, domain='offline'))
            proto = root / 'train.txt'
            proto.write_text('test protocol', encoding='utf-8')
            cfg = dict(format='rtc_noisy_pair_cache_v1', role='train', split='train',
                       protocol_sha256=sha256(proto), limit=0, sr=16000, cut=64600,
                       snr_bands=[list(b) for b in BANDS], offline_count=2,
                       optimization_schema=SCHEMA, plan_id=PLAN_ID, plan=plan_definition(),
                       allowed_settings=[list(x) for x in settings_for('train')], generation=0)
            atomic_json(root / 'config.json', cfg)
            rows = [dict(source=r['id'], label=r['label'], band=b, audio=f'{r["label"]}.wav',
                         snr_db=BANDS[b][0] + 1, role='train', generation=0, mix_id='f' * 64,
                         rtc=dict(zip(('noise_reduction', 'max_gain', 'bitrate'), settings_for('train')[0])))
                    for r in records for b in range(4)]
            manifest = root / 'manifest.jsonl'
            manifest.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            loaded, _ = cache_index(root, 'train', proto, records)
            self.assertEqual(len(loaded), 8)
            manifest.write_text('\n'.join(json.dumps(r) for r in rows[:-1]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                cache_index(root, 'train', proto, records)

    def test_rawboost_and_audio_crop_are_epoch_reproducible(self):
        import soundfile as sf
        from .launch import RAW
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'sample.wav'
            sf.write(path, np.random.default_rng(4).normal(0, .1, 16000).astype(np.float32), 16000)
            rows = [dict(id='sample.wav', audio=str(path), label=0, noisy=False)]
            a = AudioDataset(rows, training=True, epoch=1, rawboost=5, raw_config=RAW, max_seconds=.5)
            b = AudioDataset(rows, training=True, epoch=2, rawboost=5, raw_config=RAW, max_seconds=.5)
            np.testing.assert_array_equal(a[0]['wave'], a[0]['wave'])
            self.assertFalse(np.array_equal(a[0]['wave'], b[0]['wave']))
            self.assertEqual(len(a[0]['wave']), 8000)

    def test_epoch_plan_full_coverage_balanced_noisy_and_rotation(self):
        ordinary = [dict(id=str(i), label=i % 2, noisy=False) for i in range(19)]
        bank = [dict(id=str(i), label=i % 2, band=b, noisy=True) for i in range(4) for b in range(4)]
        plan = EpochPlan(ordinary, [bank], ordinary_batch=8, noisy_batch=4)
        self.assertEqual(plan.steps, 2)
        views = {str(i): set() for i in range(4)}
        for epoch in range(1, 5):
            batches = plan.batches(epoch)
            self.assertEqual(batches, plan.batches(epoch))
            self.assertEqual(sorted(i for b in batches for i in b if i < 19), list(range(19)))
            for batch in batches:
                processed = [plan.records[i] for i in batch if i >= 19]
                self.assertEqual([r['label'] for r in processed], [0, 0, 1, 1])
                for r in processed:
                    views[r['id']].add(r['band'])
        self.assertTrue(all(x == set(range(4)) for x in views.values()))

    def test_protocol_order_labels_and_path_guards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'a.wav').touch()
            (root / 'b.wav').touch()
            proto = root / 'labels.txt'
            proto.write_text('b.wav real\na.wav fake\n', encoding='utf-8')
            rows = read_protocol(proto, root)
            self.assertEqual([r['id'] for r in rows], ['b.wav', 'a.wav'])
            self.assertEqual([r['label'] for r in rows], [1, 0])
            proto.write_text('a.wav fake\na.wav real\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_protocol(proto, root)
            with self.assertRaises(ValueError):
                safe_audio_path(root, '../other.wav')
            with self.assertRaises(ValueError):
                safe_audio_path(root, '/other.wav')

    def test_unseen_processing_cannot_enter_train(self):
        with self.assertRaises(ValueError):
            check_processing({'processing': {'profile': 'unseen'}}, [], 'train')
        with self.assertRaises(ValueError):
            check_processing({}, [{'processing': {'family': 'anlmdn'}}], 'train')

    def test_official_feature_extractor_variable_duration(self):
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as td:
            SeamlessM4TFeatureExtractor().save_pretrained(td)
            collate = FeatureCollator(td)
            rows = [dict(wave=np.random.randn(n).astype(np.float32) * .01, label=i, noisy=False)
                    for i, n in enumerate([6400, 12800])]
            separate = [collate([r])[0] for r in rows]
            together = collate(rows)
            self.assertGreater(together[1]['features'].shape[1], together[0]['features'].shape[1])
            for a, b in zip(separate, together):
                torch.testing.assert_close(a['features'], b['features'])
                self.assertEqual(a['features'].shape[-1], 160)


class ArtifactTests(unittest.TestCase):
    def test_initializer_imports_only_backbone_and_preserves_original(self):
        from .train import initialize
        from transformers import SeamlessM4TFeatureExtractor
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            SeamlessM4TFeatureExtractor().save_pretrained(root)
            encoder = tiny_detector().backbone.state_dict()
            state = {'schema': 'rtc_w2v_rebuild_v1',
                     'model_config': dict(hidden_size=1024, num_hidden_layers=24, feature_projection_input_dim=160),
                     'model': {**{'backbone.' + k: v for k, v in encoder.items()}, 'head.wrong_architecture': torch.ones(7)},
                     'data_fingerprints': {str(root / 'preprocessor_config.json'): sha256(root / 'preprocessor_config.json')}}
            checkpoint = root / 'original.pt'
            atomic_save(checkpoint, state)
            digest = sha256(checkpoint)
            fresh = tiny_detector()
            untouched_head = copy.deepcopy(fresh.head.state_dict())
            with patch('w2v_multiconv.train.Detector.from_config', return_value=fresh):
                actual, _ = initialize(dict(baseline=str(checkpoint), ssl_path=str(root)))
            for key, value in encoder.items():
                torch.testing.assert_close(actual.backbone.state_dict()[key], value, rtol=0, atol=0)
            for key, value in untouched_head.items():
                torch.testing.assert_close(actual.head.state_dict()[key], value, rtol=0, atol=0)
            self.assertEqual(sha256(checkpoint), digest)

    def test_atomic_failed_save_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'last.pt'
            atomic_save(path, {'x': torch.arange(3)})
            digest = sha256(path)
            def fail(_, tmp):
                Path(tmp).write_bytes(b'partial')
                raise OSError('simulated full disk')
            with patch('w2v_multiconv.runtime.torch.save', side_effect=fail):
                with self.assertRaises(OSError):
                    atomic_save(path, {'x': torch.arange(9)})
            self.assertEqual(sha256(path), digest)
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_submission_exact_format_and_probability(self):
        with tempfile.TemporaryDirectory() as td:
            archive = package_scores(['en/b.wav', 'zh/a.wav'], [.75, .125], td)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(), ['scores.txt'])
                self.assertEqual(z.read('scores.txt').decode(), 'en/b.wav 0.7500000000\nzh/a.wav 0.1250000000\n')
            with self.assertRaises(ValueError):
                package_scores(['a'], [float('nan')], td)
            with self.assertRaises(ValueError):
                package_scores(['a', 'a'], [.1, .2], td)

    def test_metrics_fake_probability_direction(self):
        m = Metrics()
        m.update(torch.tensor([[4., -4.], [-4., 4.], [0., 0.]]), [0, 1, 1])
        self.assertEqual(m.result()['confusion'], [[1, 0], [1, 1]])


class IntegrationTests(unittest.TestCase):
    def test_training_resume_and_submission_with_actual_tiny_w2vbert(self):
        """Real WAV/fbank/HF/MultiConv/Adam/checkpoint/eval; only dataset discovery is replaced."""
        import soundfile as sf
        from transformers import SeamlessM4TFeatureExtractor
        from . import train as training
        from . import evaluate
        from .launch import RAW
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ssl = root / 'ssl'
            SeamlessM4TFeatureExtractor().save_pretrained(ssl)
            baseline = root / 'protected.pt'
            baseline.write_bytes(b'protected-original-checkpoint')
            records, noisy = [], []
            for i in range(8):
                wav = np.random.default_rng(i).normal(0, .05, 64600).astype(np.float32)
                audio = root / f'{i}.wav'
                sf.write(audio, wav, 16000)
                row = dict(id=f'{i}.wav', audio=str(audio), label=i % 2, language='en',
                           domain='online', noisy=False, band=-1)
                records.append(row)
                for b in range(4):
                    noisy.append({**row, 'noisy': True, 'band': b})
            plan = EpochPlan(records, [noisy], ordinary_batch=4, noisy_batch=2)
            validation = {'clean': records[:4], 'seen': noisy[:8], 'heldout': noisy[8:16]}
            fingerprint = {str(ssl / 'preprocessor_config.json'): sha256(ssl / 'preprocessor_config.json')}
            cfg = dict(device='cpu', amp='none', seed=71, baseline=str(baseline), baseline_sha256=sha256(baseline),
                       ordinary_batch=4, noisy_batch=2, warmup_epochs=1, joint_epochs=1,
                       trainable_layers=1, head_lr=.001, warmup_head_lr=.002, encoder_lr=.0001,
                       weight_decay=.0001, workers=0, ssl_path=str(ssl), max_seconds=.25,
                       input_policy='test fixed 0.25 s', rawboost=0, raw_config=RAW,
                       lr_warmup_steps=1, cka_weight=.02, noisy_weight=.3, grad_clip=1., patience=3)
            first, resumed = root / 'first', root / 'resumed'
            first.mkdir()
            resumed.mkdir()
            discovery = lambda _: (plan, validation, torch.ones(2), torch.tensor([4, 4]), fingerprint)
            with patch.object(training, 'build_data', discovery), \
                 patch.object(training, 'initialize', lambda _: (tiny_detector(), {})), \
                 patch.object(training, 'source_fingerprints', lambda: {'test': 'fixed'}):
                training.train(cfg, first)
                real_validate = training.validate
                def interrupted(model, val, config, device, score_path):
                    if score_path.name.startswith('epoch_2'):
                        raise RuntimeError('simulated interruption after first saved epoch')
                    return real_validate(model, val, config, device, score_path)
                with patch.object(training, 'validate', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        training.train(cfg, resumed)
                training.train(cfg, resumed, resumed / 'last.pt')
            a, b = load_checkpoint(first / 'last.pt'), load_checkpoint(resumed / 'last.pt')
            self.assertEqual(a['epoch'], 2)
            self.assertEqual(a['global_step'], b['global_step'])
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
            for state_id, state in a['optimizer']['state'].items():
                for k, value in state.items():
                    torch.testing.assert_close(value, b['optimizer']['state'][state_id][k], rtol=0, atol=0)
            self.assertEqual(sha256(baseline), cfg['baseline_sha256'])
            protocol = root / 'eval.txt'
            protocol.write_text('1.wav\n0.wav\n', encoding='utf-8')
            args = ['evaluate', '--checkpoint', str(first / 'best_model.pt'), '--protocol', str(protocol),
                    '--audio-root', str(root), '--out', str(root / 'submission'), '--device', 'cpu', '--workers', '0']
            with patch('sys.argv', args):
                evaluate.main()
            with zipfile.ZipFile(root / 'submission' / 'submission.zip') as z:
                self.assertEqual([x.split()[0] for x in z.read('scores.txt').decode().splitlines()], ['1.wav', '0.wav'])


if __name__ == '__main__':
    unittest.main()
