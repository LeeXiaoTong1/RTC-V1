"""Real files prove source/content isolation, fixed evaluation and audit gating."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf

from .validation import (GROUPS, SCENES, PANEL_WEIGHTS, build_protocol,
                         split_noise_records, binary_metrics, summarize_protocol,
                         evaluate_protocol)


def _write(path, key, subtype='FLOAT'):
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    wave = np.random.default_rng(seed).normal(0, .03, 4800).astype(np.float32)
    sf.write(path, wave, 16000, subtype=subtype)
    return dict(audio=str(path), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validation_fixture(root):
    root = Path(root); train = []; dev = []; protocol = []; pairs = []
    for language, label in GROUPS:
        for split, count in (('train', 1), ('dev', 5)):
            for n in range(count):
                name = f'{split}_{language}_{label}_{n}.wav'
                off = f'offline/{language}/{name}'; on = f'online/{language}/online_{name}'
                for kind, identity in (('offline', off), ('online', on)):
                    file_info = _write(root/split/identity, split+identity)
                    row = dict(id=identity, source_id=off if split=='train' else identity,
                               group_id=split+identity, condition=kind, domain=kind,
                               language=language, label=label, split=split, full_length=True,
                               **file_info)
                    if split=='train': train.append(row)
                    else:
                        protocol.append(identity+' '+('real' if label else 'fake'))
                        if kind=='online': dev.append(row)
                if split=='dev':
                    pairs.append(('dev/'+off, 'dev/'+on))
                    for kind in ('seen', 'heldout'):
                        dev.append(dict(id=off, source_id=off, group_id=split+off,
                                        audio='not-opened-cached-noisy.wav', condition=kind,
                                        language=language, label=label, split=split, full_length=True))
    dev_protocol = root/'dev.txt'; dev_protocol.write_text('\n'.join(protocol)+'\n', encoding='utf-8')
    pair_path = root/'dev_pairs.csv'
    with pair_path.open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(('offline_id', 'online_id')); writer.writerows(pairs)
    noise = {'train': [], 'dev': []}
    for split in noise:
        for category in SCENES[:-1]:
            for n in range(1 if split=='train' else 4):
                source = f'{split}-{category}-{n}'
                info = _write(root/'noise'/(source+'.wav'), source)
                path = Path(info['audio']); stat = path.stat()
                noise[split].append(dict(path=str(path), sha256=info['audio_sha256'],
                                         size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                         category=category, split=split, source_group=source,
                                         original_sources=[source], license='Synthetic fixture'))
    cfg = dict(seed=718, dev_pairs=str(pair_path), official_dev_protocol=str(dev_protocol),
               official_dev_root=str(root/'dev'), scene_noise_records=noise,
               workers=0, eval_workers=0, eval_batch=5, microbatch=3, frame_budget=400,
               raw_audio_cache_mib=1, noise_cache_mib=1, ffmpeg='unused', rtc_activity_policy='mixed')
    return cfg, train, dev


class ValidationTests(unittest.TestCase):
    def test_fixed_protocol_keeps_offline_online_variants_and_noise_sources_together(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            before = copy.deepcopy(cfg)
            protocol = build_protocol(cfg, train, dev, 'source-run')
            self.assertEqual(cfg, before)
            self.assertEqual(protocol, build_protocol(cfg, train, dev, 'source-run'))
            self.assertFalse(protocol['speech_provenance']['historical_cached_noisy_used'])
            self.assertTrue('not a previously unseen' in protocol['historical_dev_warning'])
            json.dumps(protocol, allow_nan=False)
            select, audit = protocol['select'], protocol['audit']
            self.assertFalse(set(select['speech_sources']) & set(audit['speech_sources']))
            owner = {}
            for split in ('select', 'audit'):
                part = protocol[split]
                self.assertEqual(len(part['rows']), len(part['tickets']))
                for row in part['rows']:
                    key = row['canonical_source']
                    self.assertEqual(owner.setdefault(key, split), split)
                    self.assertIn(row['audio'], protocol['input_files'])
                for panel in PANEL_WEIGHTS:
                    self.assertEqual({(part['rows'][i]['language'], part['rows'][i]['label'])
                                      for i in part['panels'][panel]}, set(GROUPS))
                heldout = [part['tickets'][i]['family'] for i in part['panels']['heldout_noisy']]
                self.assertEqual(set(heldout), {'anlmdn' if split=='select' else 'g711_mulaw'})
            a = {s for r in select['scene_noise_records'] for s in r['original_sources']}
            b = {s for r in audit['scene_noise_records'] for s in r['original_sources']}
            self.assertFalse(a & b)
            self.assertEqual(protocol['noise_provenance']['audit']['office'], 1)

    def test_actual_train_dev_audio_overlap_cannot_hide_under_different_names(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            source = Path(train[0]['audio'])
            target = Path(cfg['official_dev_root'])/'offline/en/dev_en_0_0.wav'
            target.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Train/Dev speech'):
                build_protocol(cfg, train, dev, 'source-run')

    def test_exact_pcm_overlap_is_detected_across_wav_container_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            source = Path(train[0]['audio'])
            wave, rate = sf.read(source, dtype='float32')
            target = Path(cfg['official_dev_root'])/'offline/en/dev_en_0_0.wav'
            sf.write(target, wave, rate, subtype='DOUBLE')
            self.assertNotEqual(source.read_bytes(), target.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Train/Dev speech'):
                build_protocol(cfg, train, dev, 'source-run')

    def test_changed_recorded_hash_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            train[0]['audio_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'Audio hash differs'):
                build_protocol(cfg, train, dev, 'source-run')

    def test_noise_slices_and_compositions_cannot_split_original_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = validation_fixture(directory)
            office = [r for r in cfg['scene_noise_records']['dev'] if r['category']=='office']
            office[1]['original_sources'] = [office[0]['original_sources'][0], 'second-material']
            office[2]['original_sources'] = ['second-material']
            split, _ = split_noise_records(cfg)
            ownership = {r['path']: side for side, rows in split.items() for r in rows}
            self.assertEqual(len({ownership[str(Path(r['path']).resolve())] for r in office[:3]}), 1)

    def test_noise_duplicate_pcm_cannot_hide_in_fake_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = validation_fixture(directory)
            a = cfg['scene_noise_records']['train'][0]
            b = cfg['scene_noise_records']['dev'][0]
            wave, rate = sf.read(a['path'], dtype='float32')
            sf.write(b['path'], wave, rate, subtype='DOUBLE')
            b['sha256'] = hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'Train/Dev noise'):
                split_noise_records(cfg)

    def test_small_bank_cannot_claim_source_isolated_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = validation_fixture(directory)
            for row in cfg['scene_noise_records']['dev']:
                if row['category']=='office': row['original_sources'] = ['one-long-recording']
            with self.assertRaisesRegex(ValueError, 'only 1 independent Dev'):
                split_noise_records(cfg)

    def test_macro_f1_real_false_positive_and_fake_false_negative(self):
        rows = [dict(label=x) for x in (0, 0, 1, 1)]
        result = binary_metrics(rows, [[2, 0], [0, 2], [1, 1], [0, 2]])
        self.assertEqual(result['macro_f1'], .5)
        self.assertEqual(result['real_false_positive'], .5)
        self.assertEqual(result['fake_false_negative'], .5)
        self.assertEqual(result['threshold'], .5)

    def test_select_weighted_scalar_never_uses_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            protocol = build_protocol(cfg, train, dev, 'source-run')
            part = protocol['select']; z = np.zeros((len(part['rows']), 2))
            for i, row in enumerate(part['rows']): z[i, row['label']] = 1
            for i in part['panels']['regular_noisy']: z[i] = z[i][::-1]
            result = summarize_protocol(protocol, z)
            self.assertAlmostEqual(result['score'], .65)
            audit = protocol['audit']; az = np.zeros((len(audit['rows']), 2))
            for i, row in enumerate(audit['rows']): az[i, row['label']] = 1
            self.assertIsNone(summarize_protocol(protocol, az, 'audit')['score'])
            self.assertFalse(summarize_protocol(protocol, az, 'audit')['selection_eligible'])
            self.assertEqual(set(result['groups']), {p+'/'+lang for p in PANEL_WEIGHTS for lang in ('en', 'zh')})

    def test_locked_audit_needs_selected_checkpoint_and_inference_once(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            protocol = build_protocol(cfg, train, dev, 'source-run')
            calls = []
            def infer(model, rows, local, tickets, split):
                calls.append((len(rows), local, split))
                z = np.zeros((len(rows), 2))
                for i, row in enumerate(rows): z[i, row['label']] = 1
                return z, rows
            with self.assertRaisesRegex(ValueError, 'already selected'):
                evaluate_protocol(object(), protocol, cfg, 'audit', infer_fn=infer)
            self.assertFalse(calls)
            result = evaluate_protocol(object(), protocol, cfg, 'audit', infer_fn=infer,
                                       selected_checkpoint='epoch_9_best')
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['selected_checkpoint'], 'epoch_9_best')
            self.assertEqual(calls[0][1]['scene_noise_records']['dev'], protocol['audit']['scene_noise_records'])
            self.assertEqual(result['panels']['heldout_noisy']['macro_f1'], 1)

    def test_fixed_synthesis_retains_family_and_identical_mix_conditions(self):
        from .validation_data import FixedWaves
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            protocol = build_protocol(cfg, train, dev, 'source-run')
            part = protocol['select']; cfg['scene_noise_records'] = {'dev': part['scene_noise_records']}
            captured = []
            def generate(wave, recipe, bank, engine):
                captured.append(recipe)
                return wave, {}
            with patch('w2v_v31810.validation_data.Engines'), patch('w2v_v31810.validation_data.generate', generate):
                dataset = FixedWaves(part['rows'], cfg, 'dev')
                first = part['panels']['regular_noisy'][0]
                second = part['panels']['heldout_noisy'][0]
                a = dataset[part['tickets'][first]][0]
                b = dataset[part['tickets'][second]][0]
            self.assertEqual(captured[0]['family'], 'webrtc')
            self.assertEqual(captured[1]['family'], 'anlmdn')
            self.assertEqual(captured[0]['snr_db'], captured[1]['snr_db'])
            self.assertEqual(captured[0]['seed'], captured[1]['seed'])
            self.assertEqual(captured[1]['codec'], 'none')
            self.assertEqual(captured[1]['activity']['profile'], 'off')
            self.assertEqual(a['canonical_source'], b['canonical_source'])
            self.assertNotEqual(a['validation_panel'], b['validation_panel'])

    def test_actual_fixed_loader_restores_mode_and_never_computes_gradients(self):
        import torch
        from .validation_data import infer_fixed
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1)); self.observed = []
            def forward(self, waves):
                self.observed.append((self.training, torch.is_grad_enabled(), len(waves)))
                values = torch.tensor([float(np.mean(x)) for x in waves])*self.weight
                return torch.stack((values, -values), 1)
        class Engine:
            def __init__(self, *args): pass
            def __call__(self, wave, recipe): return wave.copy()
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            protocol = build_protocol(cfg, train, dev, 'source-run')
            part = protocol['select']; cfg['scene_noise_records'] = {'dev': part['scene_noise_records']}
            model = Detector().train(); before = model.weight.detach().clone(); rng = torch.get_rng_state()
            with patch('w2v_v31810.validation_data.Engines', Engine):
                z, rows = infer_fixed(model, part['rows'], cfg, part['tickets'])
            self.assertEqual(z.shape, (len(part['rows']), 2))
            self.assertEqual([r['row_index'] for r in rows], list(range(len(rows))))
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(model.weight, before)); self.assertIsNone(model.weight.grad)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertTrue(all(not training and not grad for training, grad, _ in model.observed))


if __name__=='__main__': unittest.main()
