"""Read-only source isolation, true-class metrics and CPU report integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf
import torch
from torch import nn

import audit_w2v_structure as audit
from w2v_rebuild import SCHEMA
from w2v_rebuild.core import sha256
from w2v_rebuild.structure_probe import (metrics, fit_probe, predict_probe, source_weights,
                                        physical_descriptors, source_bootstrap)


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, features, mask, validated_mask=False, return_frames=False):
        assert not self.training and not torch.is_grad_enabled()
        assert not any(p.requires_grad for p in self.parameters())
        frames = features*self.scale
        readout = torch.cat((frames.mean(1), frames.std(1, unbiased=False)), 1)
        logits = torch.stack((readout[:, 0], -readout[:, 0]), 1)
        return (logits, readout, frames) if return_frames else (logits, readout)


class TinyCollator:
    def __init__(self, *args):
        pass

    def __call__(self, rows):
        waves = torch.stack([r[0] for r in rows])
        signal = torch.nn.functional.adaptive_avg_pool1d(waves[:, None], 64).transpose(1, 2)
        time = torch.linspace(0, 2*np.pi, 64)[None, :, None]
        features = torch.cat([torch.sin(time*(j+1)+signal*10)+signal for j in range(8)], -1)
        return dict(features=features, mask=torch.ones(len(rows), 64, dtype=torch.long))


def fixture(root):
    from rtc_noisy.common import CACHE_FORMAT, CUT, SR, SNR_BANDS
    from rtc_noisy_v2.plan import SCHEMA as CACHE_SCHEMA, PLAN_ID, plan_definition, settings_for
    from rtc_noisy.diverse import profile_definition
    source = root/'previous'; (source/'stage3').mkdir(parents=True)
    ssl = root/'ssl'; ssl.mkdir()
    for filename in ('config.json', 'preprocessor_config.json'):
        (ssl/filename).write_text('{}', encoding='utf-8')
    config = dict(stage=3, adaptation=True, ssl_path=str(ssl), feature_cache=None,
                  model_config={'id2label': {0: 'fake', 1: 'real'}}, eval_microbatch=4)
    originals = {}
    for split in ('train', 'dev'):
        audio_root = root/split; audio_root.mkdir()
        protocol = root/(split+'.txt'); lines = []
        originals[split] = []
        for lang_index, language in enumerate(('en', 'zh')):
            for label, name in enumerate(('fake', 'real')):
                for index in range(8):
                    sid = f'{language}/{name}/offline/{index:03}.wav'
                    path = audio_root/sid; path.parent.mkdir(parents=True, exist_ok=True)
                    hz = 100+index*11+lang_index*170+label*80+(7 if split == 'dev' else 0)
                    wave = .15*np.sin(2*np.pi*hz*np.arange(CUT)/SR) + .02*np.sin(np.arange(CUT)*.0003*(index+1))
                    sf.write(path, wave, SR)
                    originals[split].append((sid, label, sha256(path), wave, index))
                    lines.append(sid+' '+name)
        protocol.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        config[split+'_protocol'] = str(protocol); config[split+'_data_path'] = str(audio_root)
    definitions = [('old', 'train', 'train', 0, None), ('diverse', 'train', 'train', 1, 'diverse'),
                   ('seen', 'dev', 'dev_seen', 0, 'diverse'), ('held', 'dev', 'dev_heldout', 0, 'unseen')]
    for bank_name, split, role, generation, profile in definitions:
        bank = root/bank_name; bank.mkdir()
        cfg = dict(format=CACHE_FORMAT, split=split, protocol_sha256=sha256(config[split+'_protocol']),
                   snr_bands=[list(x) for x in SNR_BANDS], cut=CUT, sr=SR, limit=0, offline_count=32,
                   optimization_schema=CACHE_SCHEMA, role=role, plan_id=PLAN_ID, plan=plan_definition(),
                   allowed_settings=[list(x) for x in settings_for(role)], generation=generation,
                   seed=123, ffmpeg_version='fixture', noise={'recording_ids': [split], 'file_sha256': [split]})
        if profile:
            cfg['processing'] = profile_definition(profile)
            cfg['webrtc_version'] = '0.1.3'
        rows = []
        for n, (sid, label, digest, wave, index) in enumerate(originals[split]):
            cached = bank/f'{n}.wav'; sf.write(cached, wave*.95, SR)
            for band in range(4):
                values = settings_for(role)[band]
                row = dict(source=sid, label=label, source_sha256=digest, audio=cached.name,
                           band=band, snr_db=SNR_BANDS[band][0], role=role, generation=generation,
                           mix_id=('a' if split == 'train' else 'b')*64,
                           rtc=dict(zip(('noise_reduction', 'max_gain', 'bitrate'), values)))
                if profile:
                    row['processing'] = {'family': cfg['processing']['families'][index % len(cfg['processing']['families'])]}
                rows.append(row)
        (bank/'config.json').write_text(json.dumps(cfg), encoding='utf-8')
        (bank/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    config.update(train_noisy_cache=str(root/'old'), extra_train_noisy_cache=[str(root/'diverse')],
                  dev_noisy_cache=str(root/'seen'), dev_heldout_cache=str(root/'held'))
    baseline = root/'original'/'stage3'/'best_model.pt'; baseline.parent.mkdir(parents=True)
    config['baseline_path'] = str(baseline)
    paths = audit.metadata_paths(config, source)
    config['data_fingerprints'] = {str(p): sha256(p) for p in paths if p.is_file()}
    torch.save(dict(schema=SCHEMA, stage=3, kind='weights', epoch=20,
                    model=TinyDetector().state_dict(), model_config=config['model_config'],
                    data_fingerprints=config['data_fingerprints']), baseline)
    config['init_sha256'] = sha256(baseline)
    (source/'stage3'/'config.json').write_text(json.dumps(config), encoding='utf-8')
    return source, config, baseline


class StructureAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fake_probability_label_convention_and_auc_ties(self):
        result = metrics([0, 0, 1, 1], [.9, .8, .2, .1])
        self.assertEqual(result['macro_f1'], 1.)
        self.assertEqual(result['auc'], 1.)
        self.assertEqual(metrics([0, 1], [.5, .5])['auc'], .5)
        self.assertEqual(metrics([0, 1], [.5, .5])['recall_real'], 0.)

    def test_probe_fits_train_only_and_does_not_reweight_repeated_sources(self):
        rows = [dict(split='train', source_id=str(i)) for i in range(4)]
        x = np.array([[-2., 0.], [-1., 1.], [1., 0.], [2., 1.]])
        labels = np.array([0, 0, 1, 1])
        probe = fit_probe(x, labels, rows)
        repeated = rows + [rows[0]]*5
        other = fit_probe(np.concatenate((x, x[:1].repeat(5, 0))), np.append(labels, [0]*5), repeated)
        np.testing.assert_allclose(predict_probe(probe, x), predict_probe(other, x), atol=1e-6)
        self.assertAlmostEqual(sum(source_weights(repeated)[[0, 4, 5, 6, 7, 8]]), .25)
        with self.assertRaisesRegex(ValueError, 'Only official Train'):
            fit_probe(x, labels, [dict(r, split='dev') for r in rows])

    def test_spectral_dynamic_gain_cancellation_and_finite_silence(self):
        generator = torch.Generator().manual_seed(19)
        wave = torch.randn(2, 64600, generator=generator)*.05
        dynamic, pooled = physical_descriptors(wave)
        gained, gained_pool = physical_descriptors(wave*.3)
        self.assertEqual(dynamic.shape, (2, 128))
        np.testing.assert_allclose(dynamic, gained, atol=2e-5)
        self.assertGreater(np.abs(pooled-gained_pool).mean(), .5)
        silent, _ = physical_descriptors(torch.zeros_like(wave))
        self.assertTrue(np.isfinite(silent).all())

    def test_sampling_content_dedup_and_train_exclusion(self):
        proto, rows = {}, []
        for language in ('en', 'zh'):
            for label in (0, 1):
                for i in range(10):
                    sid = f'{language}/{label}/offline/{i}.wav'
                    proto[sid] = dict(domain='offline', language_group=language, label=label)
                    rows.append(dict(source=sid, source_sha256=f'{language}-{label}-{i}'))
        excluded = {rows[0]['source_sha256']}
        selected, hashes = audit.select_sources(proto, [(rows, {})], 8, 1, excluded)
        self.assertEqual(len(selected), 32)
        self.assertFalse({hashes[s] for s in selected} & excluded)
        self.assertEqual(selected, audit.select_sources(proto, [(rows, {})], 8, 1, excluded)[0])
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            audit.select_sources(proto, [(rows, {})], 10, 1, excluded)

    def test_cpu_end_to_end_real_cache_metadata_frozen_model_and_safe_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, config, baseline = fixture(root)
            out = root/'audit'; download = root/'download'; before = sha256(baseline)
            with patch.object(audit.Detector, 'load', return_value=TinyDetector()), patch.object(audit, 'ReadOnlyCollator', TinyCollator):
                result = audit.main(['--from-run', str(source), '--out', str(out), '--download-dir', str(download),
                                     '--device', 'cpu', '--train-per-group', '8', '--dev-per-group', '8', '--bootstrap', '50'])
            self.assertEqual(result, 0)
            self.assertEqual(before, sha256(baseline))
            summary = audit.read_json(out/'summary.json')
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['counts']['sources'], 64)
            self.assertEqual(summary['counts']['views'], 576)
            self.assertEqual(len(summary['alignment_groups']), 8)
            self.assertFalse(summary['empty_coverage_cells'])
            self.assertTrue(summary['original_best_preserved'])
            manifest = audit.read_json(out/'manifest.json')
            self.assertIn(str((source/'stage3'/'config.json').resolve()), manifest['input_sha256'])
            self.assertEqual(manifest['training_recipe_contract']['settings']['local_structure_weight'], .02)
            with zipfile.ZipFile(download/'audit.zip') as archive:
                self.assertEqual(set(archive.namelist()), set(audit.EXPORT_FILES))
                self.assertFalse(any(name.endswith(('.pt', '.wav', '.npy', '.npz')) for name in archive.namelist()))
            # Unknown future files must never slip into an upload.
            (out/'secret_weights.pt').write_bytes(b'not-exported')
            with zipfile.ZipFile(audit.package_report(out, root/'second_download')) as archive:
                self.assertNotIn('secret_weights.pt', archive.namelist())
            # Controlled positive fixture exercises the real reviewer validator;
            # this synthetic fixture is not claiming positive scientific evidence.
            summary['recommendation'] = 'ready_for_review'
            audit.atomic_json(summary, out/'summary.json')
            completed = audit.read_json(out/'completed.json')
            completed['summary_sha256'] = sha256(out/'summary.json')
            audit.atomic_json(completed, out/'completed.json')
            self.assertEqual(audit.validate_reviewed_audit(out, before)['baseline_sha256'], before)
            selected_audio = Path(next(iter(manifest['selected_audio_sha256'])))
            original_audio = selected_audio.read_bytes()
            selected_audio.write_bytes(original_audio+b'changed')
            with self.assertRaisesRegex(ValueError, 'Audited metadata changed'):
                audit.validate_reviewed_audit(out, before)
            selected_audio.write_bytes(original_audio)
            # Validator must reject tampered diagnostic evidence.
            (out/'summary.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'incomplete, changed'):
                audit.validate_reviewed_audit(out, before)

    def test_duplicate_cache_band_rejected_without_decoding_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source, config, baseline = fixture(root)
            issues = audit.Issues()
            rows, _ = audit.protocol_rows(config['train_protocol'], config['train_data_path'], issues)
            protocol = {r['source_id']: r for r in rows}
            manifest = root/'old'/'manifest.jsonl'
            first = manifest.read_text(encoding='utf-8').splitlines()[0]
            with manifest.open('a', encoding='utf-8') as stream:
                stream.write(first+'\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                audit.index_bank(root/'old', 'train', config['train_protocol'], config['train_data_path'], protocol)

    def test_bootstrap_repeating_cached_views_does_not_add_independent_sources(self):
        rows = [dict(split='dev', condition='seen', band=0, language=language, label=label,
                     source_id=f'{language}/{label}/{index}')
                for language in ('en', 'zh') for label in (0, 1) for index in range(3)]
        ideal = np.array([.8 if row['label'] == 0 else .2 for row in rows])
        control = ideal.copy(); control[0] = .1; control[4] = .9
        scores = {'readout+structure': ideal, 'readout+pooled': control,
                  'structure': ideal, 'pooled': control,
                  'readout+spectral_dynamic': ideal, 'readout+spectral_pool': control}
        first = source_bootstrap(rows, scores, repetitions=50)
        repeated = source_bootstrap(rows*4, {k: np.tile(v, 4) for k, v in scores.items()}, repetitions=50)
        for a, b in zip(first, repeated):
            np.testing.assert_allclose(a['ci95'], b['ci95'], atol=1e-15)
            self.assertAlmostEqual(a['delta_macro_f1'], b['delta_macro_f1'])


if __name__ == '__main__':
    unittest.main()
