"""Real decoded PCM survives startup, Train planning, calibration and resume."""
import copy
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from w2v_v3186.test_sampling import rows_for
from .common import atomic_json, read_json
from .identity import (REGISTRY_SCHEMA, registry_config, verify_registry,
                       registered_sources, require_registry_config)
from .sampling import Plan
from .test_validation import validation_fixture
from .validation import build_protocol


def registered_fixture(root, *, embedded=False):
    cfg, _, dev = validation_fixture(root)
    train = rows_for((23, 23, 23, 23)); pairs = []
    for i, row in enumerate(train):
        # Old/stale identity labels must not override freshly decoded content.
        row['canonical_source'] = 'stale-untrusted-component'
        path = Path(root)/'actual-train'/(row['id']+'.wav')
        path.parent.mkdir(parents=True, exist_ok=True)
        wave = np.random.default_rng(9000+i).integers(-3000, 3000, 1200, dtype=np.int16)
        sf.write(path, wave, 16000, subtype='PCM_16')
        row.update(audio=str(path), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if row['condition'] == 'online':
            pairs.append((train[i-1]['id'], row['id']))
            # Explicit pairing is essential: Online metadata alone no longer
            # shares the Offline source, group or original-file hash.
            row.update(source_id=row['id'], group_id=row['id'], source_sha256='on-'+row['id'])
    # WAV/WAVEX have identical PCM16 samples, but different file byte hashes.
    wave, rate = sf.read(train[0]['audio'], dtype='int16')
    sf.write(train[2]['audio'], wave, rate, format='WAVEX', subtype='PCM_16')
    train[2]['audio_sha256'] = hashlib.sha256(Path(train[2]['audio']).read_bytes()).hexdigest()
    # PCM A--B, official pair B--Bon, metadata Bon--C, official pair C--Con.
    train[3]['group_id'] = train[4]['group_id']
    if embedded:
        for i, (off, on) in enumerate(pairs):
            train[2*i].update(offline_id=off, online_id=on)
    else:
        path = Path(root)/'train_pairs.csv'
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(('offline_id', 'online_id')); writer.writerows(pairs)
        cfg['train_pairs'] = str(path)
    protocol, prepared, registry = build_protocol(cfg, train, dev, 'source-run', return_registry=True)
    cfg.update(version='3.18.10', source_registry=registry_config(registry),
               source_draws_per_stream=[113, 89, 89, 89], consistency='none')
    return cfg, train, prepared, protocol, registry


class SourceRegistryTests(unittest.TestCase):
    def test_actual_pcm_pair_transitivity_reaches_sampler_calibration_and_saved_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, original, train, protocol, registry = registered_fixture(directory)
            self.assertNotIn('decoded_pcm_sha256', original[0])
            self.assertNotEqual(train[0]['audio_sha256'], train[2]['audio_sha256'])
            self.assertEqual(train[0]['decoded_pcm_sha256'], train[2]['decoded_pcm_sha256'])
            self.assertEqual(len({r['canonical_source'] for r in train[:6]}), 1)
            self.assertEqual(registry['schema'], REGISTRY_SCHEMA)
            self.assertEqual(registry['official_pair_records']['train'], 92)
            self.assertIn(str(Path(cfg['train_pairs']).resolve()), protocol['input_files'])
            verify_registry(train, registry, cfg, protocol)
            plan = Plan(train, cfg=cfg)
            batches = list(plan.batches(0))
            calibration = plan.calibration_batches()
            for batch in batches + calibration:
                self.assertEqual(len(batch), len({t['canonical_source'] for t in batch}))
                self.assertEqual(len(batch), len({train[t['index']]['decoded_pcm_sha256'] for t in batch}))
                self.assertLessEqual(sum(t['index'] < 6 for t in batch), 1)
            # Use precisely the persisted files loaded by the production resume
            # path; no audio redecoding or independent source union on resume.
            root = Path(directory)/'run'; root.mkdir()
            for name, value in (('config', cfg), ('train_rows', train),
                                ('validation_protocol', protocol), ('source_registry', registry)):
                atomic_json(root/(name+'.json'), value)
            resumed_cfg = read_json(root/'config.json'); resumed_train = read_json(root/'train_rows.json')
            verify_registry(resumed_train, read_json(root/'source_registry.json'),
                            resumed_cfg, read_json(root/'validation_protocol.json'))
            resumed = Plan(resumed_train, cfg=resumed_cfg)
            self.assertEqual(batches, list(resumed.batches(0)))
            self.assertEqual(calibration, resumed.calibration_batches())

    def test_embedded_official_pairs_use_the_same_pcm_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, train, protocol, registry = registered_fixture(directory, embedded=True)
            verify_registry(train, registry, cfg, protocol)
            self.assertEqual(len({r['canonical_source'] for r in train[:6]}), 1)
            self.assertEqual(registry['official_pair_records']['train'], 92)
            self.assertEqual(Plan(train, cfg=cfg).canonical, [r['canonical_source'] for r in train])

    def test_missing_old_registry_and_changed_pcm_identity_or_order_fail(self):
        with self.assertRaisesRegex(ValueError, 'NEW run.*--data-run'):
            require_registry_config(dict(version='3.18.10'))
        with self.assertRaisesRegex(ValueError, 'cannot be resumed'):
            Plan(rows_for((23, 23, 23, 23)), cfg=dict(version='3.18.10'))
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, train, protocol, registry = registered_fixture(directory)
            for field in ('canonical_source', 'decoded_pcm_sha256', 'audio_sha256'):
                changed = copy.deepcopy(train); changed[0][field] = 'e'*64
                with self.assertRaisesRegex(ValueError, 'identity/order'):
                    registered_sources(changed, cfg)
            with self.assertRaisesRegex(ValueError, 'identity/order'):
                registered_sources(list(reversed(train)), cfg)
            changed = copy.deepcopy(registry); changed['train'][0]['canonical_source'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'registry hash changed'):
                verify_registry(train, changed, cfg, protocol)
            with self.assertRaisesRegex(ValueError, 'different PCM source registries'):
                verify_registry(train, registry, cfg, dict(protocol, source_registry_hash='0'*64))


if __name__ == '__main__':
    unittest.main()
