"""Official Offline-only recordings remain valid Noisy sources, never fake Online."""
from collections import Counter
import copy
import csv
from pathlib import Path
import tempfile
import unittest

from .common import atomic_json, read_json
from .identity import (canonical_sources, read_official_pairs, registry_config,
                       verify_registry)
from .sampling import Plan
from .test_validation import _write, validation_fixture
from .validation import GROUPS, PANEL_WEIGHTS, build_protocol


def _pairs(path, records):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('offline_id', 'online_id'))
        writer.writerows(records)


def unpaired_train_fixture(root, *, embedded=False):
    """Twenty real pairs and four retained Offline-only recordings per class."""
    root = Path(root)
    cfg, _, dev = validation_fixture(root)
    train = []; pairs = []; orphan_ids = set()
    for language, label in GROUPS:
        for number in range(24):
            off = f'train-off-{language}-{label}-{number}'
            on = f'train-on-{language}-{label}-{number}' if number < 20 else ''
            pairs.append((off, on if number % 2 else '  '+on+'  '))
            for kind, name in (('offline', off), ('online', on)):
                if not name:
                    orphan_ids.add(off)
                    continue
                row = dict(id=name, source_id=name, group_id=name, condition=kind,
                           language=language, label=label, split='train', full_length=True,
                           **_write(root/'train-actual'/(name+'.wav'), name))
                if embedded and kind == 'offline':
                    row.update(offline_id=off, online_id=on if number % 2 else '  '+on+'  ')
                train.append(row)
    if not embedded:
        path = root/'train_pairs.csv'
        _pairs(path, pairs); cfg['train_pairs'] = str(path)
    cfg.update(version='3.18.10', source_draws_per_stream=[96]*4,
               consistency='none', warm_epochs=1)
    return cfg, train, dev, orphan_ids


def add_many_dev_orphans(cfg, dev):
    """Two surviving Online sources amid twenty Offline sources in each group."""
    root = Path(cfg['official_dev_root']); protocol = []; pairs = []; orphan_ids = set()
    kept_online = set()
    for language, label in GROUPS:
        for number in range(20):
            name = f'dev_{language}_{label}_{number}.wav'
            off = f'offline/{language}/{name}'
            on = f'online/{language}/online_{name}' if number < 2 else ''
            if number >= 5:
                _write(root/off, 'dev'+off)
            protocol.append(off+' '+('real' if label else 'fake'))
            if on:
                protocol.append(on+' '+('real' if label else 'fake'))
                kept_online.add(on)
            else:
                orphan_ids.add(off)
            pairs.append(('dev/'+off, 'dev/'+on if on else ' \t '))
    Path(cfg['official_dev_protocol']).write_text('\n'.join(protocol)+'\n', encoding='utf-8')
    _pairs(cfg['dev_pairs'], pairs)
    dev = [row for row in dev if row['condition'] != 'online' or row['id'] in kept_online]
    return dev, orphan_ids, kept_online


class UnpairedOfflineTests(unittest.TestCase):
    def test_explicit_blank_csv_and_embedded_online_fields_are_records_not_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, _, orphan_ids = unpaired_train_fixture(directory)
            pairs = read_official_pairs(train, cfg['train_pairs'])
            self.assertEqual({off for off, on in pairs if not on}, orphan_ids)
            self.assertEqual(len(pairs), 96)
            canonical, audit = canonical_sources(train, cfg['train_pairs'], require_online_pairs=True)
            self.assertEqual(audit['official_pair_records'], 80)
            self.assertEqual(audit['offline_without_online_records'], 16)
            self.assertEqual(audit['original_recordings'], 96)
            by_id = {row['id']: source for row, source in zip(train, canonical)}
            self.assertEqual(len({by_id[name] for name in orphan_ids}), 16)
            self.assertEqual(read_official_pairs([dict(id='a', offline_id=' off ', online_id=' \t ')]), [('off', '')])
            self.assertEqual(read_official_pairs([dict(id='a', offline_id=' off ', online_id=' on ')]), [('off', 'on')])

    def test_offline_only_train_survives_pcm_registry_sampling_calibration_and_resume(self):
        for embedded in (False, True):
            with self.subTest(embedded=embedded), tempfile.TemporaryDirectory() as directory:
                cfg, original, dev, orphan_ids = unpaired_train_fixture(directory, embedded=embedded)
                protocol, train, registry = build_protocol(cfg, original, dev, 'data-run', return_registry=True)
                cfg['source_registry'] = registry_config(registry)
                self.assertEqual(registry['pairing_audit']['train']['offline_without_online_records'], 16)
                self.assertEqual(registry['official_pair_records']['train'], 80)
                self.assertEqual(len(train), len(original))
                self.assertEqual({row['id'] for row in train}, {row['id'] for row in original})
                verify_registry(train, registry, cfg, protocol)
                plan = Plan(train, cfg=cfg)
                self.assertEqual(plan.identity_audit['offline_only_sources'], 16)
                self.assertEqual(plan.identity_audit['paired_sources'], 80)
                batches = list(plan.batches(0)); calibration = plan.calibration_batches()
                totals = Counter(); seen_orphans = set()
                for batch in batches + calibration:
                    self.assertEqual(len(batch), len({ticket['canonical_source'] for ticket in batch}))
                    for ticket in batch:
                        row = train[ticket['index']]
                        self.assertEqual(ticket['kind'], row['condition'])
                        if row['id'] in orphan_ids:
                            self.assertEqual(ticket['kind'], 'offline')
                            self.assertIsNotNone(ticket['domain_override'])
                for batch in batches:
                    for ticket in batch:
                        row = train[ticket['index']]
                        totals[ticket['kind'], row['language'], row['label']] += 1
                        if row['id'] in orphan_ids: seen_orphans.add(row['id'])
                self.assertEqual(seen_orphans, orphan_ids)
                self.assertEqual(totals, Counter({(kind, *group): 96 for kind in ('online', 'offline') for group in GROUPS}))
                self.assertEqual(len(calibration), 12)
                root = Path(directory)/'saved'; root.mkdir()
                for name, value in (('config', cfg), ('train_rows', train),
                                    ('validation_protocol', protocol), ('source_registry', registry)):
                    atomic_json(root/(name+'.json'), value)
                resumed_cfg = read_json(root/'config.json'); resumed_train = read_json(root/'train_rows.json')
                verify_registry(resumed_train, read_json(root/'source_registry.json'), resumed_cfg,
                                read_json(root/'validation_protocol.json'))
                resumed = Plan(resumed_train, cfg=resumed_cfg)
                self.assertEqual(batches, list(resumed.batches(0)))
                self.assertEqual(calibration, resumed.calibration_batches())

    def test_many_offline_only_dev_sources_keep_both_noisy_panels_and_domain_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev = validation_fixture(directory)
            dev, orphan_ids, kept_online = add_many_dev_orphans(cfg, dev)
            protocol, _, registry = build_protocol(cfg, train, dev, 'data-run', return_registry=True)
            self.assertEqual(protocol, build_protocol(cfg, train, dev, 'data-run'))
            self.assertEqual(registry['pairing_audit']['dev']['offline_without_online_records'], 72)
            self.assertEqual(registry['official_pair_records']['dev'], 8)
            seen = Counter(); actual_online = set(); owner = {}
            for split in ('select', 'audit'):
                part = protocol[split]
                for panel in PANEL_WEIGHTS:
                    rows = [part['rows'][index] for index in part['panels'][panel]]
                    self.assertEqual({(row['language'], row['label']) for row in rows}, set(GROUPS))
                    for row in rows:
                        self.assertEqual(owner.setdefault(row['canonical_source'], split), split)
                        if panel == 'official_online':
                            actual_online.add(row['id']); self.assertNotIn(row['id'], orphan_ids)
                        if row['id'] in orphan_ids:
                            self.assertIn(panel, ('regular_noisy', 'heldout_noisy'))
                            seen[row['id'], panel] += 1
            self.assertEqual(actual_online, kept_online)
            self.assertEqual(seen, Counter({(name, panel): 1 for name in orphan_ids
                                           for panel in ('regular_noisy', 'heldout_noisy')}))

    def test_nonempty_unresolved_endpoint_still_fails_for_train_and_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev, _ = unpaired_train_fixture(directory)
            for split in ('Train', 'Dev'):
                local = copy.deepcopy(cfg)
                original_path = cfg['train_pairs'] if split == 'Train' else cfg['dev_pairs']
                records = read_official_pairs([], original_path)
                off, _ = records[0]
                records[0] = (off, 'nonempty-but-missing-online.wav')
                path = Path(directory)/('bad-'+split+'.csv'); _pairs(path, records)
                local['train_pairs' if split == 'Train' else 'dev_pairs'] = str(path)
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'Official '+split+' pair has a missing member'):
                    build_protocol(local, train, dev, 'data-run')

    def test_missing_columns_and_empty_offline_are_not_silently_treated_as_orphans(self):
        cases = [
            ('offline_id,online_id\noff\n', 'Malformed official pair CSV'),
            ('offline_id,online_id\noff,,unexpected\n', 'Malformed official pair CSV'),
            ('offline_id\noff\n', 'lacks offline_id/online_id'),
            ('offline_id,online_id\n,on\n', 'nonempty Offline ID'),
            ('offline_id,online_id\n,\n', 'nonempty Offline ID'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pairs.csv'
            for content, error in cases:
                path.write_text(content, encoding='utf-8')
                with self.subTest(content=content), self.assertRaisesRegex(ValueError, error):
                    read_official_pairs([], path)

    def test_orphan_status_does_not_relax_actual_train_dev_pcm_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, train, dev, _ = unpaired_train_fixture(directory)
            dev, orphan_ids, _ = add_many_dev_orphans(cfg, dev)
            orphan = Path(cfg['official_dev_root'])/sorted(orphan_ids)[0]
            orphan.write_bytes(Path(train[0]['audio']).read_bytes())
            with self.assertRaisesRegex(ValueError, 'Train/Dev speech source or exact audio content overlaps'):
                build_protocol(cfg, train, dev, 'data-run')


if __name__ == '__main__':
    unittest.main()
