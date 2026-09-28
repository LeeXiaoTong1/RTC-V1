"""Read-only inference, resumed score identity and artifact integrity regressions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch import nn
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig

from w2v_rebuild import SCHEMA
import audit_w2v_dev as audit
from w2v_rebuild.core import sha256
from w2v_rebuild.data import FeatureCollator


class TinyDetector(nn.Module):
    def __init__(self, bias=0.):
        super().__init__()
        self.proj = nn.Linear(1, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[1.], [0.]]))
            self.proj.bias.copy_(torch.tensor([bias, 0.]))

    def forward(self, features, mask, validated_mask=False):
        assert not self.training and not torch.is_grad_enabled()
        assert not any(p.requires_grad for p in self.parameters())
        z = self.proj(features[:, 0, :1])
        return z, z


def batches(rows):
    result = []
    for start in range(0, len(rows), 5):
        selected = rows[start:start+5]
        features = torch.zeros(len(selected), 12, 160)
        for i, r in enumerate(selected):
            number = int(Path(r['source_id']).stem)
            features[i, :, 0] = [.4, -.1, .04, -.4][number] + max(0, r['band'])*.02
        batch = dict(features=features, mask=torch.ones(len(selected), 12, dtype=torch.long),
                     labels=torch.tensor([r['label'] for r in selected]))
        if selected[0]['band'] == -1:
            batch['ids'] = [r['source_id'] for r in selected]
        else:
            batch['bands'] = [r['band'] for r in selected]
        result.append(batch)
    return result


def score_rows(rows, bias):
    result = []
    for batch in batches(rows):
        z = torch.stack((batch['features'][:, 0, 0]+bias, torch.zeros(len(batch['labels']))), 1)
        for values, p in zip(z.tolist(), z.softmax(1)[:, 0].tolist()):
            result.append(dict(logit_fake=values[0], logit_real=values[1], pfake=p, margin=values[0]-values[1]))
    return result


class DevAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, root):
        stage = root/'adapt'/'stage3'
        stage.mkdir(parents=True)
        model_dir = root/'ssl'
        model_dir.mkdir()
        paths = [root/'dev.txt', model_dir/'preprocessor_config.json', model_dir/'config.json']
        for bank in ('seen', 'held'):
            (root/bank).mkdir()
            paths.extend([root/bank/'config.json', root/bank/'manifest.jsonl'])
        for path in paths:
            path.write_text('{}', encoding='utf-8')
        fingerprints = {str(p.resolve()): sha256(p) for p in paths}
        streams = []
        for condition in ('clean', 'seen', 'heldout'):
            records = []
            for domain in (('online', 'offline') if condition == 'clean' else ('offline',)):
                for i in range(4):
                    for band in ([-1] if condition == 'clean' else range(4)):
                        path = root/'audio'/condition/domain/f'{i}_{band}.wav'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'fixture-input-never-written-by-audit')
                        records.append(audit.base_record(domain if condition == 'clean' else condition,
                            f'{domain}/{i}.wav', int(i >= 2), path, band))
            streams.append((condition, records, 'ordinary' if condition == 'clean' else 'noisy_dev', records))
        records = [r for _, _, _, rs in streams for r in rs]
        baseline = root/'original'/'stage3'/'best_model.pt'
        baseline.parent.mkdir(parents=True)
        model_config = Wav2Vec2BertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, conv_depthwise_kernel_size=3, layerdrop=0., apply_spec_augment=False).to_dict()
        config = dict(stage=3, baseline_path=str(baseline), ssl_path=str(model_dir), dev_protocol=str(paths[0]),
            dev_data_path=str(root/'audio'), dev_noisy_cache=str(root/'seen'), dev_heldout_cache=str(root/'held'),
            data_fingerprints=fingerprints, model_config=model_config, eval_batch=5, eval_microbatch=4, feature_cache=None)
        for path, bias, epoch in [(baseline, 0., 20), (stage/'candidate_best.pt', -.08, 1)]:
            torch.save(dict(schema=SCHEMA, stage=3, kind='weights', epoch=epoch, model=TinyDetector(bias).state_dict(),
                            model_config=model_config, data_fingerprints=fingerprints), path)
        config['init_sha256'] = sha256(baseline)
        (stage/'config.json').write_text(json.dumps(config), encoding='utf-8')
        (stage/'completed.json').write_text(json.dumps(dict(candidate_epoch=1)), encoding='utf-8')
        old = audit.replay_metrics(records, [s for _, _, _, rs in streams for s in score_rows(rs, 0.)])
        new = audit.replay_metrics(records, [s for _, _, _, rs in streams for s in score_rows(rs, -.08)])
        (stage/'baseline_dev.json').write_text(json.dumps(old), encoding='utf-8')
        (stage/'epoch_001_evaluation.json').write_text(json.dumps(dict(dev=new)), encoding='utf-8')
        args = SimpleNamespace(run_dir=str(stage.parent), out=str(root/'audit'), download_dir=str(root/'temp'),
                               device='cpu', workers=0, bootstrap=12, resume=False, log_file=None)
        return args, streams, records, baseline, stage

    def test_full_export_parity_preservation_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            args, streams, records, baseline, stage = self.fixture(Path(d))
            before = {p: sha256(p) for p in [baseline, stage/'candidate_best.pt', *(Path(r['audio_path']) for r in records)]}
            with patch.object(audit, 'dev_streams', return_value=streams), \
                 patch.object(audit, 'loader_for', side_effect=lambda ds, *a: batches(ds)), \
                 patch.object(audit.Detector, 'load', side_effect=lambda *a, **kw: TinyDetector()) as load:
                audit.run(args)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(before, {p: sha256(p) for p in before})
            out = Path(args.out)
            for tag in ('baseline', 'candidate'):
                comparison = audit.read_json(out/f'{tag}_model_config_check.json')
                self.assertTrue(comparison['matched'])
                self.assertIn('id2label', comparison['representation_only_keys'])
            self.assertTrue(all(x['matched'] for x in audit.read_json(out/'metric_parity.json').values()))
            archive = Path(args.download_dir)/(out.name+'.zip')
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                names = z.namelist()
                self.assertEqual(len(names), len(set(names)))
                self.assertFalse(any(n.endswith(('.wav', '.pt', '.npy')) for n in names))
                inventory = json.loads(z.read(out.name+'/file_hashes.json'))
                for name, value in inventory.items():
                    self.assertEqual(audit.hashlib.sha256(z.read(out.name+'/'+name)).hexdigest(), value)
            args.resume = True
            args.download_dir = str(Path(d)/'second_download')
            with patch.object(audit, 'dev_streams', return_value=streams), \
                 patch.object(audit.Detector, 'load', side_effect=AssertionError('Completed inference must not repeat')):
                audit.run(args)
            with zipfile.ZipFile(Path(args.download_dir)/(out.name+'.zip')) as z:
                self.assertEqual(len(z.namelist()), len(set(z.namelist())))
            bad = out/'baseline_clean_scores.jsonl'
            bad.write_text(bad.read_text()+'\n', encoding='utf-8')
            with patch.object(audit, 'dev_streams', return_value=streams), self.assertRaisesRegex(ValueError, 'Stored scores changed'):
                audit.run(args)

    def test_input_changes_and_protected_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            args, _, records, baseline, stage = self.fixture(Path(d))
            args.out = str(stage/'diagnosis')
            with self.assertRaisesRegex(ValueError, 'separate'):
                audit.run(args)
            Path(records[0]['audio_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'waveform changed'):
                audit.check_audio_unchanged(records)
            cfg = audit.read_json(stage/'config.json')
            Path(cfg['dev_protocol']).write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Dev input differs'):
                audit.input_fingerprints(cfg)

    def test_readonly_features_equal_official_and_existing_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            SeamlessM4TFeatureExtractor().save_pretrained(root/'ssl')
            gen = torch.Generator().manual_seed(49)
            rows = [(torch.randn(64600, generator=gen)*.02, i, str(i)) for i in (0, 1)]
            writable = FeatureCollator(root/'ssl', cache_root=root/'cache')
            expected = writable(rows)
            files = {p: (sha256(p), p.stat().st_mtime_ns) for p in (root/'cache').rglob('*.npy')}
            cached = audit.ReadOnlyCollator(root/'ssl', cache_root=root/'cache')(rows)
            absent = audit.ReadOnlyCollator(root/'ssl', cache_root=root/'no_new_cache')(rows)
            for actual in (cached, absent):
                torch.testing.assert_close(actual['features'], expected['features'], rtol=0, atol=0)
            self.assertEqual(files, {p: (sha256(p), p.stat().st_mtime_ns) for p in files})
            self.assertFalse((root/'no_new_cache').exists())

    def test_parity_flags_any_changed_confusion(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, _, _, stage = self.fixture(Path(d))
            old = audit.read_json(stage/'baseline_dev.json')
            altered = json.loads(json.dumps(old))
            altered['seen']['bands'][3]['confusion'][1][0] += 1
            self.assertFalse(audit.metric_parity(altered, old)['matched'])

    def test_config_json_roundtrip_and_true_differences(self):
        checkpoint = Wav2Vec2BertConfig().to_dict()
        recorded = json.loads(json.dumps(checkpoint))
        self.assertNotEqual(checkpoint, recorded)  # Reproduces the reported false rejection.
        result = audit.compare_model_configs(checkpoint, recorded)
        self.assertTrue(result['matched'])
        self.assertEqual(result['representation_only_keys'], ['id2label'])
        for field, value in [('hidden_size', 2048), ('num_hidden_layers', 12), ('layer_norm_eps', 1e-3),
                             ('id2label', {'0': 'CHANGED_LABEL', '1': 'LABEL_1'})]:
            changed = dict(recorded, **{field: value})
            result = audit.compare_model_configs(checkpoint, changed)
            self.assertFalse(result['matched'])
            self.assertIn(field, result['differences'])
        missing = dict(recorded)
        del missing['pad_token_id']
        self.assertFalse(audit.compare_model_configs(checkpoint, missing)['matched'])

    def test_real_config_mismatch_persists_details_and_stops_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            args, streams, _, _, stage = self.fixture(Path(d))
            config = audit.read_json(stage/'config.json')
            config['model_config']['hidden_size'] = 128
            (stage/'config.json').write_text(json.dumps(config), encoding='utf-8')
            with patch.object(audit, 'dev_streams', return_value=streams), \
                 patch.object(audit.Detector, 'load', side_effect=AssertionError('Must stop before loading')):
                with self.assertRaisesRegex(ValueError, 'hidden_size'):
                    audit.run(args)
            details = audit.read_json(Path(args.out)/'baseline_model_config_check.json')
            self.assertFalse(details['matched'])
            self.assertEqual(details['differences']['hidden_size']['checkpoint'], 16)
            self.assertEqual(details['differences']['hidden_size']['recorded'], 128)


if __name__ == '__main__':
    unittest.main()
