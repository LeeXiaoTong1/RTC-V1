import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

import cleanup_before_v312 as cleanup


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root/'exp').mkdir()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        return path

    def json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def folder(self, n):
        return self.root/'exp'/f'w2v_v3{n}_20261001_123456_abcd'

    def make(self, n, name='last.pt'):
        p = self.folder(n)/name
        self.save(p, {'schema': 'test_resume'})
        self.save(p.parent/'best_model.pt', {'schema': 'test_best'})
        return p

    def validate(self, target, root):
        return {str(target.parent/'best_model.pt'): cleanup.sha(target.parent/'best_model.pt')}

    def plan(self):
        return cleanup.plan(self.root, self.validate)

    def test_numeric_boundary_best_and_original_data_preserved(self):
        old = [self.make(n) for n in ('', '1', '9', '10', '11')]
        old.append(self.make('2', 'epoch_2_step_100.pt'))
        for n in ('12', '13', '15', '151', '16', '161', '17', '18'):
            self.make(n)
        for name in ('best_noisy.pt', 'control_best.pt', 'best_candidate.pt', 'candidate_best.pt'):
            self.save(self.folder('1')/name, {'keep': True})
        self.save(self.folder('1')/'stages'/'epoch_3.pt', {'keep': True})
        self.save(self.root/'exp'/'custom'/ 'last.pt', {'keep': True})
        self.json(self.root/'data'/'audio.wav', {'keep': True})
        before = {p: cleanup.sha(p) for p in self.root.rglob('*') if p.is_file()}
        result = self.plan()
        self.assertEqual({Path(i['path']) for i in result['candidates']}, set(old))
        cleanup.apply(result)
        for p, digest in before.items():
            if p in old:
                self.assertFalse(p.exists())
            else:
                self.assertEqual(cleanup.sha(p), digest)
        self.assertFalse(self.plan()['candidates'])

    def test_reviewed_v33_nested_branches_and_earlier_aasist(self):
        targets = []
        for branch in ('candidate', 'control'):
            p = self.folder('3')/branch/'last.pt'
            self.save(p, {'schema': 'test_resume'})
            self.save(p.parent/'best_model.pt', {'schema': 'test_best'})
            targets.append(p)
        p = self.root/'exp'/cleanup.EARLY_AASIST/'last.pt'
        self.save(p, {'schema': 'test_resume'})
        self.save(p.parent/'best_model.pt', {'schema': 'test_best'})
        targets.append(p)
        self.assertEqual({Path(x['path']) for x in self.plan()['candidates']}, set(targets))

    def test_explicit_best_dependencies_propagate_transitively(self):
        a, b = self.make('1'), self.make('2')
        self.save(a, {'base_checkpoint': str(b)})
        self.save(self.folder('18')/'best.pt', {'base_checkpoint': str(a)})
        self.assertFalse(self.plan()['candidates'])
        self.assertTrue(a.exists() and b.exists())

    def test_known_source_directory_uses_best_but_explicit_last_pins(self):
        p = self.make('1')
        q = self.folder('18')/'config.json'
        self.json(q, {'source_run': str(p.parent)})
        self.save(q.parent/'best.pt', {'config': {'source_run': str(p.parent)}})
        self.assertEqual(len(self.plan()['candidates']), 1)
        self.json(q, {'source_run': str(p.parent), 'source_fingerprints': {str(p): 'hash'}})
        self.assertFalse(self.plan()['candidates'])

    def test_unknown_or_resume_directory_reference_pins(self):
        p = self.make('1')
        q = self.root/'exp'/'custom'/'selection.json'
        self.json(q, {'source_run': str(p.parent)})
        self.assertFalse(self.plan()['candidates'])
        q.unlink()
        self.json(self.folder('18')/'config.json', {'resume': str(p.parent)})
        self.assertFalse(self.plan()['candidates'])

    def test_inventory_does_not_pin_and_no_audio_metadata_scan(self):
        self.make('1')
        self.json(self.root/'exp'/'storage_audit_old.json', {'paths': ['invalid']})
        p = self.folder('1')/'features'/'train'/'rows.json'
        p.parent.mkdir(parents=True)
        p.write_text('not JSON', encoding='utf-8')
        self.assertEqual(len(self.plan()['candidates']), 1)

    def test_invalid_best_retained_without_deletion(self):
        p = self.make('1')
        (p.parent/'best_model.pt').unlink()
        result = self.plan()
        self.assertFalse(result['candidates'])
        self.assertEqual(len(result['retained']), 1)
        self.assertTrue(p.exists())

    def test_changed_target_best_or_new_reference_cancels(self):
        a, b = self.make('1'), self.make('2')
        result = self.plan()
        self.save(b, {'changed': True})
        with self.assertRaisesRegex(ValueError, 'changed'):
            cleanup.apply(result)
        self.assertTrue(a.exists())
        result = self.plan()
        self.save(a.parent/'best_model.pt', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'changed'):
            cleanup.apply(result)
        self.assertTrue(a.exists())
        result = self.plan()
        self.json(self.folder('18')/'new_selection.json', {'base': str(a)})
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            cleanup.apply(result)
        self.assertTrue(a.exists())

    def test_hardlink_and_scope_tampering_blocked(self):
        a = self.make('1')
        alias = a.parent/'keep_copy.pt'
        os.link(a, alias)
        self.assertFalse(self.plan()['candidates'])
        alias.unlink()
        result = self.plan()
        best = a.parent/'best_model.pt'
        result['candidates'][0] = {'path': str(best), 'identity': cleanup.identity(best)}
        with self.assertRaisesRegex(ValueError, 'scope'):
            cleanup.apply(result)
        self.assertTrue(a.exists() and best.exists())

    def test_unreadable_retained_weight_aborts_and_journal_is_ignored(self):
        a = self.make('1')
        corrupt = self.root/'exp'/'unknown.pt'
        corrupt.write_bytes(b'unknown format')
        with self.assertRaises(ValueError):
            self.plan()
        self.assertTrue(a.exists())
        corrupt.unlink()
        result = self.plan()
        report = self.root/'exp'/'cleanup_before_v312_test.json'
        cleanup.write_report(report, result)
        cleanup.apply(result, report)
        saved = json.loads(report.read_text(encoding='utf-8'))
        self.assertEqual(len(saved['removed']), 1)

    def test_real_full_best_shape_validation_and_embedded_best(self):
        from transformers import Wav2Vec2BertConfig
        from w2v_v3.model import Detector
        cfg = Wav2Vec2BertConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=32, feature_projection_input_dim=160, conv_depthwise_kernel_size=7,
            num_conv_pos_embedding_groups=2)
        model = Detector.from_config(cfg.to_dict(), dict(input_dim=16, projection=8, expansion=32,
            blocks=4, kernels=(3, 7), merge_kernel=3, dropout=.1), checkpointing=False)
        p = self.make('1')
        schema = cleanup.FULL[1]
        self.save(p, dict(schema=schema, kind='training'))
        state = dict(schema=schema, kind='weights', model=model.state_dict(), **model.architecture())
        self.save(p.parent/'best_model.pt', state)
        result = cleanup.plan(self.root)
        self.assertEqual(len(result['candidates']), 1)
        state['model'].pop(next(iter(state['model'])))
        self.save(p.parent/'best_model.pt', state)
        self.assertFalse(cleanup.plan(self.root)['candidates'])
        self.save(p, dict(schema=schema, best_model={'embedded': torch.ones(1)}))
        with self.assertRaisesRegex(ValueError, 'Embedded best'):
            cleanup.verified_best(p, self.root)

    def test_real_v310_v311_fallback_best_and_base_survive(self):
        for n in (10, 11):
            module = cleanup.importlib.import_module('w2v_v3'+str(n)+'.state')
            p = self.make(str(n))
            base = self.save(self.folder('5')/'best_model.pt', {'tensor': torch.ones(1)})
            cfg = {'base_checkpoint': str(base), 'base_checkpoint_sha256': cleanup.sha(base)}
            best = dict(schema=module.SCHEMA, config=cfg, identity=module.identity(cfg),
                        selected='baseline', model=None)
            self.save(p.parent/'best.pt', best)
            self.json(p.parent/'completed.json', dict(version=f'3.{n}', status='complete', selected='baseline',
                checkpoint_sha256=cleanup.sha(p.parent/'best.pt'), base_checkpoint_sha256=cleanup.sha(base),
                baseline_fallback=True))
            self.save(p, dict(schema=module.SCHEMA, identity=best['identity'], selected='baseline', best_model=None))
            protected = cleanup.verified_best(p, self.root)
            self.assertIn(str(base), protected)
            self.save(p, dict(schema=module.SCHEMA, identity=best['identity'], selected='new', best_model=None))
            with self.assertRaisesRegex(ValueError, 'not an exported'):
                cleanup.verified_best(p, self.root)
            self.save(p, dict(schema=module.SCHEMA, identity=best['identity'], selected='baseline', best_model=None))
        result = cleanup.plan(self.root)
        self.assertEqual(len(result['candidates']), 2)
        protected = {p: cleanup.sha(p) for p in self.root.rglob('*') if p.is_file() and p.name != 'last.pt'}
        cleanup.apply(result)
        for p, digest in protected.items():
            self.assertEqual(cleanup.sha(p), digest)


if __name__ == '__main__':
    unittest.main()
