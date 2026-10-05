"""Real small w2v-BERT + MultiConv cache reuse and actual single-model export."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .common import atomic_json, digest, read_json
from .config import code_fingerprints, configuration
from .data import borrowed_bundles
from .evaluate import export
from .model import ResidualClassifier, CenteredStudent
from .patch import save_patch, load_selected
from .workflow import Stages, run_experiment, verify_inputs
from .test_core import config as small_config


def source_fixture(root):
    from w2v_v36.test_workflow import fixture
    from w2v_v3.model import MultiConvHead, HeadConfig
    from w2v_v36.features import extract_cache
    from w2v_v37.language_cache import extract_language_cache
    from w2v_v37.patch import save_patch as old_save_patch
    cfg, model, records = fixture(root)
    model.head = MultiConvHead(HeadConfig(input_dim=16, projection=128, expansion=32,
                              kernels=(3, 5, 7, 9), merge_kernel=5, dropout=0.)).eval()
    base = Path(cfg['base_checkpoint'])
    processor = Path(cfg['ssl_path']) / 'preprocessor_config.json'
    fingerprints = {str(processor): digest(processor)}
    torch.save(dict(kind='weights', tag='baseline', model=model.state_dict(), **model.architecture(),
                    data_fingerprints=fingerprints), base)
    cfg.update(version='3.7', base_checkpoint_sha256=digest(base), code_fingerprints=code_fingerprints())
    source = root / 'v37'
    source.mkdir()
    for split in ('train', 'dev'):
        for row in records[split]:
            row['split'] = split
        identity = dict(base_checkpoint_sha256=cfg['base_checkpoint_sha256'], data_fingerprints=fingerprints,
                        code_fingerprints=cfg['code_fingerprints'], split=split)
        cache = extract_cache(model, records[split], cfg, source / 'features' / split, identity)
        for key in ('x', 'logits'):
            cache[key]._mmap.close()
    class ConstantTeacher:
        identity = dict(fixture='constant teacher, tests only')
        def encode_waveforms(self, waveforms):
            return np.ones((len(waveforms), 256), dtype=np.float32) / 16.
    language = extract_language_cache(records['train'], cfg, source / 'language' / 'train',
        dict(base_checkpoint_sha256=cfg['base_checkpoint_sha256'], data_fingerprints=fingerprints, split='train'),
        encoder=ConstantTeacher())
    language['lid']._mmap.close()
    atomic_json(source / 'data_fingerprints.json', fingerprints)
    atomic_json(source / 'feature_reuse.json', {})
    old_save_patch(source / 'best_patch.pt', cfg, dict(selected='baseline', selected_patch=dict(
        weight=model.head.classifier[-1].weight.tolist(), bias=model.head.classifier[-1].bias.tolist(),
        language_state=None, student_state=None)))
    atomic_json(source / 'completed.json', dict(version='3.7', selected='baseline', baseline_fallback=True,
        base_checkpoint_sha256=cfg['base_checkpoint_sha256'], patch_sha256=digest(source / 'best_patch.pt')))
    args = type('Args', (), dict(source_run=str(source), device='cpu'))()
    new = configuration(args)
    new.update(small_config(), student_epochs=2, epochs=2)
    return source, new, model, records


def completed_v38(root, cfg, original, selected='baseline'):
    from w2v_v38.model import ResidualClassifier as PreviousClassifier
    from w2v_v38.patch import save_patch as save_previous
    run = root / 'v38'
    run.mkdir()
    layer = original.head.classifier[-1]
    previous_cfg = dict(cfg, version='3.8')
    spec = PreviousClassifier(layer.weight, layer.bias, arm=selected).spec()
    save_previous(run / 'best_patch.pt', previous_cfg, selected, spec)
    atomic_json(run / 'completed.json', dict(version='3.8', status='complete', selected=selected,
        baseline_fallback=selected == 'baseline', language_debias_applied=False,
        patch_sha256=digest(run / 'best_patch.pt'), base_checkpoint_sha256=cfg['base_checkpoint_sha256']))
    return run


class WorkflowTests(unittest.TestCase):
    def test_v38_baseline_unwraps_original_caches_and_defaults_prefer_v38(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, cfg, original, _ = source_fixture(root)
            current = completed_v38(root, cfg, original)
            before = {str(p): digest(p) for p in root.rglob('*') if p.is_file()}
            args = type('Args', (), dict(source_run=str(current), device='cpu'))()
            unwrapped = configuration(args)
            self.assertEqual(unwrapped['source_version'], '3.8')
            self.assertEqual(unwrapped['v37_run'], str(source.resolve()))
            self.assertEqual(unwrapped['base_checkpoint_sha256'], cfg['base_checkpoint_sha256'])
            self.assertEqual(unwrapped['seed'], 3901)
            self.assertEqual(unwrapped['lambda_grid'], [.005, .02])
            verify_inputs(unwrapped)
            with borrowed_bundles(unwrapped) as (_, _, paths):
                self.assertTrue(all(str(source.resolve()) in p for p in paths.values()))
            self.assertEqual(before, {str(p): digest(p) for p in root.rglob('*') if p.is_file()})
            (root / 'exp').mkdir()
            (root / 'exp' / '.latest_v37_run').write_text(str(source), encoding='utf-8')
            (root / 'exp' / '.latest_v38_run').write_text(str(current), encoding='utf-8')
            args.source_run = None
            with patch('w2v_v39.config.ROOT', root), patch('w2v_v39.config.code_fingerprints', return_value=cfg['code_fingerprints']):
                preferred = configuration(args)
            self.assertEqual(preferred['source_run'], str(current.resolve()))
            self.assertTrue(any(Path(p).parent.name == 'w2v_v38' and Path(p).name == 'patch.py'
                                for p in cfg['code_fingerprints']))

    def test_v38_nonbaseline_and_tampered_source_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            _, cfg, original, _ = source_fixture(root)
            current = completed_v38(root, cfg, original, selected='calibration')
            args = type('Args', (), dict(source_run=str(current), device='cpu'))()
            with self.assertRaisesRegex(ValueError, 'discard an existing V3.8 correction'):
                configuration(args)
            path = current / 'best_patch.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            saved['tag'] = saved['spec']['arm'] = 'baseline'
            saved['baseline_fallback'] = True
            saved['spec']['state'].pop('raw_scale')
            saved['spec']['state'].pop('raw_shift')
            saved['config']['v37_patch_sha256'] = 'changed'
            torch.save(saved, path)
            done = read_json(current / 'completed.json')
            done.update(selected='baseline', baseline_fallback=True, patch_sha256=digest(path))
            atomic_json(current / 'completed.json', done)
            with self.assertRaisesRegex(ValueError, 'Pinned input changed'):
                configuration(args)

    def test_source_or_runtime_changes_cannot_resume_committed_stages(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            _, cfg, original, _ = source_fixture(root)
            current = completed_v38(root, cfg, original)
            args = type('Args', (), dict(source_run=str(current), device='cpu'))()
            unwrapped = configuration(args)
            with patch('w2v_v39.workflow.runtime_versions', return_value={'torch': 'different'}):
                with self.assertRaisesRegex(ValueError, 'runtime changed'):
                    verify_inputs(unwrapped)
            with (current / 'completed.json').open('a', encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'Pinned input changed'):
                verify_inputs(unwrapped)

    def test_real_completed_caches_train_resume_and_full_wave_submission(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source, cfg, original, records = source_fixture(root)
            snapshots = {str(p): digest(p) for p in source.rglob('*') if p.is_file()}
            base_hash = digest(cfg['base_checkpoint'])
            run = root / 'v39'
            run.mkdir()
            atomic_json(run / 'config.json', cfg)
            with patch('w2v_v3.model.Detector.forward', side_effect=AssertionError('No encoder forward during cached fit')), \
                    patch('w2v_v37.language.ensure_language_assets', side_effect=AssertionError('No download or teacher')):
                done = run_experiment(cfg, run)
                with patch('w2v_v39.student.train_student', side_effect=AssertionError('Committed stages reused')):
                    second = run_experiment(cfg, run)
            self.assertEqual(done['selected'], second['selected'])
            self.assertEqual(snapshots, {str(p): digest(p) for p in source.rglob('*') if p.is_file()})
            self.assertEqual(base_hash, digest(cfg['base_checkpoint']))
            self.assertFalse((run / 'features').exists())
            self.assertFalse((run / 'language').exists())
            self.assertLess(sum(p.stat().st_size for p in run.rglob('*') if p.is_file()), 10 * 1024**2)
            self.assertEqual(read_json(run / 'cache_reuse.json')['new_audio_bytes'], 0)
            result = read_json(run / 'fit_report.json')
            self.assertEqual(result['train_views'], len(records['train']))
            self.assertEqual(result['split']['group_overlap'], 0)
            selected, _ = load_selected(run)
            self.assertEqual(selected['spec']['arm'], done['selected'])
            # Force a nonzero language patch ONLY to exercise the actual export plumbing.
            layer = original.head.classifier[-1]
            student = CenteredStudent(512, 256, 8, torch.zeros(512), torch.ones(512))
            module = ResidualClassifier(layer.weight, layer.bias, arm='language_residual',
                mean=torch.zeros(512), scale=torch.ones(512), hidden=8, student=student)
            with torch.no_grad():
                student.output.weight.fill_(.015)
                student.output.bias.fill_(.025)
                module.hidden.weight[:, 512:].fill_(.01)
                module.output.weight.fill_(.03)
                module.output.bias.fill_(.2)
                probe = torch.randn(3, 512)
                margin = torch.nn.functional.linear(probe, layer.weight, layer.bias).diff(dim=1).neg().squeeze(1)
                actual_context = module.adjustment(probe, margin)
                empty_context = module.adjustment(probe, margin, context=torch.zeros(3, 256))
                self.assertFalse(torch.allclose(actual_context, empty_context))
            save_patch(run / 'best_patch.pt', cfg, 'language_residual', module.spec())
            done.update(selected='language_residual', baseline_fallback=False, language_debias_applied=True,
                        patch_sha256=digest(run / 'best_patch.pt'))
            atomic_json(run / 'completed.json', done)
            rows = [records['dev'][i] for i in (6, 0, 9, 3)]
            protocol = root / 'progress.txt'
            protocol.write_text('\n'.join(r['id'] for r in rows) + '\n', encoding='utf-8')
            from w2v_v36.features import inference_batches
            from w2v_v3.step import predict
            from .patch import apply_patch
            patched, _ = load_selected(run)
            original = apply_patch(original, patched)
            expected = []
            for examples in inference_batches(rows, cfg):
                with torch.inference_mode():
                    expected.extend(predict(original, examples, torch.device('cpu'), 'none', 2, 2400).softmax(-1)[:, 0].tolist())
            with patch('w2v_v37.language.ensure_language_assets', side_effect=AssertionError('No teacher at inference')):
                archive = export(run, protocol, root / 'audio', root / 'submission', 'cpu', 0)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(), ['scores.txt'])
                scored = [r.split() for r in z.read('scores.txt').decode().splitlines()]
            self.assertEqual([r[0] for r in scored], [r['id'] for r in rows])
            np.testing.assert_allclose([float(r[1]) for r in scored], expected, rtol=0, atol=1e-9)
            meta = read_json(root / 'submission' / 'submission_meta.json')
            self.assertTrue(meta['language_debias_applied'])
            self.assertFalse(meta['external_teacher_at_inference'])
            self.assertEqual(meta['base_checkpoint_sha256'], base_hash)

    def test_missing_or_corrupted_cache_is_not_regenerated(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, cfg, _, _ = source_fixture(Path(directory))
            path = source / 'features' / 'train' / 'x.npy'
            with path.open('r+b') as stream:
                stream.seek(-4, 2)
                stream.write(b'bad!')
            with self.assertRaisesRegex(ValueError, 'changed'), borrowed_bundles(cfg):
                self.fail('Corrupted data must not load')

    def test_stage_reuse_checks_bytes_identity_and_incomplete_commit(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            stage = Stages(directory, dict(seed=1))
            first = stage('one', lambda: dict(x=torch.tensor([2.])))
            second = stage('one', lambda: self.fail('Already committed'))
            torch.testing.assert_close(first['x'], second['x'])
            with self.assertRaises(ValueError):
                Stages(directory, dict(seed=2))('one', lambda: {})
            file = Path(directory) / 'stages' / 'one.pt'
            file.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                stage('one', lambda: {})
            stage.path.joinpath('two.pt').write_bytes(b'uncommitted')
            self.assertEqual(stage('two', lambda: dict(value=3)), dict(value=3))

    def test_patch_tampering_and_other_base_are_rejected(self):
        from .test_core import bundle
        from .patch import apply_patch
        data, w, b = bundle()
        with tempfile.TemporaryDirectory() as folder:
            cfg = dict(base_checkpoint='best.pt', base_checkpoint_sha256='a', base_tag='baseline')
            spec = ResidualClassifier(w, b).spec()
            path = Path(folder) / 'best_patch.pt'
            saved = save_patch(path, cfg, 'baseline', spec)
            atomic_json(Path(folder) / 'completed.json', dict(version='3.9', selected='baseline',
                        patch_sha256=digest(path), base_checkpoint_sha256='a'))
            load_selected(folder)
            saved['threshold'] = .9
            torch.save(saved, path)
            marker = read_json(Path(folder) / 'completed.json')
            marker['patch_sha256'] = digest(path)
            atomic_json(Path(folder) / 'completed.json', marker)
            with self.assertRaises(ValueError):
                load_selected(folder)
            model = torch.nn.Module()
            model.head = torch.nn.Module()
            model.head.classifier = torch.nn.Sequential(torch.nn.Linear(8, 2))
            with self.assertRaisesRegex(ValueError, 'Original classifier'):
                from .patch import SCHEMA
                apply_patch(model, dict(schema=SCHEMA, spec=spec))


if __name__ == '__main__':
    unittest.main()
