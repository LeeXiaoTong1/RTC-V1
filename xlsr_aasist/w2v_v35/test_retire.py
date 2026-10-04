"""Retirement deletes only regenerable V3.5 audio and one abandoned resume state."""
import json
import os
from pathlib import Path
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from w2v_aasist.runtime import atomic_save, sha256
from . import SCHEMA, cache
from .test_cache import fixture as cache_fixture, FakeRTC
from .test_losses import model
import retire_w2v_v35 as retire


def fixture(root):
    root = Path(root).resolve()
    cfg, rows = cache_fixture(root, per_group=1)
    cfg['train_cache_root'] = str(root/'data'/'rtc_v35'/'train')
    cfg['full_dev_cache_root'] = str(root/'data'/'rtc_v35'/'dev')
    with patch.object(cache, 'DiverseRTC', FakeRTC):
        cache.prepare_base(cfg, rows['train'], rows['dev'])
        old = cache.prepare_epoch(cfg, 5)
        current = cache.prepare_epoch(cfg, 6)
        worker = cache.EpochCache(cfg, 6)
        worker.get(next(row for row in rows['train'] if row['domain'] == 'offline'))
        worker.close()
    run = root/'exp'/'w2v_v35_fixture'; run.mkdir(parents=True)
    for name in ('reference', 'baseline'):
        path = root/(name+'.pt'); path.write_bytes(name.encode())
        key = 'reference_checkpoint' if name == 'reference' else name
        cfg.update({key:str(path), key+'_sha256':sha256(path)})
    cfg.update(version='3.5', run_dir=str(run))
    (run/'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    metadata = [Path(cfg['train_cache_root'])/name for name in ('owner.json','recipe.json','sources.json')]
    metadata += [Path(cfg['full_dev_cache_root'])/'sources.json']
    fingerprints = {str(path):sha256(path) for path in metadata}
    detector = model()
    state = dict(schema=SCHEMA, kind='weights', config=cfg, model=detector.state_dict(),
                 **detector.architecture(), data_fingerprints=fingerprints,
                 reference_checkpoint_sha256=cfg['reference_checkpoint_sha256'])
    for name in ('best_model.pt', 'best_candidate.pt', 'best_recovered_epoch_3_best_candidate.pt'):
        atomic_save(run/name, dict(state, epoch=5, tag='epoch_5'))
    (run/'last.pt').write_bytes(b'nonessential raw model and Adam')
    (run/'report.md').write_text('Keep report')
    (run/'epoch_5_scores.jsonl').write_text('Keep scores')
    old_cache = root/'data'/'rtc_v33'; old_cache.mkdir()
    (old_cache/'views.wav').write_bytes(b'old reusable full cache')
    return root, run, cfg, old, current


class RetirementTests(unittest.TestCase):
    def test_cli_defaults_to_preview_and_apply_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, _, _, current = fixture(directory)
            base=['retire_w2v_v35.py','--run',str(run)]
            with patch.object(retire,'ROOT',root), patch.object(retire,'run_lock',return_value=nullcontext()), \
                    patch.object(retire,'active_jobs',return_value=[]), \
                    patch.object(retire,'os',SimpleNamespace(name='posix',walk=os.walk)):
                with patch('sys.argv',base):retire.main()
                self.assertTrue((run/'last.pt').exists());self.assertTrue(current.exists())
                with patch('sys.argv',base+['--apply']):retire.main()
                self.assertFalse((run/'last.pt').exists());self.assertFalse(current.exists())

    def test_large_source_inventory_is_not_subject_to_small_config_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sources.json'
            path.write_text(json.dumps({'padding':'x'*(17*1024**2)}))
            self.assertEqual(len(retire.read_inventory(path)['padding']),17*1024**2)

    def test_readonly_snapshot_symlink_is_protected_and_retargeting_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();target=root/'blob';target.write_text('metadata')
            link=root/'snapshot.json'
            try:link.symlink_to(target)
            except OSError:self.skipTest('Symlink creation unavailable on this Windows host')
            protected=retire.guard(link)
            plan=dict(root=str(root),run=str(root),guards=[protected],files=[])
            retire.validate(plan)
            other=root/'other';other.write_text('metadata')
            link.unlink();link.symlink_to(other)
            with self.assertRaisesRegex(RuntimeError,'Protected file changed'):retire.validate(plan)

    def test_preview_and_apply_preserve_standalone_bests_originals_dev_and_old_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, cfg, old, current = fixture(directory)
            keep = list(run.glob('best*.pt')) + [Path(cfg['baseline']), Path(cfg['reference_checkpoint']),
                Path(cfg['full_dev_cache_root'])/'sources.json', root/'data'/'rtc_v33'/'views.wav',
                run/'report.md', run/'epoch_5_scores.jsonl']
            hashes = {str(path):sha256(path) for path in keep}
            plan = retire.build_plan(root, run, lambda: [])
            self.assertTrue((run/'last.pt').exists()); self.assertTrue(old.exists())
            self.assertFalse((run/'retired.json').exists())
            result = retire.apply_plan(plan, lambda: [], run/'retirement.json')
            self.assertEqual(result['status'], 'complete')
            self.assertFalse((run/'last.pt').exists())
            self.assertFalse(old.exists()); self.assertFalse(current.exists())
            self.assertEqual(hashes, {name:sha256(name) for name in hashes})
            self.assertFalse(json.loads((run/'retired.json').read_text())['resume_allowed'])
            self.assertTrue((Path(cfg['train_cache_root'])/'owner.json').exists())
            self.assertIsInstance(result['measured_free_change_bytes'], int)
            # Repeated retirement is harmless and retains all best weights.
            self.assertEqual(retire.build_plan(root,run,lambda:[])['files'], [])

    def test_refuses_active_job_in_planning_and_before_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, _, _, _ = fixture(directory)
            with self.assertRaisesRegex(RuntimeError, 'active'):
                retire.build_plan(root, run, lambda:[{'pid':123}])
            plan = retire.build_plan(root, run, lambda:[])
            with self.assertRaisesRegex(RuntimeError, 'active'):
                retire.apply_plan(plan, lambda:[{'pid':123}],run/'retirement.json')
            self.assertTrue((run/'last.pt').exists())

    def test_refuses_unknown_file_and_unknown_generation_and_previous_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, _, _, current = fixture(directory)
            for path in (current/'important.wav', run/'best_candidate.pt.previous'):
                path.write_bytes(b'do not remove')
                with self.assertRaises(ValueError):retire.build_plan(root, run, lambda:[])
                path.unlink()
            generation = current/'generation.json'
            value = json.loads(generation.read_text()); value['recipe_sha256']='incorrect'
            generation.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'generation'):
                retire.build_plan(root,run,lambda:[])
            self.assertTrue((run/'last.pt').exists())

    def test_refuses_wrong_run_namespace_and_missing_or_incomplete_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, _, _, _ = fixture(directory)
            with self.assertRaisesRegex(ValueError,'direct'):
                retire.build_plan(root, root/'exp'/'w2v_v34_other',lambda:[])
            candidate = run/'best_candidate.pt'
            state=torch.load(candidate,map_location='cpu',weights_only=True)
            state['model'].pop(next(iter(state['model'])))
            atomic_save(candidate,state)
            with self.assertRaisesRegex(ValueError,'complete model'):
                retire.build_plan(root,run,lambda:[])
            candidate.unlink()
            with self.assertRaisesRegex(ValueError,'must exist'):
                retire.build_plan(root,run,lambda:[])
            self.assertTrue((run/'last.pt').exists())

    def test_refuses_metadata_or_target_changed_after_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, _, _, current = fixture(directory)
            plan = retire.build_plan(root,run,lambda:[])
            (run/'last.pt').write_bytes(b'changed resume checkpoint')
            with self.assertRaisesRegex(RuntimeError,'target changed'):
                retire.apply_plan(plan,lambda:[],run/'retirement.json')
            plan = retire.build_plan(root,run,lambda:[])
            (run/'config.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Protected file changed'):
                retire.apply_plan(plan,lambda:[],run/'retirement.json')
            self.assertTrue((run/'last.pt').exists());self.assertTrue(current.exists())

    def test_refuses_symlink_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, _, _, current = fixture(directory)
            with self.assertRaises(ValueError):retire.safe(root/'data'/'..'/'exp',root)
            link=current/'foreign'
            try:link.symlink_to(root/'ssl',target_is_directory=True)
            except OSError:self.skipTest('Symlink creation unavailable on this Windows host')
            with self.assertRaisesRegex(ValueError,'Symlink'):
                retire.build_plan(root,run,lambda:[])
            self.assertTrue((run/'last.pt').exists())

    def test_new_unknown_file_after_plan_is_never_recursively_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, _, _, current = fixture(directory)
            plan=retire.build_plan(root,run,lambda:[])
            unknown=current/'keep-me.txt';unknown.write_text('external file')
            with self.assertRaises(OSError):
                retire.apply_plan(plan,lambda:[],run/'retirement.json')
            self.assertTrue(unknown.exists())
            self.assertEqual(json.loads((run/'retired.json').read_text())['status'],'interrupted')
            self.assertTrue((run/'best_model.pt').exists())


if __name__=='__main__':unittest.main()
