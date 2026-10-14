"""Missing/stale pointers must not hide runs or silently select another model."""
from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import parent_discovery as discovery


def candidate(path,version='3.18.5',status='complete',aug5=False):
    path.mkdir(parents=True,exist_ok=True)
    for name in discovery.REQUIRED:(path/name).write_text('{}',encoding='utf-8')
    cfg=dict(version=version,recipe=discovery.RECIPE,variant='C3',risk_coefficient=.25,
             pooling='asp',view_factors=[5,5,1,3])
    if aug5:
        cfg.update(version='3.18.4',recipe=discovery.AUG5_RECIPE,view_factor=5)
        cfg.pop('view_factors')
    (path/'config.json').write_text(json.dumps(cfg),encoding='utf-8')
    (path/'completed.json').write_text(json.dumps(dict(version=cfg['version'],recipe=cfg['recipe'],
        status=status,best_tag='epoch_4_step_2417')),encoding='utf-8')
    return path.resolve()


class DiscoveryTests(unittest.TestCase):
    def test_completed_sealed_aug5_is_eligible_without_renaming_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=candidate(root/'exp/aug5_sealed_e4_fixture',aug5=True)
            before=(parent/'config.json').read_bytes()
            self.assertEqual(discovery.resolve_parent(root=root),parent)
            self.assertEqual(discovery.resolve_parent(parent,root=root),parent)
            self.assertEqual((parent/'config.json').read_bytes(),before)
            self.assertEqual(discovery.inspect_candidate(parent)['parent_view_factors'],[5,5,5,5])

    def test_both_supported_families_require_an_explicit_choice(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=candidate(root/'exp/aug5',aug5=True);b=candidate(root/'exp/targeted')
            with self.assertRaisesRegex(ValueError,'Multiple completed'):discovery.resolve_parent(root=root)
            self.assertEqual(discovery.resolve_parent(a,root=root),a)
            self.assertEqual(discovery.resolve_parent(b,root=root),b)

    def test_aug5_recipe_pooling_and_completion_are_not_inferred_from_name(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=candidate(root/'v3184_aug5',aug5=True)
            cfg=json.loads((parent/'config.json').read_text())
            for changed in (dict(cfg,recipe='v3184_scene_v1'),dict(cfg,pooling='mean'),
                            dict(cfg,view_factor=1),dict(cfg,variant='C2')):
                (parent/'config.json').write_text(json.dumps(changed))
                self.assertFalse(discovery.inspect_candidate(parent)['eligible'])
            (parent/'config.json').write_text(json.dumps(cfg))
            done=json.loads((parent/'completed.json').read_text())
            for changed in (dict(done,version='3.18.5'),dict(done,recipe=discovery.RECIPE),
                            dict(done,status='running')):
                (parent/'completed.json').write_text(json.dumps(changed))
                self.assertFalse(discovery.inspect_candidate(parent)['eligible'])

    def test_missing_pointer_finds_sibling_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'RTC-w2v-improved/xlsr_aasist'
            root.mkdir(parents=True)
            parent=candidate(base/'RTC/xlsr_aasist/exp/actual_training_name')
            self.assertEqual(discovery.resolve_parent(root=root),parent)
            self.assertEqual(discovery.resolve_parent('',root=root),parent)
            self.assertFalse((root/'exp/.latest_v3185_run').exists())

    def test_valid_relative_pointer_preserves_choice_among_multiple_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'project/xlsr_aasist'
            parent=candidate(root/'exp/chosen');candidate(root/'exp/newer')
            (root/'exp/.latest_v3185_run').write_text('exp/chosen\n')
            self.assertEqual(discovery.resolve_parent(root=root),parent)

    def test_stale_and_empty_pointer_find_unique_compatible_run(self):
        with tempfile.TemporaryDirectory() as d,redirect_stderr(io.StringIO()):
            root=Path(d)/'project/xlsr_aasist';parent=candidate(root/'exp/parent')
            pointer=root/'exp/.latest_v3185_run'
            for value in ('','/missing/old/run'):
                pointer.write_text(value)
                self.assertEqual(discovery.resolve_parent(root=root),parent)

    def test_multiple_candidates_never_choose_by_date_or_score(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'project/xlsr_aasist'
            a=candidate(root/'exp/first');b=candidate(root/'exp/second')
            with self.assertRaisesRegex(ValueError,'Multiple completed') as error:
                discovery.resolve_parent(root=root)
            self.assertIn(str(a),str(error.exception));self.assertIn(str(b),str(error.exception))
            self.assertEqual(discovery.resolve_parent(a,root=root),a)

    def test_wrong_version_incomplete_or_missing_files_are_not_parents(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'project/xlsr_aasist'
            a=candidate(root/'exp/v3184',version='3.18.4')
            b=candidate(root/'exp/unfinished',status='running')
            c=candidate(root/'exp/missing');(c/'last.pt').unlink()
            with self.assertRaisesRegex(FileNotFoundError,'No compatible') as error:
                discovery.resolve_parent(root=root)
            self.assertIn('3.18.4',str(error.exception))
            for path in (a,b,c):
                with self.assertRaises(ValueError):discovery.resolve_parent(path,root=root)

    def test_duplicate_search_roots_do_not_create_ambiguous_parents(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=candidate(root/'exp/parent')
            self.assertEqual(discovery.resolve_parent(root=root,search_roots=[root/'exp']*2+[parent]),parent)

    def test_stdout_is_only_path_and_ambiguous_errors_go_to_stderr(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'project/xlsr_aasist';parent=candidate(root/'exp/parent')
            with patch.object(discovery,'ROOT',root),patch('sys.argv',['discovery']),redirect_stdout(io.StringIO()) as out:
                self.assertEqual(discovery.main(),0)
            self.assertEqual(out.getvalue(),str(parent)+'\n')
            candidate(root/'exp/second')
            with patch.object(discovery,'ROOT',root),patch('sys.argv',['discovery']),redirect_stdout(io.StringIO()) as out,redirect_stderr(io.StringIO()) as err:
                self.assertEqual(discovery.main(),2)
            self.assertEqual(out.getvalue(),'');self.assertIn('--parent-run PATH',err.getvalue())

    def test_explicit_invalid_parent_never_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);candidate(root/'exp/good')
            with self.assertRaises(ValueError):discovery.resolve_parent(root/'wrong',root=root)


if __name__=='__main__':unittest.main()
