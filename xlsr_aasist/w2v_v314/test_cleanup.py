"""Interrupted compaction, source protection and unexpected-file refusal."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from w2v_v39.common import atomic_json, digest, read_json
from .cleanup import cleanup
from .state import SCHEMA, identity, load_selected


def fixture(root):
    root = Path(root)
    source = root/'source_last.pt'
    torch.save({'weight':torch.tensor([93.558])},source)
    run = root/'v314'; run.mkdir()
    cfg = dict(version='3.14',base_checkpoint=str(source),base_checkpoint_sha256=digest(source),
        starting_checkpoint=str(source),starting_checkpoint_sha256=digest(source),
        source_fingerprints={},data_fingerprints={},disk_margin_bytes=0)
    selection = dict(best_guarded='starting_last',best_weighted='joint')
    model = dict(weight=torch.tensor([[1.,2.],[-1.,-2.]]),bias=torch.zeros(2))
    state = dict(schema=SCHEMA, identity=identity(cfg),cursor=2,last_tag='joint',selections=selection,
        model=model,candidates={'joint':dict(kind='partial',state=model)},
        history=[dict(tag='joint',cursor=2,committed=True)],
        optimizer={'state':torch.ones(40000)},rng={'torch':torch.get_rng_state()})
    torch.save(state,run/'last.pt')
    atomic_json(run/'config.json',cfg)
    done = dict(version='3.14',status='complete',state_file='last.pt',checkpoint_sha256=digest(run/'last.pt'),
        selections=selection,last_tag='joint',committed_updates=2,
        base_checkpoint_sha256=digest(source),starting_checkpoint_sha256=digest(source))
    atomic_json(run/'completed.json',done)
    for kind in ('best_weighted','best_guarded','last'):
        atomic_json(run/(kind+'.json'),dict(selector=kind,checkpoint='last.pt'))
    return run, cfg


class CleanupTests(unittest.TestCase):
    def test_manifest_commit_interruption_leaves_valid_previous_checkpoint_and_retry_works(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            run,cfg = fixture(root)
            protected = digest(cfg['starting_checkpoint'])
            real = atomic_json
            def fail_manifest(path,value):
                if Path(path).name=='completed.json':
                    raise InterruptedError('before pointer switch')
                return real(path,value)
            with patch('w2v_v314.cleanup.atomic_json',side_effect=fail_manifest):
                with self.assertRaises(InterruptedError):
                    cleanup(run,apply=True)
            self.assertTrue((run/'last.pt').is_file())
            load_selected(run,'last')
            cleanup(run,apply=True)
            self.assertFalse((run/'last.pt').exists())
            self.assertEqual(read_json(run/'completed.json')['state_file'],'inference.pt')
            load_selected(run,'best_weighted')
            self.assertEqual(digest(cfg['starting_checkpoint']),protected)

    def test_incomplete_run_and_external_targets_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            run,_=fixture(root)
            previous=digest(run/'last.pt')
            real=Path.is_symlink
            target=(run/'inference.pt').resolve()
            with patch.object(Path,'is_symlink',lambda p:p in (run/'inference.pt',target) or real(p)):
                with self.assertRaises(ValueError):cleanup(run,apply=True)
            self.assertEqual(digest(run/'last.pt'),previous)
            done=read_json(run/'completed.json');done['status']='training'
            atomic_json(run/'completed.json',done)
            with self.assertRaises(ValueError):cleanup(run,apply=True)
            self.assertEqual(digest(run/'last.pt'),previous)

    def test_future_compaction_destination_cannot_overwrite_a_protected_input(self):
        with tempfile.TemporaryDirectory() as root:
            run,cfg=fixture(root)
            source=run/'inference.pt'
            torch.save({'protected':torch.tensor([1.])},source)
            altered=copy.deepcopy(cfg)
            altered['source_fingerprints'][str(source)]=digest(source)
            real=read_json
            previous=digest(source)
            with patch('w2v_v314.cleanup.read_json',side_effect=lambda p:altered if Path(p).name=='config.json' else real(p)):
                with self.assertRaisesRegex(ValueError,'protected checkpoint'):
                    cleanup(run,apply=True)
            self.assertEqual(digest(source),previous)

    def test_nonowned_feature_tree_is_never_removed(self):
        with tempfile.TemporaryDirectory() as root:
            run,_=fixture(root)
            path=run/'features';path.mkdir()
            important=path/'keep.txt';important.write_text('keep',encoding='utf-8')
            previous=digest(run/'last.pt')
            with self.assertRaisesRegex(ValueError,'unexpected'):
                cleanup(run,apply=True,remove_features=True)
            self.assertEqual(important.read_text(encoding='utf-8'),'keep')
            self.assertEqual(digest(run/'last.pt'),previous)

    def test_cleanup_cli_refuses_running_training(self):
        from .cleanup import main
        with patch('sys.argv',['cleanup','--run','unused','--apply']), \
             patch('w2v_aasist.full_workflow.ensure_idle',side_effect=RuntimeError('active training')), \
             patch('w2v_v314.cleanup.cleanup') as operation:
            with self.assertRaisesRegex(RuntimeError,'active training'):main()
            operation.assert_not_called()


if __name__=='__main__':unittest.main()
