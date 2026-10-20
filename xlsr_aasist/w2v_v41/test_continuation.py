"""V4 -> V4.1 retains Adam, schedule, PCL, source cursor, RNG and selection."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from w2v_v4 import VERSION as OLD_VERSION, SCHEMA as OLD_SCHEMA
from w2v_v4.train import run_experiment as old_run
from w2v_v31810.test_meta import assert_state_equal
from .common import atomic_json, read_json, digest, fingerprint
from .test_core import Tiny, cfg
from .test_training import FixturePlan, FixtureWaves, FixtureLoader, optimizer, Evaluator
from .continuation import migrate
from .config import parser
from .train import run_experiment
from . import VERSION, SCHEMA


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def patches(self,stack,package):
        stack.enter_context(patch(package+'.train.Plan',FixturePlan))
        stack.enter_context(patch(package+'.train.Waves',FixtureWaves))
        stack.enter_context(patch(package+'.train.loader',lambda d,c,*,batch_sampler:FixtureLoader(d,batch_sampler)))
        stack.enter_context(patch(package+'.train.close'))

    def test_migrate_after_second_epoch_preserves_state_and_remaining_budget(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack,redirect_stdout(io.StringIO()):
            root=Path(directory);parent=root/'v4';parent.mkdir()
            settings=cfg(version=OLD_VERSION,recipe=OLD_SCHEMA,epochs=4,warm_epochs=1,
                         lr_floor=.1,lr_warmup_updates=2,free_reserve_bytes=0,workers=4,eval_workers=2,version_code={})
            for name,value in (('pcl_pairs.json',dict(official={},noise={})),('train_rows.json',[]),('validation_protocol.json',{})):
                atomic_json(parent/name,value)
            settings['saved_manifests']={p.name:digest(p) for p in parent.iterdir()}
            atomic_json(parent/'config.json',settings)
            self.patches(stack,'w2v_v4');self.patches(stack,'w2v_v41')
            evaluator=Evaluator(self,parent)
            def evaluate(*args,**kwargs):
                out=evaluator(*args,**kwargs)
                if kwargs['checkpoint_tag']=='epoch_2':atomic_json(parent/'stop_requested.json',dict(test=True))
                return out
            done=old_run(settings,parent,[],{},model_factory=Tiny,optimizer_factory=optimizer,evaluator=evaluate)
            self.assertEqual(done['status'],'paused')
            before=torch.load(parent/'last.pt',weights_only=True);sha=digest(parent/'last.pt')
            self.assertEqual((before['epoch'],before['next_step'],before['cursor']),(2,0,4))
            args=parser().parse_args(['--continue-from',str(parent),'--workers','6','--checkpoint-policy','fixed'])
            with patch('w2v_v41.continuation.verify_v4',return_value=settings):
                config,rows,protocol=migrate(args,root/'v41')
            after=torch.load(root/'v41/last.pt',weights_only=True)
            for key in before:
                if key not in ('identity','schema'):assert_state_equal(self,before[key],after[key])
            self.assertEqual(config['epochs'],4);self.assertEqual(config['workers'],6)
            self.assertEqual(after['identity'],fingerprint(config));self.assertEqual(after['schema'],SCHEMA)
            self.assertEqual(digest(parent/'last.pt'),sha)
            ev=Evaluator(self,root/'v41')
            with patch('w2v_v41.train.calibrate',side_effect=AssertionError('PCL recalibrated')):
                run_experiment(config,root/'v41',rows,protocol,model_factory=Tiny,optimizer_factory=optimizer,evaluator=ev)
            final=torch.load(root/'v41/last.pt',weights_only=True)
            self.assertEqual((final['epoch'],final['cursor']),(4,8))
            self.assertEqual([r['epoch'] for r in final['history']],[1,2,3,4])
            self.assertEqual(ev.calls[:2],[('select','epoch_3'),('select','epoch_4')])
            self.assertEqual(final['best_tag'],before['best_tag'])
            self.assertEqual(final['weights'],before['weights'])

    def test_migration_rejects_unpaused_or_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'v4';parent.mkdir()
            atomic_json(parent/'paused.json',dict(status='paused',version='4.0',checkpoint_sha256='wrong'))
            (parent/'last.pt').write_bytes(b'changed')
            args=parser().parse_args(['--continue-from',str(parent)])
            with patch('w2v_v41.continuation.verify_v4',return_value={}):
                with self.assertRaisesRegex(ValueError,'Stop V4 safely'):migrate(args,root/'v41')
            self.assertFalse((root/'v41').exists())


if __name__=='__main__':unittest.main()
