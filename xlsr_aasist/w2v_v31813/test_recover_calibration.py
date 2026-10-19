import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from w2v_v31810.common import atomic_json,atomic_save,read_json,digest,fingerprint,capture_rng
from w2v_v31810.test_meta import assert_state_equal
from . import SCHEMA,VERSION
from .recover_calibration import recover,current_code,validate_code,validate_boundary,ADDED,OLD_STEP


def fixture(root):
    root.mkdir();files={}
    for name,value in [('train_rows.json',[]),('dev_rows.json',[]),('source_registry.json',{}),
                       ('validation_protocol.json',{}),('parent_config.json',{})]:
        atomic_json(root/name,value);files[name]=digest(root/name)
    code={k:OLD_STEP if Path(k).name=='step.py' else v for k,v in current_code().items() if Path(k).name not in ADDED}
    cfg=dict(version=VERSION,recipe=SCHEMA,epochs=3,warm_epochs=1,version_code=code,saved_manifests=files)
    parameter=torch.nn.Parameter(torch.tensor([.2,-.7]));optimizer=torch.optim.AdamW([parameter],lr=.002)
    parameter.square().sum().backward();optimizer.step()
    state=dict(schema=SCHEMA,identity=fingerprint(cfg),epoch=1,next_step=0,weights=None,totals={},
        history=[dict(epoch=1,tag='epoch_1')],cursor=1403,last_tag='epoch_1',best_tag='parent',best_score=.9,
        model={'fusion.weight':parameter.detach().clone()},best_model={'fusion.weight':torch.ones(2)},
        optimizer=optimizer.state_dict(),grad_scaler=dict(scale=1024.),rng=capture_rng())
    atomic_json(root/'config.json',cfg);atomic_save(root/'last.pt',state);atomic_json(root/'training_history.json',state['history'])
    (root/'failure.log').write_text('calibrate: No usable prior calibration gradient; inspect Train pair coverage',encoding='utf-8')
    atomic_json(root/'submission_request.json',dict(protocol='retained',upload=True))
    return cfg,state


class RecoveryTests(unittest.TestCase):
    def test_copy_retains_every_training_state_and_original_files(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            run=Path(tmp)/'failed';cfg,state=fixture(run)
            before={p.name:digest(p) for p in run.iterdir()}
            stack.enter_context(patch('w2v_v31813.recover_calibration.verify_parent'))
            stack.enter_context(patch('w2v_v31813.recover_calibration.verify_registry'))
            stack.enter_context(patch('w2v_v31813.config.verify_parent'))
            stack.enter_context(patch('w2v_v31813.config.verify_registry'))
            out=recover(run);new=read_json(out/'config.json');saved=torch.load(out/'last.pt',weights_only=True)
            self.assertNotEqual(out,run);self.assertEqual(new['epochs'],3)
            self.assertEqual({p.name:digest(p) for p in run.iterdir()},before)
            self.assertEqual(saved.pop('identity'),fingerprint(new));state.pop('identity')
            assert_state_equal(self,saved,state)
            self.assertEqual(read_json(out/'submission_request.json'),dict(protocol='retained',upload=True))
            self.assertFalse((out/'failure.log').exists());self.assertTrue((out/'recovery/original_failure.log').exists())
            self.assertEqual(read_json(out/'calibration_recovery.json')['completed_epochs'],1)

    def test_rejects_unknown_code_and_post_calibration_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,state=fixture(Path(tmp)/'failed');current=current_code()
            validate_code(cfg['version_code'],current)
            altered=dict(cfg['version_code']);key=next(k for k in altered if k.endswith('objectives.py'));altered[key]='0'*64
            with self.assertRaises(ValueError):validate_code(altered,current)
            for field,value in [('epoch',2),('weights',dict(prior=.1,pcl=.01)),('next_step',1)]:
                changed=copy.deepcopy(state);changed[field]=value
                with self.assertRaises(ValueError):validate_boundary(cfg,changed,'calibrate: No usable prior calibration gradient')


if __name__=='__main__':unittest.main()
