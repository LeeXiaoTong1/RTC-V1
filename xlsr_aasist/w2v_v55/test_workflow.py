import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import config, parent, workflow, delivery
from .common import read_json
from .test_configuration import submitted_parent
from .test_lifecycle import Model


class WorkflowTests(unittest.TestCase):
    def test_final_weights_anchor_to_submitted_v51_and_use_locked_tail_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            run,_,weights=submitted_parent(base)
            old,origin,rows,_=parent.metadata(run)
            args=config.parser().parse_args(['--workers','0'])
            with patch.object(config,'runtime_versions',return_value={}):
                cfg=config.training_configuration(old,origin,args)
            cfg['window_resource_audit']={'fixture':True}
            root=base/'v55'
            workflow.persist_stage(root,cfg,rows,read_json(origin['source_registry']))
            tail={k:v+2 for k,v in weights.items()}
            state=dict(status='complete',tail_model=tail,model={k:v+10 for k,v in weights.items()})
            with patch.object(workflow,'run_stage',return_value=(Model(),state)), \
                 patch.object(workflow,'bind_parent_phonemes'), \
                 patch.object(workflow,'verify_inputs'):
                self.assertTrue(workflow.final_fit(root,cfg))
            final_cfg,selected,meta=delivery.selected(root)
            for name,value in weights.items():
                torch.testing.assert_close(selected[name],value+1,rtol=0,atol=0)
            self.assertEqual(meta['selected'],'final_anchor_tail_50')
            self.assertEqual(final_cfg['v5_input'],old['v5_input'])
            self.assertEqual(read_json(root/'completed.json')['evaluation'],None)
            self.assertEqual(read_json(root/'final_data_audit.json')['evaluation_rows'],0)
            # A pending run cannot silently switch its final averaging rule.
            changed=copy.deepcopy(cfg)
            changed['parent_init']['locked_recipe']['weights']=dict(name='last',source='last',parent_weight=0.)
            with self.assertRaisesRegex(ValueError,'Locked continuation recipe changed'):
                workflow.continuation_lock(root,changed)


if __name__=='__main__':
    unittest.main()
