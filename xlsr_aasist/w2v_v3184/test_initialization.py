"""Full R2 best and BN inheritance, reject malformed transfer before mutations."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .test_support import tiny_cfg,tiny_model
from .common import partial_state,seed_all
from .parent import initialize


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self):
        cfg=tiny_cfg();cfg['initialization']=dict(mode='parent-transfer',run='fixture',tag='epoch_2_step_3',
            checkpoint_sha256='fixture-hash',files={},exact_continuation=True)
        seed_all(17);parent=tiny_model(cfg)
        with torch.no_grad():
            for name,value in parent.named_parameters():
                if name.endswith('.lora_b'):value.normal_(0,.01)
            for name,value in parent.named_buffers():
                if name.endswith('running_mean'):value.fill_(.03)
                if name.endswith('running_var'):value.fill_(1.3)
                if name.endswith('num_batches_tracked'):value.fill_(19)
        values=partial_state(parent);seed_all(17);child=tiny_model(cfg)
        meta=dict(tag=cfg['initialization']['tag'],checkpoint_sha256='fixture-hash')
        old=dict(cfg,version='3.18.3-R2')
        return cfg,old,values,meta,parent,child

    def test_r2_all_tensors_and_predictions_preserved_including_bn(self):
        cfg,old,values,meta,parent,child=self.fixture()
        with patch('w2v_v3184.parent.inspect_parent',return_value=(old,values,meta)):audit=initialize(child,cfg)
        self.assertTrue(audit['exact_continuation']);self.assertEqual(audit['bn_statistics_reset'],[])
        for key,value in partial_state(child).items():torch.testing.assert_close(value,values[key],rtol=0,atol=0)
        parent.eval();child.eval();x=np.sin(np.arange(10000)*.03).astype(np.float32)
        with torch.no_grad():torch.testing.assert_close(parent([x]),child([x]),rtol=0,atol=0)
        self.assertFalse(audit['optimizer_inherited'])
        self.assertTrue(all(not p.requires_grad for n,p in child.named_parameters() if n.endswith(('.lora_a','.lora_b'))))

    def test_malformed_dtype_or_inventory_leaves_model_unchanged(self):
        cfg,old,values,meta,parent,child=self.fixture();before=partial_state(child)
        for mode in ('dtype','missing','nan'):
            bad=copy.deepcopy(values);key='fusion.layer_logits'
            if mode=='dtype':bad[key]=bad[key].double()
            elif mode=='missing':bad.pop(key)
            else:bad[key][0]=float('nan')
            with self.subTest(mode=mode),patch('w2v_v3184.parent.inspect_parent',return_value=(old,bad,meta)):
                with self.assertRaises(ValueError):initialize(child,cfg)
            for name,value in partial_state(child).items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_explicit_old_parent_uses_structural_migration(self):
        cfg,old,values,meta,parent,child=self.fixture();old['version']='3.18.3'
        with patch('w2v_v3184.parent.inspect_parent',return_value=(old,values,meta)), \
             patch('w2v_v3183r2.parent.initialize',return_value={'bn_statistics_reset':['one']}) as migrate:
            result=initialize(child,cfg)
        migrate.assert_called_once_with(child,cfg);self.assertFalse(result['exact_continuation'])


if __name__=='__main__':unittest.main()
