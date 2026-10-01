"""Numerical equivalence and bounded placement checks, plus optional CUDA spill."""
import copy
import unittest
from unittest.mock import patch
import torch
from w2v_v3.model import MultiConvHead, diversity_cka
from w2v_v3.test_model_step import ToyDetector, small_head, tiny_detector
from .model import Detector, install_fast_fusion, install_runtime, microbatches
from .runtime import HybridActivationStore, _keep_on_gpu, _usable_cuda_bytes

torch.set_num_threads(1)


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)

    def test_install_preserves_parameters_keys_and_rng(self):
        model = tiny_detector()
        params = dict(model.named_parameters())
        keys = tuple(model.state_dict())
        rng = torch.get_rng_state().clone()
        self.assertIs(install_fast_fusion(model), model)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertTrue(all(dict(model.named_parameters())[k] is v for k, v in params.items()))
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)

    def test_twenty_five_states_loss_and_all_gradients_match(self):
        reference = MultiConvHead(small_head()).train()
        wrapper = torch.nn.Module(); wrapper.head = copy.deepcopy(reference)
        actual = install_fast_fusion(wrapper, 5).head.train()
        states = [torch.randn(3,17,16,requires_grad=True) for _ in range(25)]
        clones = [h.detach().clone().requires_grad_() for h in states]
        mask = torch.ones(3,17,dtype=torch.long); mask[0,14:] = 0
        zr, hr = reference(states,mask)
        za, ha = actual(clones,mask)
        lr = torch.nn.functional.cross_entropy(zr,torch.tensor([0,1,0])) + .01*diversity_cka(hr)
        la = torch.nn.functional.cross_entropy(za,torch.tensor([0,1,0])) + .01*diversity_cka(ha)
        lr.backward(); la.backward()
        torch.testing.assert_close(la,lr,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(za,zr,atol=2e-6,rtol=2e-5)
        for a,b in zip(actual.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=3e-5)
        for a,b in zip(clones,states):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=3e-5)

    def test_eval_is_bitwise_reference_and_strict_checkpoint_compatible(self):
        reference = tiny_detector().eval()
        actual = install_fast_fusion(copy.deepcopy(reference), 5).eval()
        actual.load_state_dict(reference.state_dict(), strict=True)
        for length in (7,13,28):
            x=torch.randn(2,length,160); mask=torch.ones(2,length,dtype=torch.long)
            with torch.no_grad():
                expected=reference(x,mask); got=actual(x,mask)
            for a,b in zip(expected,got): torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_checkpointed_encoder_joint_gradients_and_dropout_rng(self):
        reference = tiny_detector(checkpointing=True,dropout=.1).configure_trainable_layers(1).train()
        actual = install_fast_fusion(copy.deepcopy(reference),2).train()
        x=torch.randn(3,15,160); mask=torch.ones(3,15,dtype=torch.long)
        torch.manual_seed(90)
        zr,hr=reference(x,mask); lr=zr.square().sum()+.01*diversity_cka(hr);lr.backward()
        rng=torch.get_rng_state().clone()
        torch.manual_seed(90)
        za,ha=actual(x,mask); la=za.square().sum()+.01*diversity_cka(ha);la.backward()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for (an,a),(bn,b) in zip(actual.named_parameters(),reference.named_parameters()):
            self.assertEqual(an,bn)
            if a.grad is None:self.assertIsNone(b.grad)
            else:torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=5e-5)

    def test_padding_remains_rejected_and_invalid_chunk_rejected(self):
        model=install_fast_fusion(tiny_detector()).train()
        mask=torch.ones(1,7,dtype=torch.long);mask[0,-1]=0
        with self.assertRaisesRegex(ValueError,'exact-length'):
            model(torch.randn(1,7,160),mask)
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError):install_fast_fusion(model,value)


class StorageTests(unittest.TestCase):
    def test_placement_budget_and_transient_reserve_policy(self):
        self.assertTrue(_keep_on_gpu(20,30,50,100,40))
        self.assertFalse(_keep_on_gpu(21,30,50,100,40))
        self.assertFalse(_keep_on_gpu(20,30,50,39,40))
        self.assertTrue(_keep_on_gpu(20,30,50,40,40))

    def test_cuda_usable_memory_includes_only_own_unused_allocator_cache(self):
        with patch('torch.cuda.mem_get_info',return_value=(10,100)), \
             patch('torch.cuda.memory_reserved',return_value=30), \
             patch('torch.cuda.memory_allocated',return_value=8):
            self.assertEqual(_usable_cuda_bytes(torch.device('cuda:0')),32)
        with patch('torch.cuda.mem_get_info',return_value=(10,100)), \
             patch('torch.cuda.memory_reserved',return_value=5), \
             patch('torch.cuda.memory_allocated',return_value=8):
            self.assertEqual(_usable_cuda_bytes(torch.device('cuda:0')),10)

    def test_cpu_gradient_equivalence_and_no_cuda_queries(self):
        reference=ToyDetector();actual=copy.deepcopy(reference)
        x=torch.randn(3,12,5);mask=torch.ones(3,12,dtype=torch.long)
        z,h=reference(x,mask);loss=z.square().sum()+.01*diversity_cka(h);loss.backward()
        with patch('torch.cuda.mem_get_info',side_effect=AssertionError('CPU must not query CUDA')):
            store=HybridActivationStore(actual)
            with store:
                z,h=actual(x,mask);(z.square().sum()+.01*diversity_cka(h)).backward()
        self.assertEqual((store.bytes,store.gpu_bytes,store.offloaded_tensors),(0,0,0))
        for a,b in zip(actual.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_invalid_budget_and_unknown_host_memory(self):
        model=torch.nn.Linear(3,2)
        for kwargs in ({'gpu_budget_gib':-1.},{'host_budget_gib':float('inf')},
                       {'reserve_gib':float('nan')},{'gpu_budget_gib':True}):
            with self.assertRaises(ValueError):HybridActivationStore(model,**kwargs)
        with patch('w2v_v32.runtime.available_host_bytes',return_value=None):
            self.assertEqual(HybridActivationStore(model).limit,0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA residency/spill requires CUDA')
    def test_cuda_residency_aliases_cpu_spill_and_exact_gradients(self):
        reference=ToyDetector().cuda();actual=copy.deepcopy(reference)
        x=torch.randn(3,12,5,device='cuda');mask=torch.ones(3,12,dtype=torch.long,device='cuda')
        z,h=reference(x,mask);loss=z.square().sum()+.01*diversity_cka(h);loss.backward()
        # Force mixed residency/spill even for this tiny model.
        store=HybridActivationStore(actual,gpu_budget_gib=2000/1024**3,host_budget_gib=.1,reserve_gib=0)
        with store:
            z,h=actual(x,mask);(z.square().sum()+.01*diversity_cka(h)).backward()
        self.assertGreater(store.gpu_bytes,0)
        self.assertGreater(store.bytes,0)
        self.assertLessEqual(store.gpu_bytes,store.gpu_limit)
        self.assertEqual(len(store._kept_storages),0)
        for a,b in zip(actual.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)
        aliases=HybridActivationStore(actual,gpu_budget_gib=.1,host_budget_gib=.1,reserve_gib=0)
        temp=torch.randn(100,device='cuda',requires_grad=True)
        with aliases:
            aliases._pack(temp);aliases._pack(temp[10:30]);aliases._pack(temp.view(10,10))
            self.assertEqual(aliases.gpu_bytes,temp.untyped_storage().nbytes())
            self.assertEqual(aliases.gpu_kept_tensors,3)
            self.assertTrue(all(t.grad_fn is None for t in aliases._kept_storages.values()))


class PaddedModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(817)

    def test_buckets_preserve_native_features_labels_and_budget(self):
        lengths=(7,9,7,41,17,16,12,8)
        examples=[dict(features=torch.randn(1,n,160),mask=torch.ones(1,n).long()) for n in lengths]
        seen=[]
        for indices,features,mask in microbatches(examples,4,50,1.5):
            seen.extend(indices)
            self.assertLessEqual(len(indices),4)
            if len(indices)>1:
                self.assertLessEqual(features.shape[0]*features.shape[1],50)
                self.assertLessEqual(max(lengths[i] for i in indices)/min(lengths[i] for i in indices),1.5)
            for row,i in enumerate(indices):
                n=lengths[i]
                torch.testing.assert_close(features[row,:n],examples[i]['features'][0],atol=0,rtol=0)
                self.assertEqual(int(mask[row].sum()),n)
                self.assertEqual(float(features[row,n:].abs().sum()),0.)
        self.assertEqual(sorted(seen),list(range(len(examples))))
        self.assertEqual(len(seen),len(set(seen)))

    def test_training_padding_and_poison_match_exact_encoder_for_all_position_types(self):
        from dataclasses import asdict
        base=tiny_detector(checkpointing=True)
        for position_type in ('relative_key','relative','rotary'):
            with self.subTest(position_type=position_type):
                config=base.backbone.config.to_dict();config['position_embeddings_type']=position_type
                config['conformer_conv_dropout']=0.
                model=Detector.from_config(config,asdict(small_head()),True).configure_trainable_layers(1).train()
                waves=[torch.randn(1,n,160) for n in (13,21,15)]
                exact=[model(x,torch.ones(1,x.shape[1]).long()) for x in waves]
                padded=torch.nn.utils.rnn.pad_sequence([x[0] for x in waves],batch_first=True)
                mask=torch.arange(21)[None,:]<torch.tensor([13,21,15])[:,None]
                padded[~mask]=12345.
                got=model(padded,mask.long())
                for j in (0,1):
                    torch.testing.assert_close(got[j],torch.cat([e[j] for e in exact]),atol=2e-6,rtol=3e-5)

    def test_install_runtime_preserves_parameter_identity_and_eval_path(self):
        model=tiny_detector().eval();reference=copy.deepcopy(model)
        params=dict(model.named_parameters());rng=torch.get_rng_state().clone()
        self.assertIs(install_runtime(model),model)
        self.assertIsInstance(model,Detector)
        self.assertTrue(all(p is params[k] for k,p in model.named_parameters()))
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        x=torch.randn(1,13,160);mask=torch.ones(1,13).long()
        for a,b in zip(model(x,mask),reference(x,mask)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'exact-length'):
            model(torch.randn(1,13,160),torch.tensor([[1]*12+[0]]))

    def test_invalid_prefixes_and_pre_padded_examples_rejected(self):
        model=install_runtime(tiny_detector()).train()
        for mask in (torch.tensor([[1,0,1]]),torch.zeros(1,3).long(),torch.tensor([[1,2,0]])):
            with self.assertRaises(ValueError):model(torch.randn(1,3,160),mask)
        with self.assertRaisesRegex(ValueError,'valid native'):
            list(microbatches([dict(features=torch.randn(1,3,160),mask=torch.tensor([[1,1,0]]))]))


if __name__=='__main__':unittest.main()
