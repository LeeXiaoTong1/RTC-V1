"""Ensure preparation/transfer/checkpoint optimizations preserve actual math."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor
from w2v_v3.test_model_step import tiny_detector
from w2v_v31.data import ViewCollator
from w2v_v31.test_data_step import expanded_examples
from .batching import PreparedBatch, PreparedCollator, DeviceBatches
from .model import install_runtime, microbatches, _TrainableCheckpoint
from .step import supervised_step

torch.set_num_threads(1)


class PreparedTests(unittest.TestCase):
    def test_official_independent_features_exact_for_mixed_lengths_and_views(self):
        with tempfile.TemporaryDirectory() as td:
            SeamlessM4TFeatureExtractor().save_pretrained(td)
            rows=[]
            for i,n in enumerate((8191,16000,16160,16319,16320,32001,11019,9021,48000)):
                wave=np.random.default_rng(i).normal(0,.03,n).astype(np.float32)
                meta=dict(id=str(i),label=i%2,language='en' if i%3 else 'zh',noisy=i>3)
                rows.append([dict(meta,wave=wave,view='full',view_weight=.7),
                             dict(meta,wave=wave[:5101].copy(),view='short',view_weight=.3)])
            expected=ViewCollator(td)(rows)
            actual=PreparedCollator(td,4,100)(rows)
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):
                self.assertEqual(set(a),set(b))
                for key in a:
                    if isinstance(a[key],torch.Tensor):
                        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
                    else:self.assertEqual(a[key],b[key])
            original=list(microbatches(expected,4,100))
            self.assertEqual(len(actual.batches),len(original))
            for (ids,f,m),(ids2,f2,m2) in zip(actual.batches,original):
                self.assertEqual(ids,ids2)
                torch.testing.assert_close(f,f2,atol=0,rtol=0)
                torch.testing.assert_close(m,m2,atol=0,rtol=0)
            # Already padded batches are reused without another allocation.
            self.assertIs(list(microbatches(actual,4,100))[0][1],actual.batches[0][1])
            self.assertEqual(sum(len(b[0]) for b in microbatches(actual,1,100)),len(actual))

    def test_cpu_transfer_keeps_order_and_data(self):
        examples=expanded_examples(160)
        original=list(microbatches(examples,3,80))
        moved=list(DeviceBatches(iter(original),'cpu'))
        for (i,f,m),(j,g,n) in zip(original,moved):
            self.assertEqual(i,j);self.assertIs(f,g);self.assertIs(m,n)

    def test_prepared_features_preserve_dropout_gradients_and_adam_exactly(self):
        torch.manual_seed(174)
        model=install_runtime(tiny_detector(checkpointing=True,dropout=.2).configure_trainable_layers(1)).train()
        expected=copy.deepcopy(model)
        examples=expanded_examples(160)
        prepared=PreparedBatch(examples,list(microbatches(examples,3,80)),3,80)
        optimizers=[torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-6)
                    for m in (expected,model)]
        results=[];states=[]
        for m,ex,opt in zip((expected,model),(examples,prepared),optimizers):
            torch.manual_seed(415)
            results.append(supervised_step(m,ex,opt,torch.ones(2),'cpu',amp='none',
                           microbatch=3,frame_budget=80,offload_activations=False))
            states.append(torch.get_rng_state().clone())
        self.assertEqual(results[0][0],results[1][0])
        torch.testing.assert_close(results[0][1],results[1][1],rtol=0,atol=0)
        torch.testing.assert_close(states[0],states[1],rtol=0,atol=0)
        for a,b in zip(expected.parameters(),model.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        for a,b in zip(optimizers[0].state.values(),optimizers[1].state.values()):
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'Pinned lookahead transfer requires CUDA')
    def test_cuda_pins_final_batches_and_preserves_async_transfer(self):
        examples=expanded_examples(160)
        batch=PreparedBatch(examples,list(microbatches(examples,3,80)),3,80)
        from torch.utils.data._utils.pin_memory import pin_memory
        self.assertIs(pin_memory(batch),batch)
        self.assertTrue(all(f.is_pinned() and m.is_pinned() for _,f,m in batch.batches))
        transfer=DeviceBatches(batch.batches,'cuda')
        outputs=[]
        for ids,f,m in transfer:
            outputs.append((ids,f.square(),m.clone()))
        torch.cuda.synchronize()
        for (ids,f,m),(i,actual,mask) in zip(batch.batches,outputs):
            self.assertEqual(ids,i)
            torch.testing.assert_close(actual.cpu(),f.square(),rtol=0,atol=0)
            torch.testing.assert_close(mask.cpu(),m,rtol=0,atol=0)
        self.assertEqual(transfer.pinned_batches,len(batch.batches))
        self.assertEqual(transfer.copy_batches,len(batch.batches))


class FrozenCheckpointTests(unittest.TestCase):
    def test_frozen_skip_preserves_outputs_rng_gradients_and_optimizer_with_dropout(self):
        torch.manual_seed(123)
        model=install_runtime(tiny_detector(checkpointing=True,dropout=.2).configure_trainable_layers(1)).train()
        reference=copy.deepcopy(model)
        wrapped=reference.backbone.encoder._gradient_checkpointing_func
        reference.backbone.encoder._gradient_checkpointing_func=wrapped.original
        original=model.backbone.encoder._gradient_checkpointing_func.original
        calls=[]
        def checkpoint(function,*args,**kwargs):
            calls.append(function.__self__)
            return original(function,*args,**kwargs)
        model.backbone.encoder._gradient_checkpointing_func.original=checkpoint
        x=torch.randn(2,23,160);mask=torch.ones(2,23).long()
        optimizers=[torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-6)
                    for m in (reference,model)]
        outputs=[];states=[]
        for m,opt in zip((reference,model),optimizers):
            torch.manual_seed(671)
            z,h=m(x,mask);(z.square().sum()+h.square().mean()).backward();opt.step()
            outputs.append((z,h));states.append(torch.get_rng_state().clone())
        self.assertEqual(calls,[model.backbone.encoder.layers[1]])
        for a,b in zip(outputs[0],outputs[1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(states[0],states[1],rtol=0,atol=0)
        for a,b in zip(reference.parameters(),model.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        for a,b in zip(optimizers[0].state.values(),optimizers[1].state.values()):
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        model.configure_trainable_layers(2)
        self.assertTrue(all(not layer._rtc_frozen_checkpoint for layer in model.backbone.encoder.layers))

    def test_frozen_parameter_with_gradient_input_still_checkpoints(self):
        layer=torch.nn.Linear(3,3)
        layer.requires_grad_(False);layer._rtc_frozen_checkpoint=True
        from torch.utils.checkpoint import checkpoint
        called=[]
        def wrapped(fn,*args,**kwargs):
            called.append(True)
            return checkpoint(fn,*args,use_reentrant=False,**kwargs)
        optimized=_TrainableCheckpoint(wrapped)
        x=torch.randn(2,3,requires_grad=True)
        optimized(layer.__call__,x).sum().backward()
        self.assertEqual(called,[True]);self.assertIsNotNone(x.grad)


if __name__=='__main__':unittest.main()
