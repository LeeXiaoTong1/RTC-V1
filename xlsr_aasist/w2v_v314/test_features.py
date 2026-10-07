"""Real small w2v-BERT/MultiConv replay, ordering, recovery and reuse checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import SeamlessM4TFeatureExtractor

from w2v_v3.test_model_step import tiny_detector
from .features import extract_cache,load_cache,inference_batches

torch.set_num_threads(1)


def fixture(root):
    root=Path(root); processor=root/'processor'
    SeamlessM4TFeatureExtractor().save_pretrained(processor)
    rows=[]
    for i,length in enumerate((8000,11200,8000)):
        path=root/f'{i}.wav'
        sf.write(path,np.random.default_rng(i).normal(0,.1,length).astype(np.float32),16000,subtype='FLOAT')
        rows.append(dict(id=f'offline/en/{i}.wav',source_id=str(i),group_id=str(i),condition='offline',
            language='en',label=i%2,noisy=False,audio=str(path),full_length=True,output_samples=length,view='full'))
    cfg=dict(ssl_path=str(processor),workers=0,eval_batch=2,microbatch=2,frame_budget=2400,
             seed=18,device='cpu',feature_free_margin_bytes=0,feature_commit_rows=1)
    return cfg,rows


class FeatureTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(49)

    def test_interrupted_initial_allocation_is_recovered_without_an_unowned_delete(self):
        from .features import _flush
        with tempfile.TemporaryDirectory() as directory:
            cfg, rows = fixture(directory)
            model = tiny_detector(checkpointing=False)
            out = Path(directory)/'features'
            with patch('w2v_v314.features._flush', side_effect=InterruptedError('initial allocation')):
                with self.assertRaises(InterruptedError):
                    extract_cache(model, rows, cfg, out, {'base_sha':'base'})
            self.assertTrue((out/'owner.json').exists())
            self.assertFalse((out/'cursor.json').exists())
            bundle = extract_cache(model, rows, cfg, out, {'base_sha':'base'})
            self.assertEqual(bundle['x'].shape, (3,32))
            for name in ('x', 'logits'):
                bundle[name]._mmap.close()

    def test_real_model_replay_full_length_order_and_completed_cache_no_forward(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows=fixture(d);model=tiny_detector(checkpointing=False,dropout=.2)
            out=Path(d)/'features';identity=dict(base_sha='base',split='train')
            bundle=extract_cache(model,rows,cfg,out,identity)
            self.assertEqual(bundle['x'].shape,(3,32))
            self.assertEqual([r['id'] for r in bundle['rows']],[r['id'] for r in rows])
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertFalse(model.training)
            x=torch.from_numpy(np.array(bundle['x']));final=model.head.classifier[-1]
            replay=torch.nn.functional.linear(x,final.weight,final.bias).detach().numpy()
            np.testing.assert_allclose(replay,bundle['logits'],rtol=1e-5,atol=2e-6)
            # Encoder receives each whole length, never pads a shorter recording
            # to the longer recording's length or chops a common prefix.
            seen=[]
            for batch in inference_batches(rows,cfg): seen.extend(r['features'].shape[1] for r in batch)
            self.assertGreater(seen[1],seen[0]);self.assertEqual(seen[0],seen[2])
            with patch.object(model,'forward',side_effect=AssertionError('completed cache must not run model')):
                second=extract_cache(model,rows,cfg,out,identity)
            np.testing.assert_array_equal(second['x'],bundle['x'])
            for item in (second,bundle):
                item['x']._mmap.close();item['logits']._mmap.close()

    def test_interrupted_cursor_resumes_only_uncommitted_rows(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows=fixture(d);cfg['eval_batch']=1;model=tiny_detector(checkpointing=False)
            out=Path(d)/'features';identity=dict(base_sha='base',split='train')
            original=model.forward; calls=[]
            def interrupted(*args):
                calls.append(1)
                if len(calls)==2:raise RuntimeError('fixture interruption')
                return original(*args)
            with patch.object(model,'forward',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'fixture interruption'):
                    extract_cache(model,rows,cfg,out,identity)
            self.assertFalse((out/'complete.json').exists())
            self.assertEqual(json.loads((out/'cursor.json').read_text())['cursor'],1)
            with self.assertRaises(FileNotFoundError):load_cache(out)
            calls.clear()
            def counted(*args):calls.append(1);return original(*args)
            with patch.object(model,'forward',side_effect=counted):
                resumed=extract_cache(model,rows,cfg,out,identity)
            self.assertEqual(len(calls),2)
            clean=extract_cache(model,rows,cfg,Path(d)/'clean',identity)
            np.testing.assert_array_equal(resumed['x'],clean['x'])
            np.testing.assert_array_equal(resumed['logits'],clean['logits'])
            for item in (resumed,clean):
                item['x']._mmap.close();item['logits']._mmap.close()

    def test_completed_byte_changes_identity_changes_and_unowned_outputs_fail(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows=fixture(d);model=tiny_detector(checkpointing=False);out=Path(d)/'features'
            identity=dict(base_sha='base',split='train')
            saved=extract_cache(model,rows,cfg,out,identity)
            saved['x']._mmap.close();saved['logits']._mmap.close()
            with self.assertRaisesRegex(ValueError,'changed base'):
                extract_cache(model,rows,cfg,out,dict(base_sha='different',split='train'))
            x=np.load(out/'x.npy',mmap_mode='r+');x[0,0]+=1;x.flush();del x
            with self.assertRaisesRegex(ValueError,'changed'):load_cache(out,identity)
            unowned=Path(d)/'unowned';unowned.mkdir();(unowned/'keep.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError,'unowned'):extract_cache(model,rows,cfg,unowned,identity)
            self.assertEqual((unowned/'keep.txt').read_text(),'keep')

    def test_partial_bytes_changed_are_rejected_before_more_forward(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows=fixture(d);cfg['eval_batch']=1;model=tiny_detector(checkpointing=False);out=Path(d)/'features'
            original=model.forward;calls=[]
            def interrupted(*args):
                calls.append(1)
                if len(calls)==2:raise RuntimeError('stop')
                return original(*args)
            with patch.object(model,'forward',side_effect=interrupted):
                with self.assertRaises(RuntimeError):extract_cache(model,rows,cfg,out,{'base_sha':'base'})
            x=np.load(out/'x.npy',mmap_mode='r+');x[0,0]+=1;x.flush();del x
            with patch.object(model,'forward',side_effect=AssertionError('must validate first')):
                with self.assertRaisesRegex(ValueError,'bytes changed'):
                    extract_cache(model,rows,cfg,out,{'base_sha':'base'})

    def test_insufficient_space_and_duplicate_rows_fail_without_deleting(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows=fixture(d);model=tiny_detector(checkpointing=False);out=Path(d)/'features'
            with patch('w2v_v314.features.shutil.disk_usage',return_value=type('Disk',(),{'free':0})()):
                with self.assertRaisesRegex(OSError,'no existing data deleted'):
                    extract_cache(model,rows,cfg,out,{'base_sha':'base'})
            self.assertFalse((out/'owner.json').exists())
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                extract_cache(model,rows+[rows[0]],cfg,out,{'base_sha':'base'})

    def test_spawn_numpy_transport_matches_single_process_features(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows=fixture(d)
            direct=[r for batch in inference_batches(rows,cfg) for r in batch]
            cfg['workers']=1
            spawned=[r for batch in inference_batches(rows,cfg) for r in batch]
            self.assertEqual([r['id'] for r in direct],[r['id'] for r in spawned])
            for a,b in zip(direct,spawned):
                torch.testing.assert_close(a['features'],b['features'],rtol=0,atol=0)
                torch.testing.assert_close(a['mask'],b['mask'],rtol=0,atol=0)

    def test_commit_interval_batches_cursor_writes_and_replays_only_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows=fixture(d);cfg.update(eval_batch=1,feature_commit_rows=2)
            model=tiny_detector(checkpointing=False);original=model.forward;calls=[];out=Path(d)/'features'
            def interrupted(*args):
                calls.append(1)
                if len(calls)==3:raise RuntimeError('stop after committed interval')
                return original(*args)
            with patch.object(model,'forward',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'committed interval'):
                    extract_cache(model,rows,cfg,out,{'base_sha':'base'})
            state=json.loads((out/'cursor.json').read_text())
            self.assertEqual(state['cursor'],2);self.assertEqual(len(state['chunks']),1)
            calls.clear()
            def counted(*args):calls.append(1);return original(*args)
            with patch.object(model,'forward',side_effect=counted):
                result=extract_cache(model,rows,cfg,out,{'base_sha':'base'})
            self.assertEqual(len(calls),1)
            state=json.loads((out/'cursor.json').read_text())
            self.assertEqual(state['cursor'],3);self.assertEqual(len(state['chunks']),2)
            result['x']._mmap.close();result['logits']._mmap.close()


if __name__=='__main__':unittest.main()
