"""Exact values/gradients and real spawned worker transport under bounded IPC."""
import copy
import os
import pickle
from pathlib import Path
import tempfile
import unittest
import torch
from torch.utils.data import Dataset, DataLoader
from w2v_v32.batching import PreparedBatch
from w2v_v32.model import microbatches
from .transport import PackedPreparedBatch, paired_worker_init

torch.set_num_threads(1)


def source_views(source):
    generator = torch.Generator().manual_seed(810 + source)
    rows=[]
    for c,condition in enumerate(('offline','online','noisy_a','noisy_b')):
        for view,n,weight in (('full',13+c+source%3,.7),('short',9+c,.3)):
            audible=torch.ones(n,dtype=torch.bool);audible[2:4]=False
            rows.append(dict(id=f'{source}/{condition}',source_id=f'offline/en/{source}.wav',
                language='en',condition=condition,view=view,view_weight=weight,
                label=source%2,noisy=condition.startswith('noisy'),
                features=torch.randn(1,n,160,generator=generator),mask=torch.ones(1,n,dtype=torch.long),
                audibility_mask=audible))
    return rows


def prepared(rows):
    return PreparedBatch(rows,list(microbatches(rows,4,1600)),4,1600)


class SourceDataset(Dataset):
    def __len__(self):return 160
    def __getitem__(self,index):return source_views(index)


def packed_collate(rows):
    return PackedPreparedBatch(prepared([view for source in rows for view in source]))


def low_fd_worker(index):
    paired_worker_init(index)
    import resource
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(96,hard),hard))


def tensor_storages(batch):
    tensors=[v for ex in batch for v in ex.values() if isinstance(v,torch.Tensor)]
    tensors += [v for _,f,m in batch.batches for v in (f,m)]
    return {v.untyped_storage().data_ptr() for v in tensors}


class TransportTests(unittest.TestCase):
    def assert_exact(self,a,b):
        self.assertEqual(len(a),len(b));self.assertEqual(a.size,b.size)
        for x,y in zip(a,b):
            self.assertEqual(set(x),set(y))
            for k in x:
                if isinstance(x[k],torch.Tensor):torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
                else:self.assertEqual(x[k],y[k])
        for (ia,fa,ma),(ib,fb,mb) in zip(a.batches,b.batches):
            self.assertEqual(ia,ib)
            torch.testing.assert_close(fa,fb,rtol=0,atol=0)
            torch.testing.assert_close(ma,mb,rtol=0,atol=0)

    def test_128_views_have_three_storages_and_exact_pickle_roundtrip(self):
        old=prepared([ex for i in range(16) for ex in source_views(i)])
        state=torch.get_rng_state().clone();packed=PackedPreparedBatch(old)
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
        self.assertGreater(len(tensor_storages(old)),350)
        self.assertEqual(len(tensor_storages(packed)),3)
        self.assert_exact(old,packed)
        returned=pickle.loads(pickle.dumps(packed))
        self.assert_exact(old,returned)
        self.assertEqual(len(tensor_storages(returned)),3)
        serialized=packed.__getstate__()
        self.assertNotIn('batches',serialized)
        self.assertEqual(sum(isinstance(v,torch.Tensor) for v in serialized.values()),3)
        self.assertFalse(any(isinstance(v,torch.Tensor) for ex in serialized['examples'] for v in ex.values()))

    def test_real_spawned_workers_preserve_every_source_and_value(self):
        loader=DataLoader(SourceDataset(),batch_size=16,num_workers=2,prefetch_factor=1,
                          multiprocessing_context='spawn',worker_init_fn=paired_worker_init,
                          collate_fn=packed_collate)
        seen=[]
        for index,batch in enumerate(loader):
            self.assertEqual(len(tensor_storages(batch)),3)
            self.assert_exact(prepared([ex for i in range(index*16,(index+1)*16) for ex in source_views(i)]),batch)
            seen.extend(int(ex['source_id'].split('/')[-1].split('.')[0]) for ex in batch if ex['condition']=='offline' and ex['view']=='full')
        self.assertEqual(seen,list(range(160)))

    def test_different_microbatch_request_regroups_exactly_after_unpack(self):
        old=prepared(source_views(0)+source_views(1))
        batch=pickle.loads(pickle.dumps(PackedPreparedBatch(old)))
        left=list(microbatches(old,3,37));right=list(microbatches(batch,3,37))
        self.assertEqual(len(left),len(right))
        for (ia,fa,ma),(ib,fb,mb) in zip(left,right):
            self.assertEqual(ia,ib)
            torch.testing.assert_close(fa,fb,rtol=0,atol=0)
            torch.testing.assert_close(ma,mb,rtol=0,atol=0)

    @unittest.skipUnless(os.name=='posix','Linux file descriptor limit regression')
    def test_linux_low_fd_workers_do_not_use_descriptor_transport(self):
        loader=DataLoader(SourceDataset(),batch_size=16,num_workers=2,prefetch_factor=1,
                          multiprocessing_context='spawn',worker_init_fn=low_fd_worker,
                          collate_fn=packed_collate)
        self.assertEqual(sum(len(batch) for batch in loader),1280)

    def test_worker_strategy_and_prefetch_are_local_runtime_settings(self):
        from unittest.mock import patch
        from .data import loader
        with patch('w2v_v33.transport.os.name','posix'),patch('torch.multiprocessing.get_all_sharing_strategies',return_value={'file_system','file_descriptor'}), \
             patch('torch.multiprocessing.set_sharing_strategy') as choose:
            paired_worker_init(0);choose.assert_called_once_with('file_system')
        cfg=dict(workers=6,device='cuda:0',seed=1,ssl_path='unused',prefetch_factor=2)
        stream=loader([],cfg,training=True)
        self.assertEqual(stream.prefetch_factor,1);self.assertEqual(stream.num_workers,6)
        self.assertTrue(stream.pin_memory);self.assertEqual(cfg['prefetch_factor'],2)

    def test_tiny_encoder_pair_loss_gradients_and_adam_are_unchanged(self):
        from .test_step import model
        from .step import supervised_step
        torch.manual_seed(527)
        a=model();b=copy.deepcopy(a)
        left=prepared(source_views(0)+source_views(1));right=PackedPreparedBatch(left)
        oa=torch.optim.AdamW(a.parameters(),lr=2e-6);ob=torch.optim.AdamW(b.parameters(),lr=2e-6)
        sa,za=supervised_step(a,left,oa,torch.ones(2),'cpu',amp='none',pair_weight=.02)
        sb,zb=supervised_step(b,right,ob,torch.ones(2),'cpu',amp='none',pair_weight=.02)
        torch.testing.assert_close(za,zb,rtol=0,atol=0)
        self.assertEqual(sa['loss'],sb['loss'])
        for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()):
            with self.subTest(name=name):
                torch.testing.assert_close(x,y,rtol=0,atol=0)
                if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
        for key,values in oa.state_dict()['state'].items():
            for name,value in values.items():torch.testing.assert_close(value,ob.state_dict()['state'][key][name],rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'Pinned host buffers require CUDA')
    def test_pin_rebinds_views_to_same_two_pinned_buffers(self):
        old=prepared(source_views(0)+source_views(1));batch=pickle.loads(pickle.dumps(PackedPreparedBatch(old)))
        self.assertIs(batch.pin_memory(),batch);self.assert_exact(old,batch)
        self.assertEqual(len(tensor_storages(batch)),3)
        self.assertTrue(all(f.is_pinned() and m.is_pinned() for _,f,m in batch.batches))
        self.assertTrue(all(ex['features'].is_pinned() and ex['mask'].is_pinned() for ex in batch))


if __name__=='__main__':unittest.main()
