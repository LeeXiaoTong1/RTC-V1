from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import SeamlessM4TFeatureExtractor
from w2v_v3.step import predict

from . import cache,data
from .test_cache import fixture,FakeRTC


class DataTests(unittest.TestCase):
    def test_full_features_and_predict_keep_audio_beyond_four_seconds(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            SeamlessM4TFeatureExtractor().save_pretrained(cfg['ssl_path'])
            # More than four seconds, with a distinct tail, must survive both
            # training prepared transport and ordinary validation extraction.
            first=records['train'][0]
            wave=np.random.default_rng(1111).normal(0,.05,96013).astype(np.float32)
            wave[80000:]*=3
            sf.write(first['audio'],wave,16000,subtype='FLOAT')
            plan,*_=data.build_data(cfg)
            iterator=data.loader(plan,cfg,training=True,epoch=1,batches=[[0]])
            prepared=next(iter(iterator))
            offline=next(ex for ex in prepared if ex['condition']=='offline')
            self.assertEqual(offline['crop_samples'],len(wave))
            extractor=SeamlessM4TFeatureExtractor.from_pretrained(cfg['ssl_path'])
            expected=extractor([wave],sampling_rate=16000,padding=True,pad_to_multiple_of=2,return_attention_mask=True,return_tensors='pt')
            n=int(expected['attention_mask'][0].sum())
            self.assertGreater(n,250)
            torch.testing.assert_close(offline['features'],expected['input_features'][:,:n],rtol=0,atol=0)
            lengths=[]
            class Model:
                def __call__(self,features,mask):
                    lengths.extend(mask.sum(1).tolist())
                    return torch.zeros(len(features),2),torch.zeros(len(features),2)
            logits=predict(Model(),prepared,'cpu','none',2,2400)
            self.assertEqual(tuple(logits.shape),(4,2));self.assertIn(n,lengths)
            # Same complete waveform also survives the validation loader path.
            batch=next(iter(data.loader([first],cfg)))
            self.assertEqual(int(batch[0]['mask'].sum()),n)
            iterator.dataset.close()

    def test_unique_source_coverage_and_equal_group_budget(self):
        sources=[]
        for count,(language,label) in enumerate((('en',0),('en',1),('zh',0),('zh',1)),1):
            sources.extend(dict(id=f'{language}/{label}/{i}',source_id=f'{language}/{label}/{i}',language=language,
                label=label,missing_online=False) for i in range(count))
        plan=data.SourcePlan(sources,3,12)
        order=[i for b in plan.batches(1) for i in b]
        self.assertEqual(sorted(order),list(range(10)))
        self.assertEqual(order,[i for b in plan.batches(1) for i in b])
        for group,count in plan.group_counts.items(): self.assertAlmostEqual(count*plan.weights[group]/10,.25)

    def test_build_and_dataset_only_full_views_source_read_once(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            plan,validation,weights,counts,fingerprints=data.build_data(cfg)
            self.assertEqual(set(validation),{'clean','seen','heldout'})
            self.assertTrue(all(r['domain']=='online' for r in validation['clean']))
            self.assertEqual(len(validation['seen']),plan.source_count)
            cache.prepare_epoch(cfg,1)
            dataset=data.FullSourceDataset(plan.records,cfg,1)
            with patch.object(data,'read_wave',wraps=data.read_wave) as read:
                views=dataset[0]
                self.assertEqual(read.call_count,2) # One Offline and one official Online.
            self.assertEqual({v['condition'] for v in views},{'offline','online','noisy_a','noisy_b'})
            self.assertTrue(all(v['view']=='full' and v['view_weight']==1 for v in views))
            self.assertTrue(all(v['crop_start_sample']==0 for v in views))
            self.assertTrue(all('/epoch_001/' not in key.replace('\\','/') for key in fingerprints))
            dataset.close()

    def test_worker_padding_stays_inside_source_chunks_without_dropped_views(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d);cfg['source_chunk']=2
            SeamlessM4TFeatureExtractor().save_pretrained(cfg['ssl_path'])
            plan,*_=data.build_data(cfg)
            iterator=data.loader(plan,cfg,training=True,epoch=1,batches=[[0,1,2]])
            examples=next(iter(iterator))
            groups=[{plan.records[i]['source_id'] for i in (0,1)}, {plan.records[2]['source_id']}]
            visited=[]
            for indices,features,masks in examples.batches:
                sources={examples[i]['source_id'] for i in indices}
                self.assertTrue(any(sources<=group for group in groups))
                visited.extend(indices)
                self.assertEqual(features.untyped_storage().data_ptr(),examples.feature_storage.untyped_storage().data_ptr())
            self.assertEqual(sorted(visited),list(range(12)))
            iterator.dataset.close()

    def test_missing_online_does_not_drop_source(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cache,'DiverseRTC',FakeRTC):
            cfg,records=fixture(d)
            pairs=Path(cfg['train_pair_manifest'])
            lines=pairs.read_text().splitlines();omitted=lines[1].split(',')[1]
            lines[1]=lines[1].split(',')[0]+',';pairs.write_text('\n'.join(lines)+'\n')
            protocol=Path(cfg['train_protocol'])
            protocol.write_text('\n'.join(line for line in protocol.read_text().splitlines() if not line.startswith(omitted+' '))+'\n')
            plan,*_=data.build_data(cfg)
            self.assertEqual(plan.coverage()['missing_online'],1)
            self.assertEqual(plan.source_count,16)


if __name__=='__main__':unittest.main()
